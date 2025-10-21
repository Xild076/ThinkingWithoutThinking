# Utility functions for the application

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import random
import threading
import requests
from urllib.parse import quote_plus, urlparse, parse_qs
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import trafilatura
import urllib.request
from newspaper import Article
import ssl
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Allow dynamic API key override (set by app.py)
_dynamic_api_key = None

def set_api_key(key: str):
    """Set API key dynamically (overrides environment variable)."""
    global _dynamic_api_key
    _dynamic_api_key = key if key and key.strip() else None

def get_api_key() -> str:
    """Get active API key (dynamic or environment)."""
    return _dynamic_api_key or api_key

GENAI_RATE_LIMIT_SECONDS = float(os.getenv("GENAI_RATE_LIMIT_INTERVAL", "3.0"))  # Increased from 1.5 to 3.0 seconds
_rate_limit_lock = threading.Lock()
_last_generation_time = 0.0
_consecutive_rate_limit_errors = 0

# Token quota tracking (15K tokens per minute for input)
_token_usage_window = []  # List of (timestamp, token_count) tuples
_token_quota_lock = threading.Lock()
QUOTA_LIMIT_TOKENS_PER_MINUTE = 15000  # Google's quota for gemma-3-27b
QUOTA_SAFETY_MARGIN = 0.8  # Use only 80% of quota to be safe


def _enforce_rate_limit():
    global _last_generation_time
    if GENAI_RATE_LIMIT_SECONDS <= 0:
        return

    with _rate_limit_lock:
        now = time.monotonic()
        wait_time = GENAI_RATE_LIMIT_SECONDS - (now - _last_generation_time)
        if wait_time > 0:
            logger.info(
                "Rate limiting active. Sleeping for %.2f seconds before next generate_text call",
                wait_time,
            )
            time.sleep(wait_time)
        _last_generation_time = time.monotonic()


def _estimate_token_count(text: str) -> int:
    """Estimate token count for text. Rough approximation: 1 token ≈ 4 characters.
    
    Note: This is a conservative estimate. Actual tokenization may vary.
    For very long texts (>50K chars), we cap the estimate to prevent excessive waiting.
    """
    char_count = len(text)
    estimated_tokens = char_count // 4
    
    # Cap extremely large estimates - if text is >200K chars (~50K tokens),
    # we likely have context bloat. Log a warning.
    if estimated_tokens > 50000:
        logger.warning(
            f"Extremely large token estimate: {estimated_tokens} tokens "
            f"({char_count:,} chars). This may indicate context bloat. "
            f"Consider reducing prompt size."
        )
    
    return estimated_tokens


def _clean_old_token_records():
    """Remove token usage records older than 60 seconds."""
    global _token_usage_window
    cutoff_time = time.time() - 60.0
    _token_usage_window = [(ts, count) for ts, count in _token_usage_window if ts > cutoff_time]


def _get_tokens_used_in_last_minute() -> int:
    """Get total tokens used in the last 60 seconds."""
    _clean_old_token_records()
    return sum(count for _, count in _token_usage_window)


def _check_quota_and_wait(estimated_tokens: int):
    """Check if adding these tokens would exceed quota, wait if necessary."""
    with _token_quota_lock:
        _clean_old_token_records()
        current_usage = _get_tokens_used_in_last_minute()
        safe_limit = int(QUOTA_LIMIT_TOKENS_PER_MINUTE * QUOTA_SAFETY_MARGIN)
        
        # Log current quota status
        logger.info(
            f"Quota check: {current_usage}/{safe_limit} tokens used in last 60s. "
            f"Request needs ~{estimated_tokens} tokens. "
            f"Records in window: {len(_token_usage_window)}"
        )
        
        # If we have any records, show the age of the oldest one
        if _token_usage_window:
            oldest_age = time.time() - _token_usage_window[0][0]
            logger.debug(f"Oldest token record is {oldest_age:.1f}s old")
        
        if current_usage + estimated_tokens > safe_limit:
            # Calculate how long to wait for oldest tokens to age out
            if _token_usage_window:
                oldest_timestamp = _token_usage_window[0][0]
                wait_time = 60.0 - (time.time() - oldest_timestamp) + 1.0  # +1s buffer
                
                logger.warning(
                    f"Token quota protection: {current_usage}/{safe_limit} tokens used in last minute. "
                    f"Waiting {wait_time:.1f}s before next request (estimated {estimated_tokens} tokens). "
                    f"After wait, oldest tokens will age out."
                )
                time.sleep(wait_time)
                _clean_old_token_records()
                
                # Verify we have enough quota now
                new_usage = _get_tokens_used_in_last_minute()
                logger.info(f"After waiting, quota usage: {new_usage}/{safe_limit} tokens")
                
                # If still not enough, wait a full 60 seconds
                if new_usage + estimated_tokens > safe_limit:
                    logger.error(
                        f"CRITICAL: Still insufficient quota after waiting. "
                        f"Need {estimated_tokens} tokens but only {safe_limit - new_usage} available. "
                        f"Waiting full 60 seconds to reset quota window."
                    )
                    time.sleep(60.0)
                    _token_usage_window.clear()  # Clear all records
                    logger.info("Quota window fully reset")


def _record_token_usage(token_count: int):
    """Record token usage for quota tracking."""
    with _token_quota_lock:
        _token_usage_window.append((time.time(), token_count))
        _clean_old_token_records()
        current_total = _get_tokens_used_in_last_minute()
        logger.debug(f"Token usage recorded: +{token_count} tokens. Total in last minute: {current_total}/{QUOTA_LIMIT_TOKENS_PER_MINUTE}")



def generate_text(text, model='models/gemma-3-27b-it', temperature=0.7, max_tokens=None, verbose=False, max_retries=10):
    """Generate text using Google GenAI with exponential backoff on rate limit errors."""
    global _consecutive_rate_limit_errors
    
    # Add current date context to all LLM calls
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    text_with_context = f"[SYSTEM CONTEXT: Today's date is {current_date}]\n\n{text}"
    
    # Estimate input token count and check quota
    estimated_input_tokens = _estimate_token_count(text_with_context)
    
    # CRITICAL: Hard limit check - reject requests that are too large
    ABSOLUTE_MAX_TOKENS = 14000  # Leave buffer below 15K quota
    if estimated_input_tokens > ABSOLUTE_MAX_TOKENS:
        error_msg = (
            f"FATAL: Request exceeds absolute token limit! "
            f"Estimated {estimated_input_tokens:,} tokens, maximum is {ABSOLUTE_MAX_TOKENS:,}. "
            f"This request would immediately fail. Context size must be reduced at the source."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    _check_quota_and_wait(estimated_input_tokens)
    
    for attempt in range(max_retries):
        _enforce_rate_limit()
        logger.info(f"Generating text with model: {model} (attempt {attempt + 1}/{max_retries})")
        logger.debug(f"Input text length: {len(text_with_context)} characters (~{estimated_input_tokens} tokens)")
        
        config_kwargs = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        config = types.GenerateContentConfig(**config_kwargs)

        active_api_key = get_api_key()
        masked_key = (active_api_key[:6] + "…") if active_api_key else "<missing>"
        logger.debug("Using Google API key (masked): %s", masked_key)

        client = genai.Client(api_key=active_api_key)
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=text_with_context)],
                )
            ]
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            
            # Record successful token usage
            _record_token_usage(estimated_input_tokens)
            
            _consecutive_rate_limit_errors = 0
            logger.info("Text generation completed successfully")
            return response.parts[0].text
            
        except Exception as exc:
            error_str = str(exc)
            
            # Check for rate limit / quota errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                _consecutive_rate_limit_errors += 1
                
                # CRITICAL: Record token usage on FIRST failure only!
                # Google counts the input tokens even when rejecting the request
                # But retries don't consume additional tokens - it's the same request
                if attempt == 0:  # Only record on first attempt
                    _record_token_usage(estimated_input_tokens)
                    logger.warning(f"Recording {estimated_input_tokens} tokens from failed request (API counted them)")
                else:
                    logger.debug(f"Retry attempt {attempt + 1} - not recording tokens again")
                
                # Try to extract suggested retry delay from error
                retry_delay = None
                if "retry in" in error_str.lower() or "retrydelay" in error_str.lower():
                    import re
                    # Try to match "retry in 59.636s" or similar
                    match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
                    if match:
                        retry_delay = float(match.group(1))
                    # Also try to match retryDelay in JSON response
                    match = re.search(r"'retryDelay':\s*'(\d+)s'", error_str)
                    if match and retry_delay is None:
                        retry_delay = float(match.group(1))
                
                # Use suggested delay or exponential backoff with higher minimum
                if retry_delay is None:
                    # Start with 10 seconds minimum, exponential backoff, max 120 seconds
                    retry_delay = min(120, max(10, 2 ** (_consecutive_rate_limit_errors + 2)))
                else:
                    # Add buffer to suggested delay (10% extra)
                    retry_delay = retry_delay * 1.1
                
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit error (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay:.1f} seconds... Error: {error_str[:300]}"
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Rate limit error after {max_retries} attempts: {error_str}")
                    raise
            else:
                logger.error("GenAI generate_content failed: %s", exc)
                raise
        finally:
            client.close()
    
    raise RuntimeError(f"Failed to generate text after {max_retries} attempts")

def online_query_tool(query, article_num):
    """Fetch and summarize articles based on a query.
    
    CRITICAL: If fetching fails, returns explicit failure message instead of hallucinating data.
    This prevents the LLM from making up article content when web access is unavailable.
    """
    logger.info(f"Starting online query for: '{query}' with {article_num} articles")
    
    links = retrieve_links([query], amount=article_num)
    logger.info(f"Retrieved {len(links)} links for query")
    
    # Check if we have NO links - explicit failure mode
    if not links:
        logger.error(f"FETCH FAILURE: Could not retrieve any links for query: '{query}'")
        failure_message = (
            f"⚠️ WEB FETCH FAILED: Could not retrieve search results for '{query}'\n\n"
            f"Possible reasons:\n"
            f"  • Network connectivity issue\n"
            f"  • Search service temporarily unavailable\n"
            f"  • Query may be blocked or rate-limited\n\n"
            f"ACTION: Please try a different query or rephrase your request. "
            f"Do NOT attempt to fabricate article content or sources."
        )
        logger.warning(f"Returning explicit failure message: {failure_message[:100]}...")
        return {
            'summary': failure_message,
            'articles': [],
            'fetch_status': 'FAILED',
            'error': 'No links retrieved'
        }
    
    articles = extract_many_article_details(links)
    logger.info(f"Successfully extracted {len(articles)} article details from {len(links)} links")
    
    # Check if extraction yielded NO valid articles despite having links
    article_texts = [article['text'] for article in articles if article and 'text' in article]
    if not article_texts:
        logger.error(f"EXTRACTION FAILURE: Retrieved {len(links)} links but could not extract article content")
        failure_message = (
            f"⚠️ EXTRACTION FAILED: Retrieved {len(links)} search results but could not extract article content\n\n"
            f"Possible reasons:\n"
            f"  • Articles are behind paywalls or access-restricted\n"
            f"  • Content format could not be parsed\n"
            f"  • Websites blocked automatic content extraction\n\n"
            f"Links found (but content unavailable):\n"
        )
        for i, link in enumerate(links[:5], 1):
            failure_message += f"  {i}. {link}\n"
        if len(links) > 5:
            failure_message += f"  ... and {len(links) - 5} more links\n"
        failure_message += (
            f"\nACTION: Try accessing these links directly, or use a different search query. "
            f"Do NOT make up article content based on the links."
        )
        logger.warning(f"Returning explicit extraction failure message")
        return {
            'summary': failure_message,
            'articles': links,  # Return links for reference but no content
            'fetch_status': 'EXTRACTION_FAILED',
            'error': 'Could not extract article content'
        }
    
    total_chars = sum(len(t) for t in article_texts)
    logger.debug(f"Total article text length: {total_chars:,} characters")
    
    # CRITICAL: Truncate articles to prevent quota exhaustion
    # Max 2000 chars per article, ~8K tokens total maximum
    MAX_CHARS_PER_ARTICLE = 2000
    MAX_TOTAL_CHARS = 32000  # ~8K tokens
    
    truncated_texts = []
    current_total = 0
    
    for text in article_texts:
        if current_total >= MAX_TOTAL_CHARS:
            logger.warning(f"Reached max total chars ({MAX_TOTAL_CHARS:,}). Skipping remaining {len(article_texts) - len(truncated_texts)} articles.")
            break
        
        if len(text) > MAX_CHARS_PER_ARTICLE:
            truncated = text[:MAX_CHARS_PER_ARTICLE] + "... [TRUNCATED]"
            logger.debug(f"Truncated article from {len(text):,} to {len(truncated):,} chars")
            truncated_texts.append(truncated)
            current_total += len(truncated)
        else:
            truncated_texts.append(text)
            current_total += len(text)
    
    logger.info(f"Prepared {len(truncated_texts)} articles for summarization ({current_total:,} chars, ~{current_total//4:,} tokens)")
    
    # Summarization prompt with explicit anti-hallucination instruction
    summary_prompt = f"""# Articles Retrieved Successfully:
{chr(10).join(truncated_texts)}

Your task: Summarize the above articles into a concise summary, highlighting key points and relevant information.

CRITICAL INSTRUCTION: 
- ONLY reference information explicitly present in the articles above
- Do NOT add any external knowledge or fill in gaps with assumptions
- If key information is missing, explicitly state "[INFORMATION UNAVAILABLE]" 
- Do NOT fabricate author names, dates, statistics, or other details not in the text"""
    
    summary = generate_text(summary_prompt)
    
    logger.info("Online query completed successfully")
    return {
        'summary': summary,
        'articles': articles,
        'fetch_status': 'SUCCESS',
        'error': None
    }


def evaluate_ideas_via_client(prompt_text: str, ideas: List[str], criteria: str, model: str = 'models/gemma-3-27b-it') -> Dict[str, Any]:
    """Evaluate a list of ideas using the GenAI client and return structured scores and the selected index.

    The function sends a compact evaluation prompt to the GenAI model with low temperature
    to produce deterministic judgments. It returns a dict like:
    {
      'scores': [int,...],
      'feedback': [str,...],
      'selected_index': int,
      'raw': <string response>
    }
    """
    # Build evaluation prompt
    ideas_text = "\n\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
    eval_prompt = (
        f"You are an impartial evaluator. Given the following IDEAS and EVALUATION CRITERIA, score each idea 0-100 and pick the single best idea.\n\n"
        f"EVALUATION CRITERIA:\n{criteria}\n\n"
        f"IDEAS:\n{ideas_text}\n\n"
        f"OUTPUT FORMAT (JSON ONLY):{{\n  \"scores\": [<int 0-100>, ...],\n  \"feedback\": [<short feedback for each idea>],\n  \"selected_index\": <index_of_best (0-based)>\n}}\n\n"
        f"Provide only the JSON. If you cannot parse, make your best effort to output valid JSON."
    )

    # Use a deterministic low temperature for evaluation
    active_api_key = get_api_key()
    client = genai.Client(api_key=active_api_key)
    try:
        config = types.GenerateContentConfig(temperature=0.15, max_output_tokens=800)
        contents = [
            types.Content(role="user", parts=[types.Part(text=eval_prompt)])
        ]
        response = client.models.generate_content(model=model, contents=contents, config=config)
        raw = response.parts[0].text

        # Try to extract JSON from raw text
        import re, json as _json
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = _json.loads(m.group(0))
                parsed['raw'] = raw
                return parsed
            except Exception:
                logger.warning("evaluate_ideas_via_client: JSON parse failed, returning raw response")
        # Fallback: return raw string with empty structure
        return {'scores': [], 'feedback': [], 'selected_index': 0, 'raw': raw}
    finally:
        try:
            client.close()
        except Exception:
            pass

def retrieve_links(keywords, amount=10):
    """Retrieve news article links from Google News based on keywords.
    
    DUAL STRATEGY: 
    1. Try Google News RSS (more reliable)
    2. Fallback to HTML scraping if RSS fails
    """
    logger.info(f"Starting link retrieval for keywords: {keywords}")
    logger.info(f"Requested amount: {amount}")
    
    # Strategy 1: Try Google News RSS first (more reliable)
    rss_results = _try_google_news_rss(keywords, amount)
    if rss_results:
        logger.info(f"Successfully retrieved {len(rss_results)} links from Google News RSS")
        return rss_results
    
    logger.info("RSS approach failed, falling back to HTML scraping...")
    
    # Strategy 2: Fallback to HTML scraping
    return _try_google_news_html_scraping(keywords, amount)


def _try_google_news_rss(keywords, amount=10):
    """Try to get news links from Google News RSS feed."""
    try:
        import xml.etree.ElementTree as ET
        
        # Google News RSS URL format
        search_query = '+'.join([quote_plus(k) for k in keywords])
        rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        logger.info(f"Trying Google News RSS: {rss_url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(rss_url, headers=headers, timeout=10)
        logger.info(f"RSS response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.warning(f"RSS feed returned status {response.status_code}")
            return []
        
        # Parse RSS XML
        root = ET.fromstring(response.content)
        
        # Find all item links in the RSS feed
        results = []
        for item in root.findall('.//item'):
            link_elem = item.find('link')
            if link_elem is not None and link_elem.text:
                url = link_elem.text.strip()
                
                # Google News RSS URLs are redirects - we need to resolve them to get actual URLs
                # Format: https://news.google.com/rss/articles/CBMi...
                # We'll try to extract the real URL by following the redirect
                if url and url.startswith('http'):
                    # Try to resolve the Google News redirect
                    real_url = _resolve_google_news_url(url, headers)
                    if real_url:
                        results.append(real_url)
                        logger.debug(f"Resolved RSS link: {real_url[:80]}...")
                    else:
                        # If we can't resolve, still add the Google News URL
                        results.append(url)
                        logger.debug(f"Using Google News URL: {url[:80]}...")
                    
                    if len(results) >= amount:
                        break
        
        logger.info(f"Extracted {len(results)} links from RSS feed")
        return results[:amount]
        
    except Exception as e:
        logger.warning(f"RSS parsing failed: {type(e).__name__}: {str(e)}")
        return []


def _resolve_google_news_url(google_news_url, headers):
    """Resolve a Google News redirect URL to get the actual article URL."""
    try:
        # Make a HEAD request to get the redirect without downloading content
        response = requests.head(google_news_url, headers=headers, allow_redirects=True, timeout=5)
        
        # The final URL after redirects is the actual article
        final_url = response.url
        
        # Validate it's a real external URL (not Google)
        if final_url and 'google' not in final_url and final_url.startswith('http'):
            return final_url
        
        return None
        
    except Exception as e:
        logger.debug(f"Could not resolve Google News URL: {e}")
        return None


def _try_google_news_html_scraping(keywords, amount=10):
    """Fallback: Try HTML scraping of Google News search results."""
    logger.info(f"HTML scraping fallback for keywords: {keywords}")
    
    headers_list = [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    ]
    
    search_query = '+'.join([quote_plus(k) for k in keywords])
    base_url = f"https://www.google.com/search?q={search_query}&gl=us&tbm=nws&num={amount}"
    logger.info(f"Constructed search URL: {base_url}")
    
    session = requests.Session()
    max_retries = 3
    
    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries}")
        
        header = random.choice(headers_list)
        logger.debug(f"Using User-Agent: {header['User-Agent'][:50]}...")
        
        try:
            logger.debug(f"Sending GET request to Google News...")
            response = session.get(base_url, headers=header, timeout=10)
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.warning(f"Non-200 status code received: {response.status_code}")
                time.sleep(1)
                continue
            
            logger.debug(f"Response content length: {len(response.content)} bytes")
            soup = BeautifulSoup(response.content, "html.parser")
            
            # COMPREHENSIVE EXTRACTION: Handle multiple URL formats
            all_links = soup.find_all('a', href=True)
            logger.info(f"Found {len(all_links)} total link elements in page")
            
            external_urls = set()  # Use set to deduplicate
            
            # Log first 10 raw URLs for debugging
            sample_urls = [link.get('href', '')[:150] for link in all_links[:10] if link.get('href')]
            logger.info(f"Sample raw hrefs from page: {sample_urls}")
            
            for link_elem in all_links:
                raw_url = link_elem.get('href', '').strip()
                
                if not raw_url:
                    continue
                
                extracted_url = None
                
                # Strategy 1: Direct https:// external links
                if raw_url.startswith('https://'):
                    # Skip Google's own domains
                    if any(domain in raw_url for domain in ['google.com', 'accounts.google', 'googleapis.com', 'gstatic.com']):
                        continue
                    # Skip social media
                    if any(domain in raw_url for domain in ['facebook.com', 'twitter.com', 'x.com', 'reddit.com', 'youtube.com', 'instagram.com', 'linkedin.com']):
                        continue
                    extracted_url = raw_url
                
                # Strategy 2: Google redirect URLs (/url?q=...)
                elif raw_url.startswith('/url?'):
                    try:
                        parsed_query = parse_qs(urlparse(raw_url).query)
                        if 'q' in parsed_query:
                            candidate = parsed_query['q'][0]
                            # Validate it's a real external URL
                            if candidate.startswith('http') and 'google' not in candidate:
                                extracted_url = candidate
                                logger.debug(f"Extracted from redirect: {candidate[:80]}...")
                    except Exception as e:
                        logger.debug(f"Could not parse redirect URL {raw_url[:100]}: {e}")
                
                # Strategy 3: Check for data-ved or other attributes that might contain real URLs
                elif raw_url.startswith('/search?') or raw_url.startswith('?'):
                    # These are internal Google search links, skip them
                    continue
                
                # If we extracted a valid URL, clean and add it
                if extracted_url:
                    try:
                        parsed = urlparse(extracted_url)
                        # Reconstruct clean URL
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if parsed.query:
                            clean_url += f"?{parsed.query}"
                        
                        if clean_url not in external_urls and len(clean_url) > 20:  # Sanity check
                            external_urls.add(clean_url)
                            logger.debug(f"Added external URL: {clean_url[:100]}...")
                    except Exception as e:
                        logger.debug(f"Error parsing URL {extracted_url}: {e}")
                        continue
            
            # Convert to list and take first N
            results = list(external_urls)[:amount]
            
            logger.info(f"Extracted {len(results)} unique external links")
            if results:
                logger.info(f"Successfully extracted URLs:")
                for i, url in enumerate(results[:5], 1):
                    logger.info(f"  {i}. {url[:80]}...")
                return results
            else:
                logger.warning(f"No external links found on attempt {attempt + 1}")
                logger.warning(f"This might indicate Google is blocking the request or HTML structure changed")
                
        except Exception as e:
            logger.error(f"Exception on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                logger.info("Retrying after 2 seconds...")
                time.sleep(2)
            continue
    
    logger.warning(f"Failed to retrieve links after {max_retries} attempts, returning empty list")
    return []

def extract_article_details(url:str):
    """Extract article details from a given URL."""
    logger.info(f"Extracting article details from: {url[:100]}...")
    
    if not url.startswith("http"):
        logger.warning(f"Invalid URL (doesn't start with http): {url}")
        return None

    BUDGET = 8.0
    start = time.monotonic()

    def remaining_budget():
        return max(0.1, BUDGET - (time.monotonic() - start))

    session = requests.Session()
    downloaded = None

    # Primary attempt: requests library
    try:
        logger.debug("Attempting primary extraction with requests library...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        timeout_val = min(5.0, remaining_budget())
        logger.debug(f"Request timeout: {timeout_val:.2f}s")
        
        response = session.get(url, headers=headers, timeout=timeout_val)
        response.raise_for_status()
        downloaded = response.text
        logger.info(f"Successfully downloaded HTML ({len(downloaded)} chars) with requests")
    except Exception as e:
        logger.warning(f"Primary download failed: {type(e).__name__}: {str(e)}")

    # Fallback 1: urllib
    if not downloaded and remaining_budget() > 0.2:
        try:
            logger.debug("Attempting fallback extraction with urllib...")
            timeout_val = min(4.0, remaining_budget())
            logger.debug(f"urllib timeout: {timeout_val:.2f}s")
            
            with urllib.request.urlopen(url, timeout=timeout_val) as r:
                downloaded = r.read().decode("utf-8", errors="ignore")
            logger.info(f"Successfully downloaded HTML ({len(downloaded)} chars) with urllib")
        except Exception as e:
            logger.warning(f"urllib download failed: {type(e).__name__}: {str(e)}")

    # Try trafilatura extraction
    if downloaded:
        try:
            logger.debug("Attempting trafilatura extraction...")
            meta = trafilatura.extract_metadata(downloaded, default_url=url)
            text = trafilatura.extract(downloaded)
            
            if meta and text:
                logger.info(f"Trafilatura extraction successful - Title: '{meta.title}', Text length: {len(text)} chars")
                return {
                    "source": urlparse(url).netloc.replace("www.", ""),
                    "title": meta.title or "",
                    "author": meta.author or "",
                    "date": str(getattr(meta, 'date', None)) if getattr(meta, 'date', None) else "",
                    "text": text,
                    "link": url
                }
            else:
                logger.warning("Trafilatura extraction returned None for meta or text")
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {type(e).__name__}: {str(e)}")

    # Fallback 2: newspaper3k
    if remaining_budget() > 0.2:
        try:
            logger.debug("Attempting newspaper3k extraction...")
            art = Article(url)
            
            if remaining_budget() < 1.0:
                logger.warning(f"Insufficient time budget remaining ({remaining_budget():.2f}s), skipping newspaper3k")
                return None
                
            art.download()
            art.parse()
            
            title = getattr(art, 'title', '') or ""
            text = getattr(art, 'text', '')
            
            logger.info(f"Newspaper3k extraction successful - Title: '{title}', Text length: {len(text)} chars")
            
            return {
                "source": urlparse(url).netloc.replace("www.", ""),
                "title": title,
                "author": getattr(art, 'authors', []),
                "date": str(getattr(art, 'publish_date', None)) if getattr(art, 'publish_date', None) else "",
                "text": text,
                "link": url
            }
        except Exception as e:
            logger.warning(f"Newspaper3k extraction failed: {type(e).__name__}: {str(e)}")
            return None

    elapsed = time.monotonic() - start
    logger.error(f"All extraction methods failed for URL: {url[:100]} (time elapsed: {elapsed:.2f}s)")
    return None

def extract_many_article_details(urls):
    """Extract details from multiple article URLs."""
    logger.info(f"Starting batch extraction for {len(urls)} URLs")
    
    results = []
    for idx, url in enumerate(urls):
        logger.info(f"Processing URL {idx + 1}/{len(urls)}")
        details = extract_article_details(url)
        if details:
            results.append(details)
            logger.info(f"Successfully extracted article {idx + 1}: '{details.get('title', 'No title')[:60]}...'")
        else:
            logger.warning(f"Failed to extract article {idx + 1}")
    
    logger.info(f"Batch extraction complete: {len(results)}/{len(urls)} articles successfully extracted")
    return results

def python_exec_tool(code, install_packages=None, timeout=30, save_plots=True):
    """
    Execute Python code in a sandboxed environment with support for:
    - Data analysis (pandas, numpy, scipy, etc.)
    - Mathematical computations (sympy, etc.)
    - Data visualization (matplotlib, seaborn, plotly)
    - Automatic package installation
    - Plot/chart saving to temporary files
    
    Args:
        code (str): Python code to execute
        install_packages (list, optional): List of packages to install before execution
        timeout (int): Maximum execution time in seconds (default: 30)
        save_plots (bool): Whether to save matplotlib/seaborn plots to temp files
        
    Returns:
        dict: {
            'success': bool,
            'output': str,  # stdout/stderr output
            'result': Any,  # return value if code returns something
            'plots': list,  # list of paths to saved plot files
            'error': str,   # error message if failed
            'installed_packages': list  # packages that were installed
        }
    """
    logger.info("Starting python_exec_tool execution")
    logger.debug(f"Code length: {len(code)} characters")
    
    import subprocess
    import sys
    import tempfile
    import json
    from pathlib import Path
    import base64
    
    result = {
        'success': False,
        'output': '',
        'result': None,
        'plots': [],
        'error': None,
        'installed_packages': []
    }
    
    # Install packages if requested
    if install_packages:
        logger.info(f"Installing packages: {install_packages}")
        for package in install_packages:
            try:
                logger.debug(f"Installing {package}...")
                install_result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-q', package],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if install_result.returncode == 0:
                    result['installed_packages'].append(package)
                    logger.info(f"Successfully installed {package}")
                else:
                    logger.warning(f"Failed to install {package}: {install_result.stderr}")
            except Exception as e:
                logger.error(f"Error installing {package}: {str(e)}")
    
    # Create a temporary directory for plots and execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.debug(f"Created temporary directory: {temp_dir}")
        
        # Write user code to a file
        user_code_file = temp_path / "user_code.py"
        with open(user_code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Prepare the execution wrapper code
        wrapper_code = '''import sys
import os
import io
import json
import traceback
from pathlib import Path

# Redirect stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

# Setup plot saving if matplotlib is available
plots_saved = []
temp_dir = r"''' + temp_dir + '''"

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Override plt.show() to save instead
    _original_show = plt.show
    plot_counter = [0]
    
    def custom_show(*args, **kwargs):
        plot_counter[0] += 1
        plot_path = Path(temp_dir) / f"plot_{plot_counter[0]}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plots_saved.append(str(plot_path))
        plt.close('all')
    
    plt.show = custom_show
except ImportError:
    pass

# User's code execution
result_value = None
error_msg = None
success = False

try:
    # Read and execute the user's code
    with open(r"''' + str(user_code_file) + '''", 'r', encoding='utf-8') as f:
        user_code = f.read()
    
    exec_globals = {'__name__': '__main__'}
    exec_locals = exec_globals
    
    exec(user_code, exec_globals, exec_locals)
    
    # Try to capture a result if there's a 'result' variable
    if 'result' in exec_locals:
        result_value = exec_locals['result']
    
    success = True
    
except Exception as e:
    error_msg = traceback.format_exc()
    success = False

finally:
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Prepare output
    output_dict = {
        'success': success,
        'stdout': stdout_capture.getvalue(),
        'stderr': stderr_capture.getvalue(),
        'result': str(result_value) if result_value is not None else None,
        'plots': plots_saved,
        'error': error_msg
    }
    
    # Write result to a JSON file
    result_file = Path(temp_dir) / "execution_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)
'''
        
        # Write wrapper code to a temporary file
        wrapper_file = temp_path / "exec_wrapper.py"
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        logger.debug("Executing wrapped code...")
        
        try:
            # Execute the wrapper script
            exec_result = subprocess.run(
                [sys.executable, str(wrapper_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir
            )
            
            # Read the execution result
            result_file = temp_path / "execution_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    exec_output = json.load(f)
                
                stdout_text = exec_output.get('stdout') or ''
                stderr_text = exec_output.get('stderr') or ''

                result['success'] = bool(exec_output.get('success'))
                result['output'] = stdout_text + stderr_text
                result['result'] = exec_output.get('result')
                result['error'] = exec_output.get('error')
                
                # Copy plots to a persistent location if they exist
                plots = exec_output.get('plots') or []
                if plots:
                    persistent_plots_dir = Path(tempfile.gettempdir()) / "thinking_pipeline_plots"
                    persistent_plots_dir.mkdir(exist_ok=True)
                    
                    for plot_path in plots:
                        if Path(plot_path).exists():
                            import shutil
                            import time
                            # Create unique filename with timestamp
                            timestamp = int(time.time() * 1000)
                            plot_name = f"plot_{timestamp}_{Path(plot_path).name}"
                            persistent_path = persistent_plots_dir / plot_name
                            shutil.copy2(plot_path, persistent_path)
                            result['plots'].append(str(persistent_path))
                            logger.info(f"Saved plot to: {persistent_path}")
                
                if result['success']:
                    logger.info("Code execution completed successfully")
                else:
                    logger.warning(f"Code execution failed: {result['error']}")
            else:
                result['error'] = "Execution result file not found"
                result['output'] = exec_result.stdout + exec_result.stderr
                logger.error("Execution result file not created")
                
        except subprocess.TimeoutExpired:
            result['error'] = f"Execution timed out after {timeout} seconds"
            logger.error(result['error'])
        except Exception as e:
            result['error'] = f"Execution error: {type(e).__name__}: {str(e)}"
            logger.error(result['error'])
    
    return result
