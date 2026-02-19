USER_AGENT = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"}
MAX_HTML_BYTES = 1_500_000
MAX_TEXT_CHARS = 180_000
SUPPORTED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

try:
    import ddgs as ddgs
except Exception:
    try:
        import duckduckgo_search as ddgs
    except Exception:
        ddgs = None
if ddgs is not None:
    try:
        ddgs_agent = ddgs.DDGS(headers=USER_AGENT)
    except TypeError:
        ddgs_agent = ddgs.DDGS()
else:
    ddgs_agent = None

import ssl

# SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import bs4
import requests
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def get_links_from_duckduckgo(
    query: str,
    region: str = "us",
    max_results: int = 5,
    backend: str = "auto",
    safesearch: str = "moderate",
    retries: int = 2,
    debug: bool = False,
) -> list[str]:
    """Get links from DuckDuckGo search results.

    Args:
        query (str): The search query
        region (str): Region code for results (default: 'us')
        max_results (int): Maximum number of results to fetch (default: 5)
        backend (str): Backend to use (auto/html/lite/bing)
        safesearch (str): 'on', 'moderate', or 'off'
        retries (int): Number of retry attempts on failure (default: 2)
        debug (bool): If True, prints debugging info

    Returns:
        list[str]: A list of links from the search results
    """
    def _call(with_backend: str):
        if ddgs is None:
            return None
        try:
            try:
                agent = ddgs.DDGS(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"})
            except TypeError:
                agent = ddgs.DDGS()
            return agent.text(query, region=region, max_results=max_results, backend=with_backend, safesearch=safesearch)
        except Exception:
            return None

    for attempt in range(retries + 1):
        if debug:
            print(f"get_links attempt={attempt} backend={backend}")
        results = _call(backend)
        if results:
            return [result.get("href") for result in results if result.get("href")]

    for fb in ("html", "lite", "bing"):
        if fb == backend:
            continue
        if debug:
            print(f"get_links fallback backend={fb}")
        results = _call(fb)
        if results:
            return [result.get("href") for result in results if result.get("href")]

    return []

def get_site_details(url: str) -> dict:
    """Gets site details from url

    Args:
        url (str): url to fetch

    Returns:
        dict: {
            "title": title of the page,
            "text": text content of the page,
            "url": url of the page
        }
    """
    content_type = ""
    raw_text = ""
    try:
        with requests.get(
            url,
            headers=USER_AGENT,
            timeout=(5, 8),
            stream=True,
            allow_redirects=True,
        ) as r:
            r.raise_for_status()
            content_type = str(r.headers.get("Content-Type", "")).lower()
            if content_type and not any(marker in content_type for marker in SUPPORTED_CONTENT_TYPES):
                return {"title": "", "text": "", "url": url}

            chunks: list[bytes] = []
            total_bytes = 0
            for chunk in r.iter_content(chunk_size=16_384):
                if not chunk:
                    continue
                chunks.append(chunk)
                total_bytes += len(chunk)
                if total_bytes >= MAX_HTML_BYTES:
                    break

            raw_payload = b"".join(chunks)
            if not raw_payload:
                return {"title": "", "text": "", "url": url}

            encoding = r.encoding or "utf-8"
            raw_text = raw_payload.decode(encoding, errors="ignore")
    except Exception:
        return {"title": "", "text": "", "url": url}

    soup = bs4.BeautifulSoup(raw_text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    text = " ".join(soup.get_text(separator=" ").split())
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return {"title": title, "text": text, "url": url}

def get_site_details_many(urls: list[str]) -> list[dict]:
    """Gets many site details

    Args:
        urls (list[str]): list of urls to fetch

    Returns:
        list[dict]: List of dicts with: {
            "title": title of the page,
            "text": text content of the page,
            "url": url of the page
        }
    """
    if not urls:
        return []

    max_workers = min(6, len(urls))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(get_site_details, urls))
