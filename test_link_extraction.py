#!/usr/bin/env python3
"""
Test script to verify link extraction from Google News works correctly.
"""

import sys
import os
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, quote_plus
import xml.etree.ElementTree as ET

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_google_news_rss(keywords, amount=5):
    """Test Google News RSS feed."""
    try:
        search_query = '+'.join([quote_plus(k) for k in keywords])
        rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        logger.info(f"Trying RSS: {rss_url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        response = requests.get(rss_url, headers=headers, timeout=10)
        logger.info(f"RSS response status: {response.status_code}")
        
        if response.status_code != 200:
            return []
        
        root = ET.fromstring(response.content)
        results = []
        
        for item in root.findall('.//item'):
            link_elem = item.find('link')
            if link_elem is not None and link_elem.text:
                url = link_elem.text.strip()
                if url and url.startswith('http'):
                    results.append(url)
                    if len(results) >= amount:
                        break
        
        return results[:amount]
        
    except Exception as e:
        logger.error(f"RSS failed: {e}")
        return []


def test_google_news_html(keywords, amount=5):
    """Test HTML scraping."""
    try:
        search_query = '+'.join([quote_plus(k) for k in keywords])
        url = f"https://www.google.com/search?q={search_query}&gl=us&tbm=nws&num={amount}"
        logger.info(f"Trying HTML scraping: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        logger.info(f"HTML response status: {response.status_code}")
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, "html.parser")
        all_links = soup.find_all('a', href=True)
        logger.info(f"Found {len(all_links)} total links")
        
        # Show sample raw URLs
        sample = [link.get('href', '')[:100] for link in all_links[:10]]
        logger.info(f"Sample hrefs: {sample}")
        
        results = []
        for link in all_links:
            href = link.get('href', '')
            
            # Direct https links
            if href.startswith('https://') and 'google' not in href:
                results.append(href)
            # Google redirects
            elif href.startswith('/url?'):
                parsed = parse_qs(urlparse(href).query)
                if 'q' in parsed:
                    candidate = parsed['q'][0]
                    if candidate.startswith('http') and 'google' not in candidate:
                        results.append(candidate)
            
            if len(results) >= amount:
                break
        
        return results[:amount]
        
    except Exception as e:
        logger.error(f"HTML scraping failed: {e}")
        return []

def test_news_extraction():
    """Test extracting news links with various queries."""
    
    test_queries = [
        ['news', 'today'],
        ['breaking', 'news'],
        ['technology', 'news'],
    ]
    
    print("\n" + "="*80)
    print("TESTING GOOGLE NEWS LINK EXTRACTION")
    print("="*80 + "\n")
    
    for query_words in test_queries:
        print(f"\n{'─'*80}")
        print(f"Testing query: {' '.join(query_words)}")
        print(f"{'─'*80}")
        
        # Try RSS first
        print("\n[1] Trying RSS feed...")
        rss_links = test_google_news_rss(query_words, amount=5)
        if rss_links:
            print(f"✅ RSS SUCCESS: Found {len(rss_links)} links")
            for i, link in enumerate(rss_links, 1):
                print(f"   {i}. {link[:100]}...")
        else:
            print(f"❌ RSS FAILED")
        
        # Try HTML scraping
        print("\n[2] Trying HTML scraping...")
        html_links = test_google_news_html(query_words, amount=5)
        if html_links:
            print(f"✅ HTML SUCCESS: Found {len(html_links)} links")
            for i, link in enumerate(html_links, 1):
                print(f"   {i}. {link[:100]}...")
        else:
            print(f"❌ HTML FAILED")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_news_extraction()
