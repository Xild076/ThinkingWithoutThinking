# for searching

try:
    import wikipedia
except Exception:  # pragma: no cover - optional dependency
    wikipedia = None

# for retrieving page content

try:
    import wikipediaapi
except Exception:  # pragma: no cover - optional dependency
    wikipediaapi = None

if wikipediaapi is not None:
    wikiagent = wikipediaapi.Wikipedia("ThinkingWithoutThinking (harlexkin@gmail.com)", 'en')
else:
    wikiagent = None

def search_wikipedia(query: str) -> str:
    """Search Wikipedia

    Args:
        query (str): query to search

    Returns:
        str: The top result for the query.
    """
    if wikipedia is None:
        return None
    results = wikipedia.search(query)
    return results[0] if results else None

def get_wikipedia_page_content(title: str) -> dict:
    """Gets wikipedia page content

    Args:
        title (str): Page title

    Returns:
        dict: Wikipedia page content with keys: {
            "title": title of the page,
            "summary": summary of the page,
            "content": full content of the page,
            "url": url of the page
        }
    """
    if title is None or wikiagent is None:
        return {
            "title": "N/A",
            "summary": "N/A",
            "content": "N/A",
            "url": "N/A"
        }
    page = wikiagent.page(title)
    title = page.title
    summary = page.summary
    url = page.fullurl
    content = page.text
    output = {
        "title": title,
        "summary": summary,
        "content": content,
        "url": url
    }
    return output
