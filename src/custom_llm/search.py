from __future__ import annotations

from urllib.parse import quote_plus
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from .config import MAX_WEB_RESULTS, USER_AGENT


def web_search(query: str, max_results: int = MAX_WEB_RESULTS) -> List[Dict[str, str]]:
    providers = (
        _wikipedia_search,
        _instant_answer_search,
        _html_search_fallback,
    )

    for provider in providers:
        try:
            results = provider(query, max_results)
            if results:
                return results
        except Exception:
            continue

    return []


def _wikipedia_search(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": query,
        "limit": max_results,
        "namespace": "0",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 4:
        return []

    titles = payload[1] if isinstance(payload[1], list) else []
    snippets = payload[2] if isinstance(payload[2], list) else []
    links = payload[3] if isinstance(payload[3], list) else []

    results: List[Dict[str, str]] = []
    for i in range(min(len(titles), len(links), max_results)):
        title = str(titles[i]).strip()
        link = str(links[i]).strip()
        snippet = str(snippets[i]).strip() if i < len(snippets) else ""
        if title and link:
            results.append({"title": title, "link": link, "snippet": snippet})

    return results


def _instant_answer_search(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    data = response.json()

    results: List[Dict[str, str]] = []

    abstract = str(data.get("AbstractText", "")).strip()
    abstract_url = str(data.get("AbstractURL", "")).strip()
    heading = str(data.get("Heading", "")).strip() or "DuckDuckGo Instant Answer"
    if abstract:
        results.append({"title": heading, "link": abstract_url or "https://duckduckgo.com/", "snippet": abstract})

    topics = data.get("RelatedTopics", [])
    for topic in topics:
        if isinstance(topic, dict) and "Topics" in topic:
            nested = topic.get("Topics", [])
        else:
            nested = [topic]

        for item in nested:
            if not isinstance(item, dict):
                continue
            text = str(item.get("Text", "")).strip()
            link = str(item.get("FirstURL", "")).strip()
            if text and link:
                title = text.split(" - ", 1)[0]
                results.append({"title": title, "link": link, "snippet": text})
            if len(results) >= max_results:
                return results

    return results[:max_results]


def _html_search_fallback(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, params=params, headers=headers, timeout=12)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: List[Dict[str, str]] = []

    selectors = [
        "div.result",
        "article[data-testid='result']",
    ]
    nodes = []
    for selector in selectors:
        nodes.extend(soup.select(selector))

    for res in nodes:
        title_el = res.select_one("a.result__a") or res.select_one("a[data-testid='result-title-a']")
        snippet_el = res.select_one("a.result__snippet") or res.select_one("div.result__snippet")
        if not title_el:
            continue

        title = title_el.get_text(" ", strip=True)
        link = title_el.get("href", "").strip()
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        if title and link:
            results.append({"title": title, "link": link, "snippet": snippet})

        if len(results) >= max_results:
            break

    return results


def format_web_results(results: List[Dict[str, str]], query: str = "") -> str:
    if not results:
        if query.strip():
            encoded = quote_plus(query)
            return (
                "I could not fetch web results right now. Try these links:\n"
                f"- https://duckduckgo.com/?q={encoded}\n"
                f"- https://www.google.com/search?q={encoded}\n"
                f"- https://en.wikipedia.org/wiki/Special:Search?search={encoded}"
            )
        return "I could not find web results right now."

    lines = ["Here is what I found on the web:"]
    for idx, item in enumerate(results, start=1):
        lines.append(f"{idx}. {item['title']}")
        if item.get("snippet"):
            lines.append(f"   {item['snippet']}")
        lines.append(f"   Source: {item['link']}")
    return "\n".join(lines)
