"""
Lightweight Berkeley EECS crawler that uses only Python standard library modules.

Output format per file:
  URL: <source_url>
  <cleaned page text...>

This keeps dependency requirements small for autograder compatibility.
"""

import hashlib
import os
import re
import sys
import time
from collections import deque
from html import unescape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


CORPUS_DIR = "corpus"
USER_AGENT = "Mozilla/5.0 (compatible; CS288-RAG-Scraper/1.0)"
REQUEST_TIMEOUT = 15
MAX_CONTENT_LENGTH = 2 * 1024 * 1024
MIN_TEXT_LENGTH = 30
MAX_PAGES = 12000

ALLOWED_DOMAINS = {
    "eecs.berkeley.edu",
    "www2.eecs.berkeley.edu",
    "www.eecs.berkeley.edu",
    "people.eecs.berkeley.edu",
}

SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".zip", ".tar", ".gz",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".bib", ".ps", ".eps", ".dvi", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".eot",
}

SKIP_PATH_PATTERNS = [
    "/wp-admin", "/wp-login", "/feed", "/xmlrpc",
    "/wp-json", "/wp-content/uploads", "/wp-includes",
    "/calendar", "/cart", "/checkout", "/my-account",
    "?replytocom=", "/trackback", "/embed",
    "/tag/", "/comment-page-",
]

BFS_SEEDS = [
    "https://eecs.berkeley.edu/",
    "https://eecs.berkeley.edu/people/faculty",
    "https://eecs.berkeley.edu/people/alumni",
    "https://eecs.berkeley.edu/academics/courses",
    "https://eecs.berkeley.edu/academics/graduate",
    "https://eecs.berkeley.edu/academics/undergraduate",
    "https://eecs.berkeley.edu/research",
    "https://eecs.berkeley.edu/research/areas",
    "https://eecs.berkeley.edu/research/colloquium",
    "https://eecs.berkeley.edu/news",
    "https://eecs.berkeley.edu/contact",
    "https://eecs.berkeley.edu/resources",
    "https://eecs.berkeley.edu/people/leadership",
    "https://eecs.berkeley.edu/people/staff",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
    "https://www2.eecs.berkeley.edu/Courses/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://people.eecs.berkeley.edu/",
]


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
    )
    path = normalized.path.rstrip("/") or "/"
    normalized = normalized._replace(path=path)
    return normalized.geturl()


def is_allowed_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.hostname not in ALLOWED_DOMAINS:
        return False

    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False

    full = url.lower()
    for pattern in SKIP_PATH_PATTERNS:
        if pattern in full:
            return False

    return True


def url_to_filename(url):
    name = url.replace("https://", "").replace("http://", "")
    name = re.sub(r"[^\w\-.]", "_", name).strip("_")
    if len(name) > 190:
        suffix = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        name = name[:180] + "_" + suffix
    return name + ".txt"


class PageParser(HTMLParser):
    """Extract visible text and href links using stdlib HTML parser."""

    BLOCK_TAGS = {
        "p", "div", "section", "article", "br", "li", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "meta", "link"}

    def __init__(self, base_url):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._skip_depth = 0
        self._text_parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if tag in self.BLOCK_TAGS:
            self._text_parts.append(" ")

        if tag == "a":
            href = None
            for key, value in attrs:
                if key == "href":
                    href = value
                    break
            if not href:
                return
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                return
            resolved = normalize_url(urljoin(self.base_url, href))
            if is_allowed_url(resolved):
                self.links.append(resolved)

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self._text_parts.append(" ")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._text_parts.append(text)
            self._text_parts.append(" ")

    def text(self):
        raw = "".join(self._text_parts)
        normalized = re.sub(r"\s+", " ", unescape(raw)).strip()
        return normalized


def fetch_html(url):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return ""

            data = resp.read(MAX_CONTENT_LENGTH + 1)
            if len(data) > MAX_CONTENT_LENGTH:
                return ""

            charset = resp.headers.get_content_charset() or "utf-8"
            return data.decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, ValueError):
        return ""


def parse_page(html, base_url):
    parser = PageParser(base_url)
    try:
        parser.feed(html)
    except Exception:
        return "", []

    text = parser.text()
    # De-duplicate links while preserving order.
    seen = set()
    links = []
    for link in parser.links:
        if link not in seen:
            links.append(link)
            seen.add(link)
    return text, links


def save_page(url, text):
    filepath = os.path.join(CORPUS_DIR, url_to_filename(url))
    with open(filepath, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"URL: {url}\n")
        f.write(text)
    return filepath


def crawl(max_pages=MAX_PAGES):
    os.makedirs(CORPUS_DIR, exist_ok=True)

    queue = deque()
    queued = set()
    visited = set()
    saved_count = 0

    for seed in BFS_SEEDS:
        normalized = normalize_url(seed)
        if is_allowed_url(normalized) and normalized not in queued:
            queue.append(normalized)
            queued.add(normalized)

    while queue and saved_count < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        html = fetch_html(url)
        if not html:
            continue

        text, links = parse_page(html, url)
        if len(text) < MIN_TEXT_LENGTH:
            continue

        save_page(url, text)
        saved_count += 1

        for link in links:
            if link not in visited and link not in queued:
                queue.append(link)
                queued.add(link)

        if saved_count % 100 == 0:
            print(f"Saved {saved_count} pages | queue={len(queue)} | visited={len(visited)}")

    return saved_count


def main():
    max_pages = MAX_PAGES
    if len(sys.argv) > 1:
        try:
            max_pages = int(sys.argv[1])
        except ValueError:
            print("Invalid max_pages argument; using default.")

    start = time.time()
    count = crawl(max_pages=max_pages)
    elapsed = time.time() - start
    print(f"Done. Saved {count} pages to '{CORPUS_DIR}' in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
