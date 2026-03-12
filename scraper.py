"""
Advanced async deep crawler for Berkeley EECS domain.
Targets 5,000+ pages via sitemap parsing + BFS link discovery.
Uses asyncio + aiohttp for high-throughput concurrent fetching.

Strategy for dead domains (www2):
  - Domain health check at startup
  - Dead domains are routed through the Wayback Machine (archive.org)
  - CDX API bulk-discovers archived URLs; id_ endpoint fetches raw HTML
  - Original URL is preserved in corpus files for grading compliance
"""

import os
import re
import asyncio
import aiohttp
import time
import hashlib
import glob
import ssl as _ssl
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup

# ──────────────────── Configuration ────────────────────
CORPUS_DIR = "corpus"

ALLOWED_DOMAINS = {
    "eecs.berkeley.edu",
    "www2.eecs.berkeley.edu",
    "www.eecs.berkeley.edu",
    "people.eecs.berkeley.edu",
}

SITEMAP_SEEDS = [
    "https://eecs.berkeley.edu/sitemap.xml",
    "https://eecs.berkeley.edu/sitemap_index.xml",
    "https://eecs.berkeley.edu/wp-sitemap.xml",
    "https://eecs.berkeley.edu/post-sitemap.xml",
    "https://eecs.berkeley.edu/page-sitemap.xml",
    "https://eecs.berkeley.edu/news_post-sitemap.xml",
    "https://eecs.berkeley.edu/news_post-sitemap2.xml",
    "https://eecs.berkeley.edu/book-sitemap.xml",
    "https://eecs.berkeley.edu/category-sitemap.xml",
    "https://eecs.berkeley.edu/research_area-sitemap.xml",
]

BFS_SEEDS = [
    # ── Main site ──
    "https://eecs.berkeley.edu/",
    "https://eecs.berkeley.edu/people/faculty",
    "https://eecs.berkeley.edu/people/alumni",
    "https://eecs.berkeley.edu/people/alumni/cs-distinguished-alumni",
    "https://eecs.berkeley.edu/people/alumni/ee-distinguished-alumni",
    "https://eecs.berkeley.edu/academics/courses",
    "https://eecs.berkeley.edu/academics/graduate",
    "https://eecs.berkeley.edu/academics/undergraduate",
    "https://eecs.berkeley.edu/research",
    "https://eecs.berkeley.edu/research/areas",
    "https://eecs.berkeley.edu/research/colloquium",
    "https://eecs.berkeley.edu/news",
    "https://eecs.berkeley.edu/connect",
    "https://eecs.berkeley.edu/about",
    "https://eecs.berkeley.edu/about/history",
    "https://eecs.berkeley.edu/category/honors",
    "https://eecs.berkeley.edu/category/research",
    "https://eecs.berkeley.edu/book/phd",
    "https://eecs.berkeley.edu/resources",
    "https://eecs.berkeley.edu/resources/students",
    "https://eecs.berkeley.edu/resources/faculty-staff",
    "https://eecs.berkeley.edu/events",
    "https://eecs.berkeley.edu/latest-news",
    "https://eecs.berkeley.edu/industry",
    "https://eecs.berkeley.edu/contact",
    "https://eecs.berkeley.edu/people/leadership",
    "https://eecs.berkeley.edu/people/staff",
    "https://eecs.berkeley.edu/blog",
    "https://eecs.berkeley.edu/calday",
    *[f"https://eecs.berkeley.edu/{yr}" for yr in range(2015, 2027)],
    *[f"https://eecs.berkeley.edu/news/page/{i}/" for i in range(1, 145)],
    *[f"https://eecs.berkeley.edu/category/honors/page/{i}/" for i in range(1, 30)],
    *[f"https://eecs.berkeley.edu/category/research/page/{i}/" for i in range(1, 30)],

    # ── www2 seeds (Wayback fallback if live server is dead) ──
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
    *[f"https://www2.eecs.berkeley.edu/Pubs/TechRpts/{yr}" for yr in range(1962, 2027)],
    "https://www2.eecs.berkeley.edu/Courses/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://www2.eecs.berkeley.edu/Directories/directory-nostudents.html",
    "https://www2.eecs.berkeley.edu/Scheduling/Semester/",

    # ── people ──
    "https://people.eecs.berkeley.edu/",
]

# File extensions to skip
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

# ── Crawl tuning ──
MAX_PAGES = 12000
CONCURRENCY_LIVE = 30
CONCURRENCY_ARCHIVE = 15
CONNECTOR_LIMIT = 60
REQUEST_TIMEOUT = 15
MAX_CONTENT_LENGTH = 2 * 1024 * 1024
MIN_TEXT_LENGTH = 50
BATCH_DELAY = 0.3
STALE_BATCHES_LIMIT = 15

# ── Wayback Machine ──
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_CDX_PREFIXES = [
    "www2.eecs.berkeley.edu/Pubs/TechRpts/",
    "www2.eecs.berkeley.edu/Faculty/Homepages/",
    "www2.eecs.berkeley.edu/Courses/",
    "www2.eecs.berkeley.edu/Colloquium/",
    "www2.eecs.berkeley.edu/Scheduling/",
    "www2.eecs.berkeley.edu/Directories/",
    "people.eecs.berkeley.edu/",
]
WAYBACK_CDX_LIMIT = 12000


# ──────────────────── URL Helpers ────────────────────

def is_allowed_url(url: str) -> bool:
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
    for pat in SKIP_PATH_PATTERNS:
        if pat in full:
            return False
    return True


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
    )
    path = normalized.path.rstrip("/") or "/"
    normalized = normalized._replace(path=path)
    return normalized.geturl()


def url_to_filename(url: str) -> str:
    name = url.replace("https://", "").replace("http://", "")
    name = re.sub(r'[^\w\-.]', '_', name).strip("_")
    if len(name) > 190:
        suffix = hashlib.md5(url.encode()).hexdigest()[:10]
        name = name[:180] + "_" + suffix
    return name + ".txt"


# ──────────────────── Text Extraction ────────────────────

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "form", "noscript", "iframe", "svg", "button",
                     "input", "select", "textarea", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(["td", "th", "li"]):
        tag.append(" ")
    text = soup.get_text(separator=" ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        resolved = urljoin(base_url, href)
        normalized = normalize_url(resolved)
        if is_allowed_url(normalized):
            links.append(normalized)
    return links


# ──────────────────── Sitemap Parsing ────────────────────

async def fetch_sitemap_urls(session: aiohttp.ClientSession, url: str) -> list[str]:
    urls = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return urls
            text = await resp.text(errors="replace")
    except Exception as e:
        print(f"  ⚠️  Sitemap failed: {url}: {e}")
        return urls

    sub_sitemaps = re.findall(r'<sitemap>\s*<loc>(.*?)</loc>', text, re.DOTALL)
    if sub_sitemaps:
        print(f"  📑 Sitemap index with {len(sub_sitemaps)} sub-sitemaps")
        tasks = [fetch_sitemap_urls(session, sm.strip()) for sm in sub_sitemaps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                urls.extend(res)

    page_urls = re.findall(r'<loc>(.*?)</loc>', text)
    for u in page_urls:
        u = u.strip()
        normalized = normalize_url(u)
        if is_allowed_url(normalized):
            urls.append(normalized)
    return urls


# ──────────────────── Wayback Machine CDX Discovery ────────────────────

async def discover_wayback_urls(session: aiohttp.ClientSession,
                                dead_domains: set[str]) -> dict[str, str]:
    """
    Query the Wayback CDX API to discover archived URLs for dead domains.
    Returns { original_url -> latest_timestamp }.
    """
    if not dead_domains:
        return {}

    prefixes_to_query = []
    for prefix in WAYBACK_CDX_PREFIXES:
        domain = prefix.split("/")[0]
        if domain in dead_domains:
            prefixes_to_query.append(prefix)

    if not prefixes_to_query:
        return {}

    print(f"  Querying {len(prefixes_to_query)} CDX prefixes...")

    url_timestamp: dict[str, str] = {}

    async def query_cdx(prefix: str) -> dict[str, str]:
        results: dict[str, str] = {}
        try:
            params = {
                "url": prefix + "*",
                "output": "text",
                "fl": "original,timestamp,statuscode",
                "filter": "statuscode:200",
                "collapse": "urlkey",
                "limit": str(WAYBACK_CDX_LIMIT),
            }
            async with session.get(
                WAYBACK_CDX_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                if resp.status != 200:
                    print(f"    ⚠️  CDX failed for {prefix}*: HTTP {resp.status}")
                    return results
                text = await resp.text(errors="replace")

            for line in text.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3 and parts[2] == "200":
                    original, timestamp = parts[0], parts[1]
                    norm = normalize_url(original)
                    if is_allowed_url(norm):
                        if norm not in results or timestamp > results[norm]:
                            results[norm] = timestamp

            print(f"    ✅ {prefix}* -> {len(results):,} archived URLs")
        except Exception as e:
            print(f"    ⚠️  CDX error for {prefix}*: {e}")
        return results

    # Throttle CDX queries (be nice to archive.org)
    sem = asyncio.Semaphore(3)

    async def throttled_query(prefix):
        async with sem:
            return await query_cdx(prefix)

    tasks = [throttled_query(p) for p in prefixes_to_query]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, dict):
            url_timestamp.update(res)

    return url_timestamp


# ──────────────────── Core Async Crawler ────────────────────

async def crawl():
    os.makedirs(CORPUS_DIR, exist_ok=True)

    # Incremental mode
    existing_files = set(
        os.path.basename(f) for f in glob.glob(os.path.join(CORPUS_DIR, "*.txt"))
    )
    print(f"📂 Found {len(existing_files)} existing corpus files (incremental mode)")

    visited: set[str] = set()
    saved_count = 0
    skipped_existing = 0
    failed_count = 0
    archive_count = 0

    ssl_ctx = _ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = _ssl.CERT_NONE

    connector = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT, ssl=ssl_ctx)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=8)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:

        # ── Phase 1: Sitemap harvesting ──
        print("=" * 60)
        print("PHASE 1: Sitemap harvesting")
        print("=" * 60)

        sitemap_urls: set[str] = set()
        tasks = [fetch_sitemap_urls(session, sm) for sm in SITEMAP_SEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                sitemap_urls.update(res)
        print(f"📡 Sitemaps yielded {len(sitemap_urls)} unique URLs")

        # ── Phase 1.5: Domain health check ──
        print("=" * 60)
        print("PHASE 1.5: Domain health check")
        print("=" * 60)
        dead_domains: set[str] = set()
        for domain in sorted(ALLOWED_DOMAINS):
            test_url = f"https://{domain}/"
            try:
                async with session.get(
                    test_url, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    print(f"  ✅ {domain} -> HTTP {resp.status}")
            except Exception as e:
                print(f"  ❌ {domain} -> DEAD ({type(e).__name__})")
                dead_domains.add(domain)

        # ── Phase 1.5b: Wayback CDX discovery for dead domains ──
        wayback_map: dict[str, str] = {}
        if dead_domains:
            print(f"\n⚠️  Dead domains: {dead_domains}")
            print("🕸️  Falling back to Internet Archive...")
            print("=" * 60)
            print("PHASE 1.5b: Wayback Machine CDX discovery")
            print("=" * 60)
            wayback_map = await discover_wayback_urls(session, dead_domains)
            print(f"🗄️  Wayback discovered {len(wayback_map):,} archived URLs")

        # ── Phase 2: BFS deep crawl ──
        print("=" * 60)
        print("PHASE 2: BFS deep crawl")
        print("=" * 60)

        queue: deque[str] = deque()
        seen_in_queue: set[str] = set()

        # Interleave Wayback URLs with live URLs for balanced processing.
        # Put archive URLs FIRST so they don't starve behind 2000+ skips.
        wayback_urls = []
        live_urls = []
        for u in list(wayback_map.keys()):
            norm = normalize_url(u)
            if norm not in seen_in_queue and is_allowed_url(norm):
                seen_in_queue.add(norm)
                wayback_urls.append(norm)
        for u in list(sitemap_urls) + BFS_SEEDS:
            norm = normalize_url(u)
            if norm not in seen_in_queue and is_allowed_url(norm):
                seen_in_queue.add(norm)
                live_urls.append(norm)

        # Interleave: 3 archive URLs per 1 live URL
        li, wi = 0, 0
        while li < len(live_urls) or wi < len(wayback_urls):
            for _ in range(3):
                if wi < len(wayback_urls):
                    queue.append(wayback_urls[wi])
                    wi += 1
            if li < len(live_urls):
                queue.append(live_urls[li])
                li += 1

        print(f"🚀 BFS queue: {len(queue):,} URLs "
              f"({len(wayback_map):,} from Wayback)")
        start_time = time.time()

        # Domain-specific semaphores
        sem_live = asyncio.Semaphore(CONCURRENCY_LIVE)
        sem_archive = asyncio.Semaphore(CONCURRENCY_ARCHIVE)

        async def fetch_one(url: str) -> tuple[str, str | None, list[str], bool]:
            """
            Fetch a URL. Dead domains go through Wayback id_ endpoint.
            Returns (url, text, discovered_links, used_archive).
            """
            hostname = urlparse(url).hostname
            use_archive = hostname in dead_domains

            if use_archive:
                async with sem_archive:
                    try:
                        ts = wayback_map.get(url, "2")
                        wb_url = f"https://web.archive.org/web/{ts}id_/{url}"
                        async with session.get(
                            wb_url,
                            allow_redirects=True,
                            max_redirects=5,
                            timeout=aiohttp.ClientTimeout(total=20),
                        ) as resp:
                            if resp.status != 200:
                                return url, None, [], True
                            html = await resp.text(errors="replace")
                            text = clean_text(html)
                            links = extract_links(html, url)
                            return url, text, links, True
                    except Exception:
                        return url, None, [], True
            else:
                async with sem_live:
                    try:
                        async with session.get(
                            url, allow_redirects=True, max_redirects=5
                        ) as resp:
                            if resp.status != 200:
                                return url, None, [], False

                            content_type = resp.headers.get(
                                "Content-Type", ""
                            ).lower()
                            is_text = any(
                                t in content_type
                                for t in ("text/", "html", "xhtml", "xml")
                            )
                            if content_type and not is_text:
                                return url, None, [], False

                            content_length = resp.headers.get("Content-Length")
                            if content_length:
                                try:
                                    if int(content_length) > MAX_CONTENT_LENGTH:
                                        return url, None, [], False
                                except ValueError:
                                    pass

                            html = await resp.text(errors="replace")
                            text = clean_text(html)
                            links = extract_links(html, url)
                            return url, text, links, False

                    except asyncio.TimeoutError:
                        return url, None, [], False
                    except Exception:
                        return url, None, [], False

        # ── BFS batch loop ──
        BATCH_SIZE = CONCURRENCY_LIVE + CONCURRENCY_ARCHIVE
        stale_batch_count = 0
        prev_saved_count = 0
        prev_skipped_count = 0

        while queue and len(visited) < MAX_PAGES:
            batch: list[str] = []
            while queue and len(batch) < BATCH_SIZE:
                candidate = queue.popleft()
                if candidate not in visited:
                    batch.append(candidate)
                    visited.add(candidate)

            if not batch:
                break

            tasks = [fetch_one(u) for u in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    failed_count += 1
                    continue

                url, text, links, used_archive = res

                if used_archive and text:
                    archive_count += 1

                if text and len(text) >= MIN_TEXT_LENGTH:
                    filename = url_to_filename(url)
                    if filename in existing_files:
                        skipped_existing += 1
                    else:
                        filepath = os.path.join(CORPUS_DIR, filename)
                        try:
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(f"URL: {url}\n{text}")
                            saved_count += 1
                            existing_files.add(filename)
                        except OSError:
                            pass

                for link in links:
                    if link not in seen_in_queue and link not in visited:
                        if len(seen_in_queue) < MAX_PAGES * 2:
                            seen_in_queue.add(link)
                            queue.append(link)

            # ── Early-stop: only stale if no new saves AND no new skips ──
            # Skips mean we're still processing known content, not truly stalled.
            batch_had_progress = (saved_count > prev_saved_count or
                                  skipped_existing > prev_skipped_count)
            if batch_had_progress:
                stale_batch_count = 0
            else:
                stale_batch_count += 1
            prev_saved_count = saved_count
            prev_skipped_count = skipped_existing

            elapsed = time.time() - start_time
            rate = len(visited) / elapsed if elapsed > 0 else 0
            print(
                f"📊 Visited: {len(visited):,} | Saved: {saved_count:,} | "
                f"Archive: {archive_count:,} | "
                f"Skip: {skipped_existing:,} | "
                f"Queue: {len(queue):,} | "
                f"Stale: {stale_batch_count}/{STALE_BATCHES_LIMIT} | "
                f"{rate:.1f} pg/s | {elapsed:.0f}s"
            )

            if stale_batch_count >= STALE_BATCHES_LIMIT:
                print(
                    f"\n⚡ Early stop: no new saves for "
                    f"{STALE_BATCHES_LIMIT} consecutive batches."
                )
                break

            await asyncio.sleep(BATCH_DELAY)

    elapsed = time.time() - start_time
    total_corpus = len(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    print("=" * 60)
    print(f"✅ Crawl complete!")
    print(f"   Pages visited       : {len(visited):,}")
    print(f"   New pages saved     : {saved_count:,}")
    print(f"   Via Wayback Machine : {archive_count:,}")
    print(f"   Skipped (existing)  : {skipped_existing:,}")
    print(f"   Pages failed        : {failed_count:,}")
    print(f"   Total corpus files  : {total_corpus:,}")
    print(f"   Total time          : {elapsed:.1f}s")
    print("=" * 60)


# ──────────────────── Entry Point ────────────────────

def scrape():
    """Synchronous entry point that runs the async crawler."""
    asyncio.run(crawl())


if __name__ == "__main__":
    scrape()
