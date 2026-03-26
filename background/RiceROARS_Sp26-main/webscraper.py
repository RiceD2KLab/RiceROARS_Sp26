import csv
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE = "https://ga.rice.edu"
INDEX_URL = "https://ga.rice.edu/programs-study/departments-programs/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RiceGA-PLO-Scraper/1.0)"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# get department pages from the main index (rice ga)
def collect_department_links():
    soup = get_soup(INDEX_URL)

    links = []
    for a in soup.select('a[href^="/programs-study/departments-programs/"]'):
        href = a.get("href")
        name = norm(a.get_text(strip=True))
        if not href or not name:
            continue
        links.append((name, urljoin(BASE, href)))

    seen = set()
    out = []
    for name, url in links:
        if url in seen:
            continue
        seen.add(url)
        out.append((name, url))
    return out


# on departments pages either click undergrad or grad and collect the underlying hyperlinks 
def extract_credential_links(container_html: str):
    soup = BeautifulSoup(container_html, "html.parser")

    creds = []
    for a in soup.select('a[href^="/programs-study/departments-programs/"]'):
        href = a.get("href")
        title = norm(a.get_text(" ", strip=True))
        if not href:
            continue
        creds.append((title, urljoin(BASE, href)))

    seen = set()
    out = []
    for t, u in creds:
        if u in seen:
            continue
        seen.add(u)
        out.append((t, u))
    return out

def container_text(container_html: str) -> str:
    return norm(BeautifulSoup(container_html, "html.parser").get_text(" ", strip=True))

def get_dept_credentials(page, dept_url: str):
    """
    Returns:
      {
        "undergraduate": {"links": [(title,url),...], "message": str|None},
        "graduate": {"links": [(title,url),...], "message": str|None},
      }
    """
    page.goto(dept_url, wait_until="networkidle")

    result = {
        "undergraduate": {"links": [], "message": None},
        "graduate": {"links": [], "message": None},
    }

    # undergrad based on html 
    try:
        page.click('a[href="#undergraduatetextcontainer"]', timeout=3000)
        page.wait_for_timeout(500)
        ug_html = page.inner_html("#undergraduatetextcontainer")
        ug_links = extract_credential_links(ug_html)
        result["undergraduate"]["links"] = ug_links
        if not ug_links:
            txt = container_text(ug_html)
            result["undergraduate"]["message"] = txt or None
    except PlaywrightTimeoutError:
        result["undergraduate"]["message"] = "Undergraduate tab not found"

    # graduate based on html
    try:
        page.click('a[href="#graduatetextcontainer"]', timeout=3000)
        page.wait_for_timeout(500)
        gr_html = page.inner_html("#graduatetextcontainer")
        gr_links = extract_credential_links(gr_html)
        result["graduate"]["links"] = gr_links
        if not gr_links:
            txt = container_text(gr_html)
            result["graduate"]["message"] = txt or None
    except PlaywrightTimeoutError:
        result["graduate"]["message"] = "Graduate tab not found"

    return result

# take PLOs from credential pages
def extract_plos_from_credential(cred_url: str):
    soup = get_soup(cred_url)

    # find the first heading containing "Program Learning Outcomes"
    header = None
    for tag in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        if "program learning outcomes" in tag.get_text(" ", strip=True).lower():
            header = tag
            break
    if not header:
        return None

    # list outcomes
    lst = header.find_next(["ol", "ul"])
    if lst:
        items = [norm(li.get_text(" ", strip=True)) for li in lst.find_all("li")]
        items = [x for x in items if x]
        if items:
            return items

    # paragraph outcomes until next heading
    outcomes = []
    for el in header.find_all_next(["p", "h2", "h3", "h4"], limit=40):
        if el.name in ["h2", "h3", "h4"]:
            break
        t = norm(el.get_text(" ", strip=True))
        if t:
            outcomes.append(t)

    return outcomes or None

# main 
def main():
    departments = collect_department_links()
    print(f"Collected {len(departments)} department pages.")

    rows = []
    seen_cred_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, (dept_name, dept_url) in enumerate(departments, start=1):
            time.sleep(0.15)

            try:
                creds = get_dept_credentials(page, dept_url)
            except Exception as e:
                print(f"[{i}/{len(departments)}] Dept FAIL {dept_name}: {e}")
                continue

            ug_links = creds["undergraduate"]["links"]
            gr_links = creds["graduate"]["links"]

            print(f"[{i}/{len(departments)}] {dept_name}: UG={len(ug_links)} GR={len(gr_links)}")

            for level in ["undergraduate", "graduate"]:
                for cred_title, cred_url in creds[level]["links"]:
                    if cred_url in seen_cred_urls:
                        continue
                    seen_cred_urls.add(cred_url)

                    time.sleep(0.15)
                    try:
                        plos = extract_plos_from_credential(cred_url)
                    except Exception:
                        continue

                    # save even if no PLOs, prune that later
                    rows.append({
                        "department_name": dept_name,
                        "department_url": dept_url,
                        "level": level,
                        "credential_title": cred_title,
                        "credential_url": cred_url,
                        "plo_count": 0 if not plos else len(plos),
                        "plos_joined": "" if not plos else " | ".join(plos),
                        "plos_list": "" if not plos else str(plos),
                    })

        browser.close()

    out_file = "rice_ga_credential_plos.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "department_name", "department_url", "level",
                "credential_title", "credential_url",
                "plo_count", "plos_joined", "plos_list"
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()
