import requests
from bs4 import BeautifulSoup
import time

visited_urls = set()

def fetch_html(url):
    try:
        response = requests.get(url)
        return response.text
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def parse_html(html):
    return BeautifulSoup(html, 'html.parser')

def extract_links(soup, base_url):
    links = []
    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        if link.startswith('/'):
            link = base_url + link
        if link.startswith(base_url):
            links.append(link)
    return links

def scrape_website(url, base_url, depth):
    if depth > 5 or url in visited_urls:
        return
    visited_urls.add(url)
    html = fetch_html(url)
    if not html:
        return
    soup = parse_html(html)
    links = extract_links(soup, base_url)
    with open('scraped_data.txt', 'a', encoding='utf-8') as file:
        file.write(soup.get_text())
    for link in links:
        scrape_website(link, base_url, depth + 1)
        time.sleep(1)  # To avoid overloading the server

if __name__ == "__main__":
    base_url = 'https://docs.nvidia.com/cuda/'
    scrape_website(base_url, base_url, 0)