import random
import urllib.request
import requests
from scrapy.selector import Selector
from bs4 import BeautifulSoup


searchWord = "ronaldo"
bing = "https://www.bing.com/images/search?q=" + searchWord + "&FORM=HDRSC2"
r = None


def fetch_page(url):
    global r
    r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
    html = r.text
    soup = BeautifulSoup(html, 'lxml')
    return soup.prettify('utf-8')


def img_link_from_url(url):
    html = fetch_page(url)
    sel = Selector(text=html)
    img_links = sel.css('.row .item .thumb::attr(href)').extract()
    return img_links


def img_from_link(url):
    name = random.randrange(1, 1001)
    full_name = "../original_test_image/Cristiano Ronaldo/" + str(name) + ".jpg"
    if r.status_code == 200:
        urllib.request.urlretrieve(url, full_name)


for link in img_link_from_url(bing):
    img_from_link(link)
