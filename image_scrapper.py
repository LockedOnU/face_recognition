from bs4 import BeautifulSoup
import urllib.request
from selenium import webdriver
import random
import time
import os

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36")

driver = webdriver.Chrome('chromedriver', chrome_options=options)


def download_image():
    while True:
        keyword = input("데이터 확보를 종료하려면 exit를 입력하세요\n검색할 이미지를 입력하세요 : ")
        if keyword == 'exit':
            break
        if ' ' in keyword:
            url_keyword = keyword.split(' ')
            url_keyword = "%20".join(url_keyword)
        else:
            url_keyword = keyword
        url = 'https://www.bing.com/images/search?q=soccer%20player%20' + url_keyword + '&qs=n&form=QBIR&sp=-1&pq=cristiano%20ronaldo&sc=8-17&sk=&cvid=9859BB3984EB4F418553CF75774C3446'
        header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36"}
        down_counter = 0
        try:
            driver.get(url)
            i = 0
            while i < 2:
                i += 1
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
            # req_page = urllib.request.Request(url, headers=header)
            # html = urllib.request.urlopen(req_page)
            # page_soup = BeautifulSoup(html, 'lxml')
            page_soup = BeautifulSoup(driver.page_source, 'lxml')

            image_links = page_soup.find_all('a', attrs={'class': 'iusc'})
            print('찾은 이미지 링크 개수 : ' + str(image_links.__len__()))

            for image_link in image_links:
                image_link_href = image_link.get('href')
                if image_link_href is not None:
                    req_page = urllib.request.Request('https://www.bing.com' + image_link_href, headers=header)
                    html = urllib.request.urlopen(req_page)
                    page_soup = BeautifulSoup(html, 'lxml')
                    images = page_soup.find_all('img')

                    for original_image in images:
                        if original_image.get('data-reactid') is not None:
                            if '&w=60&h=60&c=7' not in original_image.get('src'):
                                original_image_src = original_image.get('src')
                                # print(original_image_src)
                                name = random.randrange(1, 1001)
                                if not os.path.isdir("../original_test_image/" + keyword + "/"):
                                    os.mkdir("../original_test_image/" + keyword + "/")
                                full_name = "../original_test_image/" + keyword + "/" + str(name) + ".jpg"
                                down_counter += 1

                            urllib.request.urlretrieve(original_image_src, full_name)
                else:
                    pass
        except urllib.request.HTTPError:
            print('error')
        print(down_counter)
