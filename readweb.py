import re
from selenium import webdriver
from bs4 import BeautifulSoup
import bs4
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

my_url = r'http://shironet.mako.co.il/artist?type=lyrics&lang=1&prfid=166&wrkid=38'
url_shishi = r'http://shironet.mako.co.il/artist?type=lyrics&lang=1&prfid=333&wrkid=35824'


def init_driver():
    driver = webdriver.Chrome()
    driver.wait = WebDriverWait(driver, 5)
    return driver
def extract_lyric(url):
    driver = init_driver()
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    mydivs = soup.findAll("span", {"class": "artist_lyrics_text"})[0].contents
    driver.quit()
    words = []
    for item in mydivs:
        if type(item) is bs4.element.NavigableString:
            words += re.findall(r"[\w']+", item, re.UNICODE)
    return words


if __name__ == "__main__":
    # driver = init_driver()
    # lookup(driver, "Selenium")
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--mute-audio")
    words = extract_lyric(url_shishi)
    for word in words:
        print word
