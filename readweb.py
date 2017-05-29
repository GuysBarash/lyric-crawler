#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Queue, Pool, cpu_count
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import bs4
from selenium.webdriver.support.ui import WebDriverWait
from lxml import etree
from io import StringIO, BytesIO

dag_nahash_page = 'http://shironet.mako.co.il/artist?type=works&lang=1&prfid=333'
english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def init_driver():
    driver = webdriver.Chrome()
    driver.wait = WebDriverWait(driver, 5)
    return driver


## Songs Extractors
def worker_init(qt, englisht):
    global q
    q = qt
    global english
    english = englisht
    print "Fork"


def worker(url):
    driver = init_driver()
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')

    mydivs = soup.findAll("span", {"class": "artist_lyrics_text"})[0].contents
    songName = soup.find_all("h1", {"class": "artist_song_name_txt"})[-1].contents[0]
    driver.quit()
    words = []
    for item in mydivs:
        if type(item) is bs4.element.NavigableString:
            line_words = re.findall(r"[\w']+", item, re.UNICODE)
            line_words = [x for x in line_words if not bool(set(x).intersection(english))]
            words += line_words

    songName = unicode(songName)
    q.put([songName, words])


## Songs url extranctors:
def get_urls_by_songs_page(url):
    table_tag = r'/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr/td[1]/table/tbody/tr/td[2]/table/tbody/tr[5]/td/table/tbody'
    urls = dict()
    driver = init_driver()
    driver.get(url)
    html = driver.page_source
    driver.quit()

    parser = etree.HTMLParser()
    root = etree.parse(StringIO(html), parser)
    song_elements = root.xpath(table_tag + '//*[contains(@class, \'artist_player_songlist\')]')

    soup = BeautifulSoup(root, 'lxml')
    mydivs = soup.findAll("a", {"class": "artist_player_songlist"})
    for song in mydivs:
        urls[song.contents[0]] = 'http://shironet.mako.co.il' + song.get_attribute_list('href')[0]
    return urls


# /html/body/table[2]/tbody/tr/td/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr/td[1]/table

if __name__ == "__main__":
    # driver = init_driver()
    # lookup(driver, "Selenium")
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--mute-audio")
    urls = get_urls_by_songs_page(dag_nahash_page)
    counter = 0
    for key in urls.keys():
        print u"[{:^3}][{:^35}]\t{}".format(counter, key, urls[key])
        counter += 1
        # q = Queue()
        # p = Pool(cpu_count(), initializer=worker_init, initargs=(q, english))
        # p.map(worker, urls)
        # p.terminate()
        #
        # counter = 0
        # while not q.empty():
        #     item = q.get()
        #     print u"[{}]\tTitle: {}".format(counter, item[0])
        #     counter += 1
