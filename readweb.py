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

Artist_dag_nahash = 'http://shironet.mako.co.il/artist?type=works&lang=1&prfid=333'
Artist_Artzi = 'http://shironet.mako.co.il/artist?type=works&lang=1&prfid=975'
Artist_Aviv_Geffen = 'http://shironet.mako.co.il/artist?type=works&lang=1&prfid=34'
global english
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


# Input: an artist page with songs links
# Output: a list of all the pages with songs (page 1, page 2 , .. )
def get_all_urls(url):
    table_tag = r'/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr/td[1]/table/tbody/tr/td[2]/table/tbody/tr[5]/td/table'
    index_urls = []
    urls = dict()
    driver = init_driver()

    c_url = url
    driver.get(c_url)
    html = driver.page_source

    # Get 'next' button
    soup = bs4.BeautifulSoup(html, 'lxml')
    next_page = soup.find_all('a', {'class': 'artist_nav_bar'}, text=lambda (x): (x.find('>>') >= 0))

    # Keep current page
    index_urls.append(c_url)

    # Get all songs from current page
    parser = etree.HTMLParser()
    root = etree.parse(StringIO(html), parser)
    song_elements = root.xpath(table_tag + '//*[contains(@class, \'artist_player_songlist\')]')
    for song in song_elements:
        urls[song.text] = 'http://shironet.mako.co.il' + song.get('href')

    while len(next_page) > 0:
        c_url = r'http://shironet.mako.co.il' + next_page[0].get('href')
        index_urls.append(c_url)
        driver.get(c_url)
        html = driver.page_source
        soup = bs4.BeautifulSoup(html, 'lxml')
        next_page = soup.find_all('a', {'class': 'artist_nav_bar'}, text=lambda (x): (x.find('>>') >= 0))

        # Get all songs from current page
        parser = etree.HTMLParser()
        root = etree.parse(StringIO(html), parser)
        song_elements = root.xpath(table_tag + '//*[contains(@class, \'artist_player_songlist\')]')
        for song in song_elements:
            urls[song.text] = 'http://shironet.mako.co.il' + song.get('href')

    driver.quit()
    return urls


if __name__ == "__main__":
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--mute-audio")

    all_url = get_all_urls(Artist_Artzi)
    counter = 0
    for key in all_url.keys():
        print u"[{:^3}][{:^35}]\t{}".format(counter, key, all_url[key])
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
