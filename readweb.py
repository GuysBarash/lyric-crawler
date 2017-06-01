#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Artists import *
import Artists
from multiprocessing import Queue, Pool, cpu_count
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import bs4
from selenium.webdriver.support.ui import WebDriverWait
from lxml import etree
from io import StringIO

global english
global q
english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
q = Queue()


def init_driver():
    driver = webdriver.Chrome()
    driver.wait = WebDriverWait(driver, 5)
    return driver


# Songs Extractors
def worker_init(qt, englisht):
    global q
    q = qt
    global english
    english = englisht
    print "Fork"


def worker(driver, url):
    print "Working: {}".format(url)
    try:
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        if len(soup.findAll("span", {"class": "artist_lyrics_text"})) < 1:
            print "Fail to load at : {}".format(url)
            return
        mydivs = soup.findAll("span", {"class": "artist_lyrics_text"})[0].contents
        songName = soup.find_all("h1", {"class": "artist_song_name_txt"})[-1].contents[0]
        words = []
        for item in mydivs:
            if type(item) is bs4.element.NavigableString:
                line_words = re.findall(r"[\w']+", item, re.UNICODE)
                line_words = [x for x in line_words if not bool(set(x).intersection(english))]
                words += line_words

        songName = unicode(songName)
        q.put([songName, words])
    except Exception as e:
        print "Fail to load at : {}".format(url)


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


def read_all_songs(urls):
    total = len(urls)

    driver = init_driver()
    for url in urls:
        worker(driver, url)

    driver.quit()
    return [total]


def handle(unit):
    print "Handling {}".format(unit[1])
    url = unit[0]
    fileName = unit[1] + '.txt'
    all_url = get_all_urls(url)
    [amount] = read_all_songs(all_url.values())
    try:
        f = file(fileName, 'w+')
        f.write("#Total: {}\n".format(amount))
        index = 0
        while not q.empty():
            [songName, words] = q.get()
            f.write("\n#NAME[{}]\n".format(index))
            f.write(songName.encode('utf8'))
            f.write('\n#WORDS\n')
            for i in range(len(words)):
                f.write(words[i].encode('utf8') + '\t')
                if (i + 1) % 10 == 0:
                    f.write('\n')
            index += 1

    except Exception as e:
        print "ERROR"
        raise e
    finally:
        f.write('\n')
        f.write('#FIN')
        print "FINNISH {}".format(unit[1])
        f.close()


if __name__ == "__main__":
    handle(Artist_Ariel_Zilber)

    q.close()
    # counter = 0
    # for key in all_url.keys():
    #     print u"[{:^3}][{:^35}]\t{}".format(counter, key, all_url[key])
    #     counter += 1
