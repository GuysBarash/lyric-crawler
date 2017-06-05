#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Artists import *
from multiprocessing import Queue, Pool, cpu_count
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import bs4
from selenium.webdriver.support.ui import WebDriverWait
from lxml import etree
from io import StringIO
from Logger import Logger

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
    global songs_counter
    songs_counter += 1
    logger.log_print("[{:>5}] Working: {}".format(songs_counter, url))
    while True:
        try:
            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            if len(soup.findAll("span", {"class": "artist_lyrics_text"})) < 1:
                logger.log_print("Fail to load at : {}".format(url))
                continue
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
            break
        except Exception as e:
            logger.log_print("Fail to load at : {}".format(url))


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
    artist_hebrew_name = soup.find('p', {'id': 'breadcrumbs'}).find_all('a')[-2].text
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
    return [urls, artist_hebrew_name]


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

    while True:
        try:
            [all_url, hebrew_name] = get_all_urls(url)
            if len(all_url) < 1:
                logger.log_print("Connection Fail for {}. Re-attempting".format(unit[1]))
            else:
                break
        except Exception as e:
            logger.log_print("Connection Fail for {}. Re-attempting".format(unit[1]))
        finally:
            pass

    [amount] = read_all_songs(all_url.values())
    try:
        f = file(fileName, 'w+')
        f.write("#Total: {}\n".format(amount))
        f.write('#Artist\n')
        f.write(hebrew_name.encode('utf8') + '\n')
        index = 0
        while not q.empty():
            [songName, words] = q.get()
            f.write("#NAME[{}]\n".format(index))
            f.write(songName.encode('utf8'))
            f.write('\n#DONE_NAME')
            f.write('\n#WORDS\n')
            for i in range(len(words)):
                f.write(words[i].encode('utf8') + '\t')
                if (i + 1) % 10 == 0:
                    f.write('\n')
            index += 1
            f.write('\n#DONE_WORDS\n')

    except Exception as e:
        logger.log_print("ERROR")
        raise e
    finally:
        f.write('#FIN')
        logger.log_print("FINNISH {}".format(unit[1]))
        f.close()


if __name__ == "__main__":
    global logger
    global songs_counter
    songs_counter = 0

    logger = Logger()
    logger.initThread("Report.txt")

    for local_artist in locals().keys():
        if 'Artist_' in local_artist:
            handle(locals()[local_artist])

    logger.log_close()
    q.close()
