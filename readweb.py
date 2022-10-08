#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import random
from multiprocessing import Queue, Pool, cpu_count
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import bs4
from selenium.webdriver.support.ui import WebDriverWait
from lxml import etree
from io import StringIO
from Logger import Logger
import datetime
from Artists import artist_list
import json
import os

global english
global hebrew
global logger
english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
hebrew = 'אבגדהוזחטיכלמנסעפצקרשת' + 'םןךףץ'


def init_driver():
    driver = webdriver.Chrome()
    driver.wait = WebDriverWait(driver, 5)
    return driver


# Songs Extractors
def analyze_given_song_by_link(driver, url):
    global q
    logger.log_print("Working: {}".format(url))
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
                    # line_words = [x for x in line_words if not bool(set(x).intersection(english))]

                    # Remove all non-hebrew words
                    line_words = [x for x in line_words if bool(set(x).intersection(hebrew))]

                    words += line_words

            # Convert song name to unicode
            songName = songName

            q.append([songName, words])
            break
        except Exception as e:
            logger.log_warning("Fail to load at : {}".format(url))
            logger.log_warning(str(e))
            raise e
    return len(words)


# Input: an artist page with songs links
# Output: a list of all the pages with songs (page 1, page 2 , .. )
def get_all_urls(url):
    table_tag = r'/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr/td[1]/table/tbody/tr/td[2]/table/tbody/tr[5]/td/table'
    index_urls = []
    urls = dict()
    driver = init_driver()

    c_url = url
    driver.get(c_url)

    # Wait for 1 second
    time_to_wait = random.randint(1, 3)
    time.sleep(time_to_wait)

    html = driver.page_source

    # Get 'next' button
    soup = bs4.BeautifulSoup(html, 'lxml')
    next_page = soup.find_all('a', {'class': 'artist_nav_bar'}, text=lambda x: x.find('>>') >= 0)
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
        next_page = soup.find_all('a', {'class': 'artist_nav_bar'}, text=lambda x: x.find('>>') >= 0)

        # Get all songs from current page
        parser = etree.HTMLParser()
        root = etree.parse(StringIO(html), parser)
        song_elements = root.xpath(table_tag + '//*[contains(@class, \'artist_player_songlist\')]')
        for song in song_elements:
            song_name = song.text.replace('\t', '')
            urls[song_name] = 'http://shironet.mako.co.il' + song.get('href')

    driver.quit()
    return [urls, artist_hebrew_name]


def read_all_songs(urls):
    global q
    total = len(urls)
    word_count = 0
    driver = init_driver()

    word_cap = 300000
    for url in urls:
        while True:
            try:
                word_count += analyze_given_song_by_link(driver, url)
                break
            except Exception as e:
                logger.log_print("Rebooting driver")
                driver = init_driver()
        if word_count > word_cap:
            logger.log_print("Reached {} words".format(word_cap))
            break

    driver.quit()
    return [total]


def handle_prep(loggert):
    global logger
    logger = loggert
    logger.log_print('Fork')


def handle(unit):
    global q
    q = []
    logger.log_print("Handling {}".format(unit[1]))
    url = unit[0]
    fileName = unit[1] + '.json'

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    index_path = os.path.join(root_path, 'index')
    index_fname = os.path.join(index_path, fileName)
    lyrics_fname = os.path.join(root_path, fileName)

    if os.path.exists(lyrics_fname):
        logger.log_print("Already exists. Skipping artist")
        return None

    if os.path.exists(index_fname):
        logger.log_print("Already exists at {}. Skipping mapping".format(index_fname))
        with open(index_fname, 'r', encoding='utf-8') as f:
            d = json.load(f)

            hebrew_name = d['artist']
            all_url = d['songs']
    else:
        while True:
            try:
                [all_url, hebrew_name] = get_all_urls(url)
                if len(all_url) < 1:
                    logger.log_print("Connection Fail for {}. Re-attempting".format(unit[1]))
                else:
                    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
                    index_path = os.path.join(root_path, 'index')
                    index_fname = os.path.join(index_path, fileName)

                    d = dict()
                    d['artist'] = hebrew_name
                    d['artist_key'] = unit[1]
                    d['index_page'] = unit[0]
                    d['songs_count'] = len(all_url)
                    d['songs'] = dict()
                    for song_key, song_url in all_url.items():
                        d['songs'][song_key.replace('\t', '')] = song_url

                    with open(index_fname, 'w', encoding='utf-8') as f:
                        json.dump(d, f, ensure_ascii=False, indent=4)

                    break
            except Exception as e:
                raise e
                logger.log_print("Connection Fail for {}. Re-attempting".format(unit[1]))
            finally:
                pass

    [amount] = read_all_songs(all_url.values())

    with open(lyrics_fname, 'w', encoding='utf-8') as f:
        d = dict()
        d['artist'] = hebrew_name
        d['songs_count'] = amount
        d_songs = dict()
        for item in q:
            d_songs[item[0]] = item[1]
        d['songs'] = d_songs
        json.dump(d, f, ensure_ascii=False, indent=4)

    logger.log_print("FINNISH {}".format(unit[1]))


if __name__ == "__main__":
    global logger
    logger = Logger()
    u_sig = datetime.datetime.now().strftime("_%H%M_%d_%m_%Y")
    logger.initThread("Report{}.txt".format(u_sig))

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    index_path = os.path.join(root_path, 'index')
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    single_thread = False
    if single_thread:
        print("Single Thread")
        handle_prep(logger)
        for unit in artist_list:
            handle(unit)
    else:
        print("Starting pool")
        p = Pool(processes=2, initializer=handle_prep, initargs=(logger,))
        p.imap_unordered(handle, artist_list)
        p.close()
        p.join()

    logger.log_close()
