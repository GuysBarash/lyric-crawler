#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from Artists import *
global english
global logger
english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


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
                    line_words = [x for x in line_words if not bool(set(x).intersection(english))]
                    words += line_words

            songName = unicode(songName)
            q.append([songName, words])
            break
        except Exception as e:
            logger.log_warning("Fail to load at : {}".format(url))
            logger.log_warning(str(e))
            raise e


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
    global q
    total = len(urls)
    driver = init_driver()
    for url in urls:
        while True:
            try:
                analyze_given_song_by_link(driver, url)
                break
            except Exception as e:
                logger.log_print("Rebooting driver")
                driver = init_driver()

    driver.quit()
    return [total]


def handle_prep(loggert):
    global logger
    logger = loggert
    logger.log_print('Fork {}'.format(value['Val']))


def handle(unit):
    global q
    q = []
    logger.log_print("Handling {}".format(unit[1]))
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
        for item in q:
            [songName, words] = item
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
        logger.log_error("ERROR")
        logger.log_error(str(e))
        raise e
    finally:
        f.write('#FIN')
        logger.log_print("FINNISH {}".format(unit[1]))
        f.close()


if __name__ == "__main__":
    global logger
    logger = Logger()
    u_sig = datetime.datetime.now().strftime("_%H%M_%d_%m_%Y")
    logger.initThread("Report{}.txt".format(u_sig))

    # for local_artist in locals().keys():
    #     if 'Artist_' in local_artist:
    #         handle(locals()[local_artist])

    all_units = []
    for local_artist in locals().keys():
        if 'Artist_' in local_artist:
            all_units.append(locals()[local_artist])

    p = Pool(processes=cpu_count(), initializer=handle_prep, initargs=(logger,))
    p.imap_unordered(handle, all_units)
    p.close()
    p.join()

    logger.log_close()
