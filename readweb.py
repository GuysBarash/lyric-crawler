#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import random
from multiprocessing import Queue, Pool, cpu_count
import re

import pandas as pd
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
import numpy as np
import requests

global english
global hebrew
global logger
english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
hebrew = 'אבגדהוזחטיכלמנסעפצקרשת' + 'םןךףץ'

from tqdm import tqdm

tqdm.pandas()


class WordNormalizer:
    def __init__(self):
        pass

    def normalize(self, word_list, top_n=None):
        if top_n is None:
            word_list = word_list[:top_n]
        request = {
            'token': 'lptwXKtDgLEzzjW',
            'readable': False,
            'words': word_list,
        }
        print("Sending request to Hebrew NLP (Word count: {})".format(len(word_list)))
        st = datetime.datetime.now()
        result = requests.post('https://hebrew-nlp.co.il/service/Morphology/Analyze', json=request).json()
        print("Got response from Hebrew NLP. Duration: {}".format(datetime.datetime.now() - st))

        res = pd.DataFrame(columns=['base', 'word'], index=range(len(word_list)))
        res['word'] = word_list
        res['base'] = [result[i][0]['baseWord'] for i in range(len(result))]
        return res

    def _normalize_load_from_memory(self, word_list, top_n=None):
        path = 'res.csv'
        if os.path.exists(path):
            res = pd.read_csv(path, index_col=0)
        else:
            res = self.normalize(word_list, top_n)
            res.to_csv(path, encoding='utf-8-sig')

        return res


def clean_song_list(l):
    lx = [lt for lt in l if bool(set(lt).intersection(hebrew))]

    patterns = ['^ו']
    patterns += ["^ת'"]
    patterns += [r'\\']
    for pattern in patterns:
        lx = [re.sub(pattern, '', lt) for lt in lx]
    lx = [lt for lt in lx if len(lt) > 1]
    return lx


def init_driver():
    driver = webdriver.Chrome()
    driver.wait = WebDriverWait(driver, 5)
    return driver


# Songs Extractors
def analyze_given_song_by_link(driver, url):
    global q
    logger.log_print("Working: {}".format(url))
    for i in range(6):
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
                    line_words = clean_song_list(line_words)
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
        success = False
        for i in range(10):
            try:
                word_count += analyze_given_song_by_link(driver, url)
                success = True
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

    section_path_creation = True
    if section_path_creation:
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        index_path = os.path.join(root_path, 'index')
        if not os.path.exists(index_path):
            os.makedirs(index_path)

        index_summary_path = os.path.join(root_path, 'index_summary.csv')
        words_hist_path = os.path.join(root_path, 'words_hist.csv')
        kaggle_path = os.path.join(root_path, 'kaggle.csv')

    section_crawling = False
    if section_crawling:
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

    section_summary = True
    if section_summary:
        print("Summary:")
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        index_path = os.path.join(root_path, 'index')

        index_summary = pd.DataFrame(columns=['artist', 'artist_key', 'index_page', 'songs_page', 'songs_count'])
        for file in os.listdir(index_path):
            if file.endswith(".json"):
                with open(os.path.join(index_path, file), 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    index_summary = pd.concat([index_summary, pd.DataFrame([d])], ignore_index=True)
        if 'songs' in index_summary.columns:
            index_summary.drop('songs', axis=1, inplace=True)

        index_summary['songs_crawled'] = -1
        index_summary['songs_failed'] = -1
        index_summary['empty_songs'] = -1
        index_summary['words_count'] = -1
        index_summary['unique_words_count'] = -1

        files_to_scan = [f for f in os.listdir(root_path) if f.endswith('.json')]
        for file in tqdm(files_to_scan, desc='reading lyrics from jsons'):
            if file.endswith(".json"):
                full_path = os.path.join(root_path, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    sr = pd.Series(d)
                    sr.pop('songs')
                    d_songs = d['songs'].copy()

                    all_words = list()
                    unique_words = list()
                    for song_key in list(d_songs.keys()):
                        l = clean_song_list(d_songs[song_key])
                        d_songs[song_key] = l
                        all_words.extend(l)
                    unique_words = list(set(all_words))

                    artist_key = file.replace('.json', '')
                    songs_detected = len(d_songs)
                    empty_songs = [k for k, v in d_songs.items() if len(v) < 1]
                    idxs = index_summary['artist_key'] == artist_key
                    if idxs.sum() == 0:
                        raise Exception("Artist {} not found in index".format(artist_key))

                    index_summary.loc[idxs, 'songs_page'] = full_path
                    index_summary.loc[idxs, 'songs_crawled'] = songs_detected
                    index_summary.loc[idxs, 'songs_failed'] = index_summary.loc[idxs, 'songs_count'] - songs_detected
                    index_summary.loc[idxs, 'empty_songs'] = len(empty_songs)
                    index_summary.loc[idxs, 'words_count'] = len(all_words)
                    index_summary.loc[idxs, 'unique_words_count'] = len(unique_words)

        index_summary.to_csv(index_summary_path, index=False, encoding='utf-8-sig')
        logger.log_print("Summary file created at {}".format(index_summary_path))

    section_build_kaggle = False
    if section_build_kaggle:
        db = pd.DataFrame(columns=['artist', 'artist_key',
                                   'song', 'song_key', 'song_url',
                                   'words', 'words count', 'unique words count',
                                   ])

        url_df = pd.DataFrame(columns=['artist_key', 'song', 'url'])
        for row_idx, artist_row in tqdm(index_summary.iterrows(),
                                        total=index_summary.shape[0],
                                        desc='reading urls from jsons'):
            artist = artist_row['artist']
            artist_key = artist_row['artist_key']
            aritst_urls = artist_row['songs_page']

            base_file_name = os.path.basename(aritst_urls)
            index_file_path = os.path.join(index_path, base_file_name)

            with open(index_file_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                d_songs = d['songs'].copy()
                for song_key in list(d_songs.keys()):
                    song = song_key
                    song_url = d_songs[song_key]
                    url_df = pd.concat(
                        [url_df, pd.DataFrame([{'artist_key': artist_key, 'song': song, 'url': song_url}])],
                        ignore_index=True)

        xdf = pd.DataFrame()
        for artist_idx, song_file_path in tqdm(index_summary['songs_page'].items(), total=index_summary.shape[0],
                                               desc="Building Kaggle from jsons"
                                               ):
            artist_key = index_summary.loc[artist_idx, 'artist_key']
            with open(song_file_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                d_songs = pd.DataFrame(d)
                d_songs['song'] = d_songs.index
                d_songs['artist_key'] = artist_key
                d_songs.pop('songs_count')
                xdf = pd.concat([xdf, d_songs], ignore_index=True)

        xdf['songs'] = xdf['songs'].progress_apply(clean_song_list)

        kaggledf = pd.merge(xdf, url_df, on=['artist_key', 'song'], how='left')
        kaggledf['words count'] = kaggledf['songs'].progress_apply(len)
        kaggledf['unique words count'] = kaggledf['songs'].progress_apply(lambda x: len(set(x)))

        kaggledf.to_csv(kaggle_path, index=False, encoding='utf-8-sig')
        logger.log_print("Kaggle file created at {}".format(kaggle_path))

    section_words_extraction = True
    if section_words_extraction:
        logger.log_print("Words extraction:")
        index_summary = pd.read_csv(index_summary_path, encoding='utf-8-sig')

        all_words = list()
        unique_words = list()
        for artist_idx, song_file_path in index_summary['songs_page'].items():
            with open(song_file_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                d_songs = d['songs']
                for song_key in list(d_songs.keys()):
                    l = d_songs[song_key]
                    l = clean_song_list(l)
                    all_words.extend(l)

        # Create histogram of words
        words_hist = pd.Series(all_words).value_counts()
        words_hist = pd.DataFrame(columns=['word', 'count'], data=words_hist.reset_index().values)

        # Min words appearance = 3
        threshold = 3
        words_hist = words_hist[words_hist['count'] >= threshold]

        # Use NLP to extract words base
        word_normalizer = WordNormalizer()
        normalized_words = word_normalizer.normalize(list(words_hist['word'].values))
        bases = list(normalized_words['base'].unique())

        normalized_words = pd.merge(normalized_words, words_hist, on='word')
        normalized_words.rename(columns={'count': 'word count'}, inplace=True)

        histdf = pd.DataFrame(columns=['base', 'count', 'example 1', 'example 2', 'example 3'],
                              index=range(len(bases)))
        for i in tqdm(range(len(bases)), desc='Words base extraction'):
            base = bases[i]
            subdf = normalized_words[normalized_words['base'] == base]
            count = subdf['word count'].sum()
            examples = list(subdf['word'].unique())
            examples_3 = examples[:3]
            examples_3 = np.append(examples_3, [''] * (3 - len(examples_3)))
            histdf.loc[i, 'base'] = base
            histdf.loc[i, 'count'] = count
            histdf.loc[i, 'example 1'] = examples_3[0]
            histdf.loc[i, 'example 2'] = examples_3[1]
            histdf.loc[i, 'example 3'] = examples_3[2]

        histdf.sort_values(by='count', ascending=False, inplace=True)
        histdf.to_csv(words_hist_path, index=False, encoding='utf-8-sig')
        logger.log_print("Words histogram file created at {}".format(words_hist_path))

    logger.log_print("FINNISH")
    logger.log_close()
