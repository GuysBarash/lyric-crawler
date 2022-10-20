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
from itertools import chain

global english
global hebrew
global logger
english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
hebrew = 'אבגדהוזחטיכלמנסעפצקרשת' + 'םןךףץ'

from tqdm import tqdm

tqdm.pandas()


class WordNormalizer:
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        self.converstion_dict = None

    def normalize(self, word_list, top_n=None, use_memory=True):
        if use_memory:
            return self._normalize_load_from_memory(word_list, top_n)
        else:
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
        path = os.path.join(self.data_path, 'words_2_base.csv')
        conversion_table = r'words_2_base_conversion.json'
        conversion_full_path = os.path.join(self.data_path, conversion_table)
        if os.path.exists(path):
            res = pd.read_csv(path, index_col=0)
        else:
            res = self.normalize(word_list, top_n, use_memory=False)
            res.to_csv(path, encoding='utf-8-sig')

        if os.path.exists(conversion_full_path):
            with open(conversion_full_path, 'r', encoding='utf-8-sig') as f:
                self.converstion_dict = json.load(f)
        else:
            conversion = dict(zip(res['word'], res['base']))
            with open(conversion_full_path, 'w', encoding='utf-8-sig') as f:
                json.dump(conversion, f, ensure_ascii=False, indent=4)

            self.converstion_dict = conversion

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
        root_path = os.path.dirname(os.path.realpath(__file__))
        root_data_path = os.path.join(root_path, 'data')
        if not os.path.exists(root_data_path):
            os.makedirs(root_data_path)

        index_path = os.path.join(root_data_path, 'index')
        if not os.path.exists(index_path):
            os.makedirs(index_path)

        labels_path = os.path.join(root_data_path, 'labels')
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)

        mapping_path = os.path.join(root_data_path, 'mapping')
        if not os.path.exists(mapping_path):
            os.makedirs(mapping_path)

        index_summary_path = os.path.join(root_data_path, 'index_summary.csv')
        words_hist_path = os.path.join(root_data_path, 'words_hist.csv')
        kaggle_path = os.path.join(root_data_path, 'kaggle.csv')
        per_artist_path = os.path.join(mapping_path, 'artist_sentiment.csv')

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

    section_summary = False
    if section_summary:
        print("Summary:")
        root_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        index_path = os.path.join(root_data_path, 'index')

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

        files_to_scan = [f for f in os.listdir(root_data_path) if f.endswith('.json')]
        for file in tqdm(files_to_scan, desc='reading lyrics from jsons'):
            if file.endswith(".json") and file.startswith('Artist'):
                full_path = os.path.join(root_data_path, file)
                with open(full_path, 'r', encoding='utf-8-sig') as f:
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

    section_words_extraction = False
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

        subsection_create_word_base = True
        if subsection_create_word_base:
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

    section_build_map_words_to_sentiments = False
    if section_build_map_words_to_sentiments:
        word_hist_labels = os.path.join(labels_path, 'words_hist.csv')
        convertion_dict_path = os.path.join(root_data_path, 'words_2_base_conversion.json')

        word_2_sentiment_path = os.path.join(labels_path, 'word_2_sentiment.csv')

        logger.log_print("Build map songs to sentiments:")
        if not os.path.exists(word_2_sentiment_path):
            with open(convertion_dict_path, 'r', encoding='utf-8-sig') as f:
                converstion_dict = json.load(f)

            sentidmentdf = pd.read_csv(word_hist_labels, encoding='utf-8-sig', low_memory=False)
            good_cols = [c for c in sentidmentdf.columns if c not in ['count'] + [f'example {i}' for i in range(10)]]
            sentidmentdf = sentidmentdf[good_cols]
            senti_cols = [c for c in sentidmentdf.columns if c != 'base']
            for c in senti_cols:
                sentidmentdf[c] = sentidmentdf[c].eq('x')

            words_2_base = pd.DataFrame(columns=['word', 'base'])
            words_2_base['word'] = converstion_dict.keys()
            words_2_base['base'] = converstion_dict.values()

            words_2_sentiment = pd.merge(words_2_base, sentidmentdf, left_on='base', right_on='base', how='left')
            words_2_sentiment.set_index('word', inplace=True)
            words_2_sentiment.to_csv(word_2_sentiment_path, encoding='utf-8-sig')
            logger.log_print("Words to sentiment file created at {}".format(word_2_sentiment_path))
        else:
            words_2_sentiment = pd.read_csv(word_2_sentiment_path, encoding='utf-8-sig', index_col='word')
            logger.log_print("Words to sentiment file loaded from {}".format(word_2_sentiment_path))

    section_build_map_songs_to_sentiments = False
    if section_build_map_songs_to_sentiments:
        songs_map_path = os.path.join(mapping_path, 'songs_map.csv')
        if not os.path.exists(songs_map_path):
            kaggledf = pd.read_csv(kaggle_path, encoding='utf-8-sig')
            if type(kaggledf['songs'].iloc[0]) == str:
                kaggledf['songs'] = kaggledf['songs'].apply(lambda x: eval(x))

            all_words = list(set(list(chain(*kaggledf['songs'].values.tolist()))))
            all_used_words = words_2_sentiment.index.to_list()
            words_not_used = list(set(all_words) - set(all_used_words))

            missing_words = pd.DataFrame(columns=words_2_sentiment.columns, data=False, index=words_not_used)
            missing_words['base'] = missing_words.index
            words_2_sentiment = pd.concat([words_2_sentiment, missing_words])
            words_2_sentiment.pop('base')


            def get_sentiment(row_idx, row):
                l = row['songs']
                ret = words_2_sentiment.loc[l]
                ret = ret.sum()
                # ret['total'] = len(l)
                ret.name = row_idx
                return ret


            tl = list()
            for row_idx, row in tqdm(kaggledf.iterrows(), desc='Songs to sentiments', total=len(kaggledf)):
                sentiments = get_sentiment(row_idx, row)
                tl.append(sentiments)
            redf = pd.concat(tl, axis=1).T
            songs_sentiment = pd.merge(kaggledf, redf, left_index=True, right_index=True)
            songs_sentiment.to_csv(songs_map_path, encoding='utf-8-sig')
            logger.log_print("Songs to sentiment file created at {}".format(songs_map_path))

        else:
            songs_sentiment = pd.read_csv(songs_map_path, encoding='utf-8-sig', index_col=0)
            logger.log_print("Songs to sentiment file loaded from {}".format(songs_map_path))

    section_build_map_artists_to_sentiments = False
    if section_build_map_artists_to_sentiments:

        if not os.path.exists(per_artist_path):
            word_hist_labels = os.path.join(labels_path, 'Index_summary.csv')
            artistdf = pd.read_csv(word_hist_labels, encoding='utf-8-sig', low_memory=True)

            artists_keys = songs_sentiment['artist_key'].unique().tolist()

            sentiments_cols = [c for c in songs_sentiment.columns if
                               c not in ['artist', 'songs', 'song', 'artist_key', 'url', 'unique words count']]
            id_cols = ['artist', 'artist_key']
            perartistdf = pd.DataFrame(columns=id_cols + sentiments_cols,
                                       index=artistdf.index,
                                       )
            perartistdf[id_cols] = artistdf[id_cols]

            per_artist_df = songs_sentiment.groupby('artist_key')[sentiments_cols].sum().astype(int)
            per_artist_df.to_csv(per_artist_path, encoding='utf-8-sig')
            logger.log_print("Per artist sentiment file created at {}".format(per_artist_path))
        else:
            per_artist_df = pd.read_csv(per_artist_path, encoding='utf-8-sig', index_col=0)
            logger.log_print("Per artist sentiment file loaded from {}".format(per_artist_path))

    section_answer_questions = True
    if section_answer_questions:
        results_path = os.path.join(root_path, 'results')
        genres_path = os.path.join(results_path, 'genres_and_era.csv')
        complete_artists_path = os.path.join(results_path, 'complete_artists.csv')
        artists_sentiments_reduced = os.path.join(results_path, 'artist_sentiments_reduced.csv')

        if not os.path.exists(results_path):
            os.mkdir(results_path)

        artists_info_df = pd.read_csv(index_summary_path, encoding='utf-8-sig', index_col=0)
        artists_sentiment_df = pd.read_csv(per_artist_path, encoding='utf-8-sig', index_col=0)
        artists_sentiment_df = artists_sentiment_df.loc[artists_info_df['artist_key'].values.tolist()]

        old_col = r'שנות ה-60- 90׳'
        new_col = 'שנות ה-90׳- כיום'
        east_col = r'Mizrahit'
        west_col = r'Pop+ Rock+ Hip-Hop'
        male_col = 'Male'
        female_col = 'Female'

        artists_info_df[old_col] = artists_info_df[old_col].eq('x')
        old_artists = artists_info_df[artists_info_df[old_col]]['artist_key'].values.tolist()
        artists_info_df[new_col] = artists_info_df[new_col].eq('x')
        new_artists = artists_info_df[artists_info_df[new_col]]['artist_key'].values.tolist()
        artists_info_df[east_col] = artists_info_df[east_col].eq('X')
        east_artists = artists_info_df[artists_info_df[east_col]]['artist_key'].values.tolist()
        artists_info_df[west_col] = artists_info_df[west_col].eq('X')
        west_artists = artists_info_df[artists_info_df[west_col]]['artist_key'].values.tolist()

        artists_info_df[male_col] = artists_info_df[male_col].eq('X')
        male_artists = artists_info_df[artists_info_df[male_col]]['artist_key'].values.tolist()
        artists_info_df[female_col] = artists_info_df[female_col].eq('X')
        female_artists = artists_info_df[artists_info_df[female_col]]['artist_key'].values.tolist()

        terms = ['אהבה',
                 'תקווה',
                 'יהדות',
                 'פוליטיקה',
                 'אינדיווידואל',
                 'קולקטיב',
                 'אלכוהול/סמים',
                 'עלבונות/ קללה',
                 'סלנג',
                 'מילים בשפות אחרות',
                 'מילים באנגלית',
                 'מילים בערבית',
                 'מילים מומצאות',
                 'מילים משובשות']

        double_terms = [
            ('צה״ל', 'מלחמה'),
            ('עיירות פיתוח', 'קיפוח'),
            ('ישראל', 'ערים ומקומות בישראל'),
            ('משפחה', 'ילדים'),
        ]

        genre_df = pd.DataFrame(columns=[old_col, new_col, east_col, west_col, male_col, female_col], index=terms)
        artist_df = pd.DataFrame(columns=artists_sentiment_df.index, index=terms)
        for term in terms:
            sentiments = artists_sentiment_df[term] / artists_sentiment_df['words count']
            # Normalize by standard deviation
            old_new_sentiment = sentiments.loc[old_artists + new_artists]
            old_new_sentiment = (old_new_sentiment - old_new_sentiment.mean()) / old_new_sentiment.std()

            east_west_sentiment = sentiments.loc[east_artists + west_artists]
            east_west_sentiment = (east_west_sentiment - east_west_sentiment.mean()) / east_west_sentiment.std()

            male_female_sentiments = sentiments.loc[male_artists + female_artists]
            male_female_sentiments = (
                                             male_female_sentiments - male_female_sentiments.mean()) / male_female_sentiments.std()

            sentiments_all = (sentiments - sentiments.mean()) / sentiments.std()

            genre_df.loc[term, old_col] = old_new_sentiment[old_artists].median().round(2)
            genre_df.loc[term, new_col] = old_new_sentiment[new_artists].median().round(2)
            genre_df.loc[term, east_col] = east_west_sentiment[east_artists].median().round(2)
            genre_df.loc[term, west_col] = east_west_sentiment[west_artists].median().round(2)
            genre_df.loc[term, male_col] = male_female_sentiments[male_artists].median().round(2)
            genre_df.loc[term, female_col] = male_female_sentiments[female_artists].median().round(2)

            artist_df.loc[term] = sentiments_all.round(2)

        double_terms = [(f'{t1} + {t2}', (t1, t2)) for t1, t2 in double_terms]
        double_genre_df = pd.DataFrame(columns=[old_col, new_col, east_col, west_col, male_col, female_col],
                                       index=[t[0] for t in double_terms])
        double_artist_df = pd.DataFrame(columns=artists_sentiment_df.index, index=[t[0] for t in double_terms])
        for term, (t1, t2) in double_terms:
            sentiments = (artists_sentiment_df[t1] + artists_sentiment_df[t2]) / artists_sentiment_df['words count']
            # Normalize by standard deviation
            old_new_sentiment = sentiments.loc[old_artists + new_artists]
            old_new_sentiment = (old_new_sentiment - old_new_sentiment.mean()) / old_new_sentiment.std()

            east_west_sentiment = sentiments.loc[east_artists + west_artists]
            east_west_sentiment = (east_west_sentiment - east_west_sentiment.mean()) / east_west_sentiment.std()

            male_female_sentiments = sentiments.loc[male_artists + female_artists]
            male_female_sentiments = (
                                             male_female_sentiments - male_female_sentiments.mean()) / male_female_sentiments.std()

            sentiments_all = (sentiments - sentiments.mean()) / sentiments.std()

            double_genre_df.loc[term, old_col] = old_new_sentiment[old_artists].median().round(2)
            double_genre_df.loc[term, new_col] = old_new_sentiment[new_artists].median().round(2)
            double_genre_df.loc[term, east_col] = east_west_sentiment[east_artists].median().round(2)
            double_genre_df.loc[term, west_col] = east_west_sentiment[west_artists].median().round(2)
            double_genre_df.loc[term, male_col] = male_female_sentiments[male_artists].median().round(2)
            double_genre_df.loc[term, female_col] = male_female_sentiments[female_artists].median().round(2)

            double_artist_df.loc[term] = sentiments_all.round(2)

        artist_df = pd.concat([artist_df, double_artist_df])
        genre_df = pd.concat([genre_df, double_genre_df])

        # get common indexes
        common_indexes = set(genre_df.index).intersection(set(double_genre_df.index))
        artist_df = artist_df.T

        genre_df.to_csv(genres_path, encoding='utf-8-sig')
        artist_df.to_csv(complete_artists_path, encoding='utf-8-sig')
        artists_sentiment_df.to_csv(artists_sentiments_reduced, encoding='utf-8-sig')
        logger.log_print("Genres and artists sentiment file created at {}".format(results_path))

    logger.log_print("FINNISH")
    logger.log_close()

terms = [
    # 'אהבה',
    # # 'חיבה',
    # # 'כעס',
    # 'תקווה',
    # # 'חיים',
    # # 'מוות',
    # # 'מחלה/ פציעה/ מגבלה',
    # # 'עצב',
    # # 'פרידה',
    # # 'שמחה',
    # # 'שלילי',
    # # 'חיובי',
    # # 'דיכאון',
    # # 'פחד',
    # # 'כאב',
    # # 'רגש חיובי',
    # # 'רגש שלילי',
    # # 'הצלחה',
    # # 'שלום',
    # # 'מלחמה',
    # # 'משפחה',
    # # 'ילדים',
    # 'יהדות',
    # # 'דת',
    # # 'חגים ומועדים',
    # # '(כינויי) זמן',
    # # 'חודשי השנה',
    # 'פוליטיקה',
    # # 'מדינה/ ממשלה',
    # # 'עונות השנה',
    # 'אינדיווידואל',
    # 'קולקטיב',
    # # 'מחאה',
    # # 'קיפוח',
    # # 'מחמאות',
    # # 'מילת תיאור',
    # # 'מוסיקה',
    # # 'איברי הגוף',
    # # 'תכונה חיצונית',
    # # 'מקום',
    # # 'ארצות',
    # # 'ערים בעולם/ חבלי ארץ',
    # # 'ערים ומקומות בישראל',
    # # 'עיירות פיתוח',
    # # 'ישראל',
    # # 'חורף',
    # # 'קיץ',
    # # 'מקצוע',
    # # 'צבא',
    # # 'צה״ל',
    # # 'חיות',
    # # 'אוכל',
    # 'אלכוהול/סמים',
    # # 'שתייה',
    # 'עלבונות/ קללה',
    # # 'לילה',
    # # 'מספרים',
    # # 'צבעים',
    # # 'שם אמן',
    # # 'שמות',
    # # 'נשים',
    # # 'גברים',
    # # 'גיל',
    # # 'טבע',
    # # 'אופנה',
    # # 'אלימות',
    # # 'כסף',
    # # 'תכונות',
    # # 'זיכרון (נוסטלגי)ֿ',
    # # 'זיכרון (אבל)',
    # 'סלנג',
    # # 'כלי תחבורה',
    # 'מילים בשפות אחרות',
    # 'מילים באנגלית',
    # 'מילים בערבית',
    # # 'מוצא',
    # # 'שפות',
    # # 'השכלה',
    # # 'תרבות- תחביבים- ספורט',
    # # 'מילת שאלה',
    # # 'מילת קישור',
    # # 'כינויי גוף',
    # # 'שייכות',
    # # 'כינויי רמז',
    # # 'מילת יחס',
    # 'מילים מומצאות',
    # 'מילים משובשות',
]
