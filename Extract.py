#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from Logger import Logger
from multiprocessing import Queue, Pool, cpu_count
from collections import Counter
import numpy as np
import random
import copy


def checkline(expected, line, fname):
    if not line.startswith(expected):
        logger.log_error('Corruption in {}'.format(fname))
        logger.log_error('Line: {}'.format(line))
        raise
    else:
        return


def extract(fname):
    # logger.log_print("Handling {}".format(fname))
    f = open(fname)
    unit = []

    # Get artist name
    line = f.readline()
    checkline('#Total', line, fname)
    line = f.readline()
    checkline('#Artist', line, fname)
    line = f.readline()
    unit.append(line.replace('\n', ''))
    all_words = []
    all_songs = dict()
    while True:
        line = f.readline()
        if line.startswith('#FIN'):
            break
        elif line.startswith('#NAME'):
            line = f.readline()
            curr_song_name = line.replace('\n', '')
            all_songs[curr_song_name] = []
            line = f.readline()
            checkline('#DONE_NAME', line, fname)
            line = f.readline()
            checkline('#WORDS', line, fname)
            line = f.readline()
            while not line.startswith("#DONE_WORDS"):
                line = line.replace('\n', '')
                curr_line_words = [w for w in line.split('\t') if len(w) > 2]
                all_songs[curr_song_name] += curr_line_words
                all_words += curr_line_words
                line = f.readline()

        else:
            logger.log_error('Corruption in {}'.format(fname))
            logger.log_error('Line: {}'.format(line))
            raise
    unit.append(all_songs)
    unit.append(all_words)
    # logger.log_print("DONE {}".format(fname))
    return unit


def extract_worker_init(loggert, ):
    global logger
    logger = loggert
    logger.log_print('FORK')


def extract_worker(fname):
    curr_item = extract(fname)
    logger.log_print(curr_item[0])
    return curr_item


def average_number_of_words(pool_ret):
    global logger
    logger.log_print('Calculating average amount of words per artist')
    c = np.array([])
    for item in pool_ret:
        c = np.append(c, [len(item[2])])
    avg = np.average(c) * 0.75
    return int(avg)


def tokens_per_artist(pool_ret, reshuffle=200):
    x = []
    y = []
    thrsh = average_number_of_words(pool_ret)
    logger.squelch(True)
    logger.log_print("Tokens per {} words:".format(thrsh))
    logger.log_print()
    for item in pool_ret:
        if len(item[2]) >= thrsh:
            x.append(item[0])
            t_avg = 0
            for i in range(reshuffle):
                t = copy.copy(item[2])
                random.shuffle(t)
                t_avg += len(set(t[:thrsh]))
            avg = t_avg / reshuffle
            y.append(avg)
    for index in range(len(x)):
        logger.log_print("{}\t{}".format(x[index], y[index]))
    logger.log_print()
    logger.squelch(False)


def total_number_of_songs(pool_ret):
    global logger
    song_counter = 0
    artist_counter = 0
    for item in pool_ret:
        artist_counter += 1
        song_counter += len(item[1])

    logger.log_print("Total number of songs {}".format(song_counter))
    logger.log_print("Total number of Artists {}".format(artist_counter))
    logger.log_print()
    return [song_counter, artist_counter]


def top_n_words_of_all(all_units, amount_to_view=1000):
    logger.log_print("**************")
    logger.log_print("To histogram")
    all_words = []
    for item in all_units:
        all_words += item[2]
    word_counts = Counter(all_words)
    mst_common = dict()

    amount_to_view_actual = min(amount_to_view, len(word_counts))
    logger.log_print("Showing {} most common words out of {} total".format(amount_to_view_actual, len(word_counts)))
    logger.squelch()
    index = 0
    for item in word_counts.most_common(amount_to_view_actual):
        index += 1
        mst_common[item[0]] = item[1]
        logger.log_print("{}\t{}\t{}".format(index, item[0], item[1]))
    logger.squelch(False)
    logger.log_print("**************")
    return word_counts.most_common(amount_to_view_actual)


if __name__ == '__main__':
    global logger
    logger = Logger()
    logger.initThread(fout='Extract_Report.txt')

    # Get all songs
    files = [f for f in os.listdir('.') if f.startswith('Artist_')]

    pool_size = cpu_count()
    pool = Pool(processes=pool_size, initializer=extract_worker_init,
                initargs=(logger,))
    pool_ret = pool.imap_unordered(extract_worker, files)
    pool.close()
    pool.join()

    all_units = []
    for item in pool_ret:
        all_units.append(item)

    top_n_words_of_all(all_units, 100000)

    logger.squelch(False)
    logger.log_print()
    logger.log_close()
