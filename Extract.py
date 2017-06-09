#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from Logger import Logger
from multiprocessing import Queue, Pool, cpu_count
from collections import Counter


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
                curr_line_words = [w for w in line.split('\t') if len(w) >2]
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


def worker_init(loggert, ):
    global logger
    logger = loggert
    logger.log_print('FORK')


def worker(fname):
    curr_item = extract(fname)
    logger.log_print(curr_item[0])
    return curr_item


if __name__ == '__main__':
    global logger
    logger = Logger()
    logger.initThread(fout='Report')

    # Get all songs
    files = [f for f in os.listdir('.') if f.startswith('Artist_')]

    pool_size = cpu_count()
    pool = Pool(processes=pool_size, initializer=worker_init,
                initargs=(logger,))
    pool_ret = pool.imap_unordered(worker, files)
    pool.close()
    pool.join()

    all_words = []
    for item in pool_ret:
        all_words += item[2]
    logger.log_print("**************")
    logger.log_print("To histogram")
    word_counts = Counter(all_words)
    logger.log_print("Graph")

    mst_common = dict()

    logger.squelch()
    for item in word_counts.most_common(35):
        mst_common[item[0]] = item[1]
        logger.log_print("{}\t{}".format(item[0], item[1]))

    logger.squelch(False)
    logger.log_print()
    logger.log_print("Total of {} different words.".format(len(word_counts)))
    logger.log_close()
