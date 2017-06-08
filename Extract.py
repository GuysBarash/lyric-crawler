#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from Logger import Logger
from multiprocessing import Queue, Pool, cpu_count


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
                all_songs[curr_song_name] += [w for w in line.split('\t') if w != '']
                line = f.readline()

        else:
            logger.log_error('Corruption in {}'.format(fname))
            logger.log_error('Line: {}'.format(line))
            raise
    unit.append(all_songs)
    # logger.log_print("DONE {}".format(fname))
    return unit


def worker_init(loggert, ):
    global logger
    logger = loggert
    logger.log_print('FORK')


def worker(fname):
    curr_item = extract(fname)
    logger.log_print(curr_item[0])
    return curr_item[0]


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

    logger.log_print("**************")
    logger.log_close()
