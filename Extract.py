#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from Logger import Logger
from multiprocessing import Queue, Pool, cpu_count
from collections import Counter
import numpy as np
import random
import copy
import cPickle as pickle
import xlrd
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('UTF8')


def in_dic(word):
    global dic
    dif_val = ""
    if word in dic:
        return [True] + dic[word]
    elif word.startswith('ה') or word.startswith('מ') or word.startswith('ל') or word.startswith(
            'ב') or word.startswith('ש') or word.startswith('ו') or word.startswith('כ'):
        temp_ret = in_dic(word[2:])
        if temp_ret[0]:
            return temp_ret
        else:
            return [False, dif_val, 'Nan']

    elif "\'" in word:
        temp_ret = in_dic(word[3:])
        if temp_ret[0]:
            return temp_ret
        else:
            return [False, dif_val, 'Nan']
    else:
        return [False, dif_val, 'Nan']


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


def tokens_per_artist_filtered(pool_ret, reshuffle=200):
    thrsh = average_number_of_words(pool_ret)
    logger.squelch(True)
    logger.log_print("Tokens per {} words:".format(thrsh))
    logger.log_print()
    for item in pool_ret:
        if len(item[2]) >= thrsh:
            t_avg = 0
            filtered_list = [in_dic(w)[1] for w in item[2]]
            for i in range(reshuffle):
                t = copy.copy(filtered_list)
                random.shuffle(t)
                t_avg += len(set(t[:thrsh]))
            avg = t_avg / reshuffle
            logger.log_print("{}\t{}".format(item[0], avg))

    logger.log_print()
    logger.squelch(False)


def tokens_per_artist(pool_ret, reshuffle=200, circular=True):
    thrsh = average_number_of_words(pool_ret)
    logger.squelch(True)
    logger.log_print("Tokens per {} words, repeat {} times:".format(thrsh, reshuffle))
    logger.log_print()

    df_o = pd.DataFrame()
    df_o['Artist_name'] = []
    df_o['Tokens'] = []
    df = df_o.copy()

    for item in pool_ret:
        if len(item[2]) >= thrsh:
            t_avg = 0
            for i in range(reshuffle):
                t = copy.copy(item[2])
                if circular:
                    st_point = random.randint(0, len(t) - 1)
                    if st_point + thrsh < len(t):
                        nt = t[st_point:st_point + thrsh]
                    else:
                        nt = t[st_point:] + t[: (st_point + thrsh) % len(t)]

                else:
                    random.shuffle(t)
                    nt = t[:thrsh]

                assert len(nt) == thrsh, "ERROR of length"
                t_avg += len(set(nt))
            avg = t_avg / reshuffle
            logger.log_print("{}\t{}".format(item[0], avg))

            df_t = df_o.copy()
            df_o['Artist_name'] = [item[0]]
            df_o['Tokens'] = [avg]
            df = pd.concat([df, df_t], ignore_index=True)

    exstra = ''
    if circular:
        exstra = '_Circular'

    excel_full_path = 'C:\\Repos\\SongReader\\Vector' + '\\' + 'Tokens_per_artist{}.xlsx'.format(exstra)
    logger.log_print("To excel at {}".format(excel_full_path))
    xl_writer = pd.ExcelWriter(excel_full_path, engine='xlsxwriter', options={'encoding': 'utf-8'})
    df.to_excel(xl_writer, sheet_name='{}_samples'.format(reshuffle), index=False)
    xl_writer.close()

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
    logger.log_print(
        "Showing {} most common words out of {} total".format(amount_to_view_actual, len(word_counts)))
    logger.squelch()
    index = 0
    for item in word_counts.most_common(amount_to_view_actual):
        index += 1
        mst_common[item[0]] = item[1]
        logger.log_print("{}\t{}\t{}".format(index, item[0], item[1]))
    logger.squelch(False)
    logger.log_print("**************")
    return word_counts.most_common(amount_to_view_actual)


def Artist_to_vector(all_units):
    all_types = {
        'propername': 0,
        'verb': 0,
        'wPrefix': 0,
        'adverb': 0,
        'pronoun': 0,
        'passiveParticiple': 0,
        'existential': 0,
        'negation': 0,
        'participle': 0,
        'adjective': 0,
        'preposition': 0,
        'interrogative': 0,
        'modal': 0,
        'numeral': 0,
        'copula': 0,
        'noun': 0,
        'quantifier': 0,
        'conjunction': 0,
        'interjection': 0}
    df_o = pd.DataFrame()
    df_o['Artist_name'] = []
    for key in all_types:
        df_o[key] = []
    df = df_o.copy()

    for unit in all_units:
        vector = all_types.copy()
        vector['Artist_name'] = unit[0]
        amount_of_tokens = 0
        total_num = 0
        for w in unit[2]:
            w_info = in_dic(w)
            total_num += 1
            if w_info[0]:
                amount_of_tokens += 1
                vector[w_info[2]] = vector.get(w_info[2], 0) + 1

        df_t = df_o.copy()
        for key in all_types:
            df_t[key] = [0]
        for key in vector.keys():
            if key != 'Artist_name':
                # vector[key] = float(vector[key]) / float(amount_of_tokens)
                df_t[key] = [float(vector[key]) / float(amount_of_tokens)]
            else:
                df_t[key] = [vector[key]]
        df = pd.concat([df, df_t], ignore_index=True)
        logger.log_print("{}:\t{:.2f}%".format(unit[0], 100 * float(amount_of_tokens) / float(total_num)))

    excel_full_path = 'C:\\Repos\\SongReader\\Vector' + '\\' + 'vec.xlsx'
    logger.log_print("To excel at {}".format(excel_full_path))
    xl_writer = pd.ExcelWriter(excel_full_path, engine='xlsxwriter', options={'encoding': 'utf-8'})
    df.to_excel(xl_writer, sheet_name='vectorized')
    xl_writer.close()


def get_nouns_by_freq(all_units):
    logger.log_print("**************")
    logger.log_print("Noun to histogram")
    logger.squelch()
    logger.log_print()

    words_dic = dict()

    for unit in all_units:
        curr_words = set()
        logger.log_print("{}".format(unit[0]))
        for name, words in unit[1].iteritems():
            for word in words:
                if is_noun(word):
                    word_to_add = word[2:]
                    words_dic[word_to_add] = words_dic.get(word_to_add, 0) + 1

    logger.log_print()
    for word, val in words_dic.iteritems():
        logger.log_print("{}\t{}".format(word, val))
    logger.log_print()
    logger.squelch(False)
    logger.log_print("Noun extracted")


def excel_to_dict():
    f_name = r"C:\Repos\SongReader\lexicon\inflections_urldecoded.xlsx"
    logger.log_print("Opening excel")
    workbook = xlrd.open_workbook(f_name, encoding_override='cp1252')
    logger.log_print("Opening excel completed")
    worksheet = workbook.sheet_by_index(0)
    numrows = worksheet.nrows - 1
    curr_row = -1
    dic = dict()
    while curr_row < numrows:
        curr_row += 1
        row = worksheet.row(curr_row)
        word = row[1].value.encode('utf8')
        type_of_word = row[3].value.encode('utf8')
        base_form = row[5].value.encode('utf8')
        dic[word] = [base_form, type_of_word]
        logger.log_print("{} {} {}".format(word, type_of_word, base_form))
    logger.log_print("Dumping start")
    pickle_dest = "lexicon.p"
    pickle.dump(dic, open("Data\\" + pickle_dest, "wb"))
    logger.log_print("Dumping completed")


def open_dict(dict_location):
    global dic
    logger.log_print("Opening dict")
    dic = pickle.load(open(dict_location, 'rb'))
    logger.log_print("dict Opended")
    logger.log_print("")
    return dic


def dump(df, fname='vec_norm', path=None):
    if path is None:
        excel_full_path = 'C:\\Repos\\SongReader\\Vector' + '\\' + fname + '.xlsx'
    else:
        excel_full_path = path + '\\' + fname + '.xlsx'
    logger.log_print("To excel at {}".format(excel_full_path))
    xl_writer = pd.ExcelWriter(excel_full_path, engine='xlsxwriter', options={'encoding': 'utf-8'})
    df.to_excel(xl_writer, sheet_name='vectorized')
    xl_writer.close()


def get_raw_data(ret_pool):
    logger.log_print("Extracting raw data")
    path = r'C:\Repos\SongReader\Vector'
    cols = ['Songs', 'Total words', 'mean per song', 'Var per song']
    df = pd.DataFrame(columns=cols)
    for item in ret_pool:
        name = item[0]
        songDict = item[1]
        num_of_songs = len(songDict.keys())
        songs_vector = [len(x) for x in songDict.itervalues()]
        mean = np.mean(songs_vector)
        std = np.std(songs_vector)
        sum = np.sum(songs_vector)
        row = pd.Series([num_of_songs, sum, mean, std], index=cols, name=name)
        df = df.append(row)
    dump(df, 'Raw_data', path)


if __name__ == '__main__':
    global logger
    logger = Logger()
    logger.initThread(fout='Extract_Report.txt')

    # Get all songs
    files = ['Data\\' + f for f in os.listdir('.\\Data') if f.startswith('Artist_')]

    pool_size = cpu_count()
    pool = Pool(processes=pool_size, initializer=extract_worker_init,
                initargs=(logger,))
    pool_ret = pool.imap_unordered(extract_worker, files)
    pool.close()
    pool.join()

    all_units = []
    for item in pool_ret:
        all_units.append(item)
        # item[0] : artist name
        # item[1] : dict , key= song_name , value= list_of_words
        # item[2] : list of all words

    # get_raw_data(all_units)
    # dict_location = r"C:\Repos\SongReader\Data\lexicon.p"
    # dic = open_dict(dict_location)
    # Artist_to_vector(all_units)
    # tokens_per_artist(all_units, reshuffle=400, circular=False)

    logger.log_print()
    logger.log_close()
