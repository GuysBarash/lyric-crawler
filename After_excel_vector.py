#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from Logger import Logger

headers = ['Artist_name']
info = [u'adjective', u'adverb', u'conjunction', u'copula', u'existential', u'indInf',
        u'interjection', u'interrogative', u'modal', u'negation', u'noun', u'numeral', u'participle',
        u'passiveParticiple', u'preposition', u'pronoun', u'propername', u'quantifier', u'title', u'verb', u'wPrefix']


def dump(df, fname='vec_norm'):
    excel_full_path = 'C:\\Repos\\SongReader\\Vector' + '\\' + fname + '.xlsx'
    logger.log_print("To excel at {}".format(excel_full_path))
    xl_writer = pd.ExcelWriter(excel_full_path, engine='xlsxwriter', options={'encoding': 'utf-8'})
    df.to_excel(xl_writer, sheet_name='vectorized')
    xl_writer.close()


if __name__ == '__main__':
    logger = Logger()
    logger.initThread()

    excel_file = r"C:\Repos\SongReader\summary\vec.xlsx"
    df = pd.read_excel(excel_file, index_col='Artist_name', options={'encoding': 'utf-8'})
    df = df.fillna(value=0)
    del df['Column1']
    del df['indInf']
    df_norm = (df - df.mean()) / (df.max() - df.min())
    dump(df_norm, 'vec_norm')
    logger.log_close()
