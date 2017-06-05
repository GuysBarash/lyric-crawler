#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import itertools as it
from Logger import Logger

ctext = "C:\Repos\SongReader\Artist_Dor_Daniel.txt"


def extract(fname):
    logger.log_print("Handling {}".format(fname))
    f = open(fname)

    line = f.readline()
    mode = 'seek'
    return_value = dict()
    currsong = 'XXX'
    while line != '#FIN':
        try:
            if mode == 'seek':
                if line.startswith('#NAME'):
                    mode = 'get_name'
                    line = f.readline().replace('\n', '')
                else:
                    continue
            elif mode == 'get_name':
                currsong = line
                return_value[currsong] = list()
                mode = 'seek_words_header'

            elif mode == 'seek_words_header':
                if line.startswith('#WORDS'):
                    mode = 'get_words'
                else:
                    continue
            elif mode == 'get_words':
                pass



            else:
                pass


        finally:



if __name__ == '__main__':
    global logger
    logger = Logger()
    logger.initThread(fout='Report')

    extract(ctext)

    logger.log_close()
