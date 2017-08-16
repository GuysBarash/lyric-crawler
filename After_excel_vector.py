#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn import preprocessing, decomposition
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN
import itertools, datetime, Tkinter, ttk
from Logger import Logger

headers = ['Artist_name']
info = [u'adjective', u'adverb', u'conjunction', u'copula', u'existential', u'indInf',
        u'interjection', u'interrogative', u'modal', u'negation', u'noun', u'numeral', u'participle',
        u'passiveParticiple', u'preposition', u'pronoun', u'propername', u'quantifier', u'title', u'verb', u'wPrefix']


class Ticker:
    def __init__(self, total):
        self.total = total
        self.strt_time = datetime.datetime.now()

        self.root = Tkinter.Tk()
        self.root.wm_title("Lamplighter")
        self.root.geometry('{}x{}'.format(400, 100))

        self.progress_var = Tkinter.DoubleVar()  # here you have ints but when calc. %'s usually floats
        self.theLabel = Tkinter.Label(self.root, text="Reading trace")
        self.theLabel.pack()

        self.theBody = Tkinter.Label(self.root, text="Completed: 0%")
        self.theBody.pack()

        self.progressbar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progressbar.pack(fill='x', expand=1)

    def update(self, val):
        curr_time = datetime.datetime.now()
        diff = (curr_time - self.strt_time).total_seconds()
        speed = float(diff) / float(val)
        time_remain = (self.total - val) * speed
        time_str = "{:>8}".format(datetime.timedelta(seconds=time_remain))

        perc = (100 * val) / self.total

        self.theBody.config(text="Completed: {}%\tTime Remain: {}".format(perc, time_str))
        self.theBody.pack()
        self.progress_var.set(perc)
        self.root.update()

    def close(self):
        self.root.destroy()


def kmean(df, general_path):
    y_res = []
    x_res = []
    df_results = pd.DataFrame(index=[n for n in df.index])
    for n_clusters in range(3, 25):
        curr_path = general_path + '\\' + '{}_clusters'.format(n_clusters)
        try:
            os.stat(curr_path)
            shutil.rmtree(curr_path)
            os.mkdir(curr_path)
        except Exception as e:
            os.mkdir(curr_path)

        if n_clusters > len(df.index):
            break

        est = KMeans(n_clusters=n_clusters)
        est.fit(df)
        y_res.append(est.inertia_)
        x_res.append(n_clusters)
        # print "Cluster: {}\tEstimator: {}".format(n_clusters, est.inertia_)
        current = pd.Series(est.labels_, index=df.index)
        df_results['n = {}'.format(n_clusters)] = current

        cols = ['Cluster'] + range(n_clusters)
        df_distances = pd.DataFrame(columns=cols)
        for k, v in current.iteritems():
            A = df.ix[k].values
            distances = []
            for i in range(len(est.cluster_centers_)):
                B = est.cluster_centers_[i]
                curr_X_mse = distance.euclidean(A, B)
                distances += [curr_X_mse]

            row = [v] + distances
            row_name = k
            s = pd.Series(row, index=cols, name=row_name)
            df_distances = df_distances.append(s)
        dump(df_distances, 'distances_{}'.format(n_clusters), curr_path)

        labels = est.labels_
        labels_legend = set(labels)
        pca = decomposition.PCA()
        pca.fit(df)
        ## 2D projection
        pca.n_components = 2
        X_reduced = np.array(pca.fit_transform(df))
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        for k in range(n_clusters):
            indexes = [i for i in range(df.shape[0]) if labels[i] == k]
            X_reduced_k = X_reduced[indexes]
            X_sctr = [X_reduced_k[:, w] for w in range(X_reduced_k.shape[1])]
            plt.scatter(X_sctr[0], X_sctr[1], c=[colors[k]] * len(indexes), alpha=0.5, label="cluster {}".format(k))
        plt.legend()
        plt.savefig(curr_path + '\\' + '2D_scatter_{}_clusters'.format(n_clusters) + '.png')
        plt.close()

        ## 3D projection
        pca.n_components = 3
        X_reduced = np.array(pca.fit_transform(df))
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k in range(n_clusters):
            indexes = [i for i in range(df.shape[0]) if labels[i] == k]
            X_reduced_k = X_reduced[indexes]
            X_sctr = [X_reduced_k[:, w] for w in range(X_reduced_k.shape[1])]
            ax.scatter(X_sctr[0], X_sctr[1], X_sctr[2], c=[colors[k]] * len(indexes), alpha=0.5,
                    label="cluster {}".format(k))
        plt.legend()
        # plt.show()
        plt.savefig(curr_path + '\\' + '3D_scatter_{}_clusters'.format(n_clusters) + '.png')
        plt.close()

    logger.log_print("Completed on {} k's.".format(len(x_res)))
    dump(df_results, 'clusters')
    plt.figure()
    plt.xlabel('Clusters')
    plt.ylabel('MSE')
    plt.title('MSE over number of clusters')
    plt.plot(x_res, y_res)
    plt.savefig('C:\\Repos\\SongReader\\Vector' + '\\' + 'MSE' + '.png')
    plt.close()


def get_epsilons(df, how_many=10000, thrsh=0.85):
    import sys
    global logger
    dist = euclidean_distances(df)
    np.fill_diagonal(dist, sys.maxint)
    closest_neighbor = dist.min(axis=1)
    max_closest_neigh = max(closest_neighbor)
    min_closest_neigh = min(closest_neighbor)
    chunck = float(max_closest_neigh - min_closest_neigh) / float(how_many)
    ret = []
    logger.log_print("Calculating epsilons")
    for eps in np.arange(min_closest_neigh, max_closest_neigh, chunck):
        b = closest_neighbor < eps
        below_it = len(closest_neighbor[b]) / float(len(closest_neighbor))
        if below_it > thrsh:
            ret.append(eps)
    logger.log_print("Calculating epsilons complete. Generated: {}".format(len(ret)))
    return ret


def dbscan(df):
    global logger
    eps_list = get_epsilons(df, how_many=1000000, thrsh=0.50)
    vars = list(itertools.product(np.arange(2, 5, 1), eps_list))
    number_of_pckgs = len(vars)
    current_pckge = 0
    ticker = Ticker(number_of_pckgs)
    for inputs in vars:
        current_pckge += 1
        eps = inputs[1]
        min_samples = inputs[0]
        ticker.update(current_pckge)
        est = DBSCAN(eps=eps, min_samples=min_samples)
        est.fit(df)
        if len(set(est.labels_)) > 5:
            noise = list(est.labels_).count(-1)
            logger.log_print(
                "[min= {}][eps= {}]\tClusters: {} Noise: {}".format(min_samples, eps, len(set(est.labels_)), noise))
    ticker.close()
    logger.log_print("Done")


def dump(df, fname='vec_norm', path=None):
    if path is None:
        excel_full_path = 'C:\\Repos\\SongReader\\Vector' + '\\' + fname + '.xlsx'
    else:
        excel_full_path = path + '\\' + fname + '.xlsx'
    logger.log_print("To excel at {}".format(excel_full_path))
    xl_writer = pd.ExcelWriter(excel_full_path, engine='xlsxwriter', options={'encoding': 'utf-8'})
    df.to_excel(xl_writer, sheet_name='vectorized')
    xl_writer.close()


if __name__ == '__main__':
    logger = Logger()
    logger.initThread()

    general_path = r'C:\Repos\SongReader\Vector'
    excel_file = general_path + '\\' + 'vec.xlsx'

    df = pd.read_excel(excel_file, options={'encoding': 'utf-8'})
    df.set_index('Artist_name', inplace=True)
    df = df.fillna(value=0)
    del df['indInf']
    df_norm = (df - df.mean()) / (df.max() - df.min())
    dump(df_norm, 'vec_norm')
    kmean(df, general_path)

    logger.log_close()
