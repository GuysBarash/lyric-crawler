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
from Logger import Logger

max_num_of_clusters = 25
min_num_of_clusters = 3

headers = ['Artist_name']
info = [u'adjective', u'adverb', u'conjunction', u'copula', u'existential', u'indInf',
        u'interjection', u'interrogative', u'modal', u'negation', u'noun', u'numeral', u'participle',
        u'passiveParticiple', u'preposition', u'pronoun', u'propername', u'quantifier', u'title', u'verb', u'wPrefix']


def kmean(df, general_path):
    y_res = []
    x_res = []
    sheets_q = list()
    df_results = pd.DataFrame(index=[n for n in df.index])
    output_file_name = 'clusters_kmean'
    for n_clusters in range(min_num_of_clusters, max_num_of_clusters):
        curr_path = general_path + '\\' + '{}_clusters'.format(n_clusters)
        logger.log_print("Processing {} clusters in Kmean".format(n_clusters))
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
        sheets_q.append((n_clusters, df_distances))

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
        plt.savefig(curr_path + '\\' + '2D_scatter_{}_clusters'.format(n_clusters) + '.jpeg')
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
        plt.savefig(curr_path + '\\' + '3D_scatter_{}_clusters'.format(n_clusters) + '.jpeg')
        plt.close()

    logger.log_print("Completed on {} k's.".format(len(x_res)))
    dump(df_results, output_file_name, sheet_name='Summary')
    while len(sheets_q) > 0:
        (n_clusters, df_distances) = sheets_q.pop(0)
        dump(df_distances, output_file_name, sheet_name='k={}'.format(n_clusters))

    plt.figure()
    plt.xlabel('Clusters')
    plt.ylabel('MSE')
    plt.title('MSE over number of clusters')
    plt.plot(x_res, y_res)
    plt.savefig('C:\\Repos\\SongReader\\Vector' + '\\' + 'MSE' + '.jpeg')
    plt.close()


def hierarchy_cluster(df, general_path):
    from matplotlib import pyplot as pltx
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import fcluster
    output_file_name = 'clusters_agglomerative'

    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        else:
            kwargs['color_threshold'] = 0.85

        annotate_above = kwargs.pop('annotate_above', 0)
        df_names = pd.read_excel(raw_data_file, options={'encoding': 'utf-8'})['English name'].tolist()
        kwargs['labels'] = df_names
        if kwargs.get('orientation', 'X') == 'right':
            kwargs['leaf_rotation'] = 0.
        else:
            kwargs['leaf_rotation'] = 90.

        ddata = dendrogram(*args, **kwargs)
        if not kwargs.get('no_plot', False):
            pltx.title('Hierarchical Clustering Dendrogram (truncated)')
            pltx.xlabel('sample index or (cluster size)')
            pltx.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    pltx.plot(x, y, 'o', c=c)
                    pltx.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                  textcoords='offset points',
                                  va='top', ha='center')
            if max_d:
                if kwargs.get('orientation', 'X') == 'right':
                    plt.axvline(x=max_d, c='k')
                else:
                    plt.axhline(y=max_d, c='k')
        return ddata

    est = linkage(df, 'ward')

    # calculate full dendrogram
    pltx.figure(figsize=(35.0, 30.0))
    fancy_dendrogram(
        est,
        truncate_mode='lastp',
        p=150,
        leaf_font_size=10.,
        orientation='right',
        show_contracted=True,
        annotate_above=0,  # useful in small plots so annotations don't overlap
        max_d=1.7,
    )

    image_dest = general_path + '\\' + 'agglomerative_clustering' + '.jpeg'
    pltx.savefig(image_dest)
    pltx.close()
    logger.log_print("Image of agglomerative clustering:\t{}".format(image_dest))

    df_results = pd.DataFrame(index=[n for n in df.index])
    for k in range(min_num_of_clusters, max_num_of_clusters):
        logger.log_print('Calculating agglomerative clustering for k = {}'.format(k))
        labels = fcluster(est, k, criterion='maxclust')
        current = pd.Series(labels, index=df.index)
        df_results['n = {}'.format(k)] = current
    dump(df_results, output_file_name, sheet_name='Summary')


global xl_writers
xl_writers = dict()


def dump(df, fname='vec_norm', path=None, sheet_name='vectorized'):
    if path is None:
        excel_full_path = 'C:\\Repos\\SongReader\\Vector' + '\\' + fname + '.xlsx'
    else:
        excel_full_path = path + '\\' + fname + '.xlsx'

    xl_writer = xl_writers.get(excel_full_path,
                               pd.ExcelWriter(excel_full_path, engine='xlsxwriter', options={'encoding': 'utf-8'}))
    logger.log_print("To excel at {}\tTab: {}".format(excel_full_path, sheet_name))

    df.to_excel(xl_writer, sheet_name=sheet_name)
    xl_writers[excel_full_path] = xl_writer


def close_all_xl_files(xl_writers_t):
    for k, v in xl_writers_t.iteritems():
        v.close()


if __name__ == '__main__':
    logger = Logger()
    logger.initThread()

    general_path = r'C:\Repos\SongReader\Vector'
    excel_file = general_path + '\\' + 'vec.xlsx'
    raw_data_file = general_path + '\\' + 'Raw_data.xlsx'

    df = pd.read_excel(excel_file, options={'encoding': 'utf-8'})
    df.set_index('Artist_name', inplace=True)
    df = df.fillna(value=0)
    del df['indInf']
    df_norm = (df - df.mean()) / (df.max() - df.min())
    dump(df_norm, 'vec_norm')
    kmean(df_norm, general_path)
    hierarchy_cluster(df_norm, general_path)
    close_all_xl_files(xl_writers_t=xl_writers)
    logger.log_close()
