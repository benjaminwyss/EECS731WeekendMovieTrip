import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
plt.close('all')
import warnings
warnings.filterwarnings('ignore')
import random as rand

df = pd.read_csv('../data/processed/movies_transformed.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

array = df.values
X = array[:, 2:]

dbsModel = DBSCAN(eps=0.8, min_samples=2).fit(X)
dbsLabels = dbsModel.labels_
df['cluster_dbs'] = pd.Series(dbsLabels)

msModel = MeanShift().fit(X)
msLabels = msModel.labels_
df['cluster_ms'] = pd.Series(msLabels)

aggModel = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5).fit(X)
aggLabels = aggModel.labels_
df['cluster_agg'] = pd.Series(aggLabels)

dbsCluster1 = df.loc[df.movieId == 110603, 'cluster_dbs'].values[0]
dbsCluster2 = df.loc[df.movieId == 260, 'cluster_dbs'].values[0]
dbsCluster3 = df.loc[df.movieId == 1, 'cluster_dbs'].values[0]
dbsCluster4 = df.loc[df.movieId == 176601, 'cluster_dbs'].values[0]
dbsCluster5 = df.loc[df.movieId == 85788, 'cluster_dbs'].values[0]

aggCluster1 = df.loc[df.movieId == 110603, 'cluster_agg'].values[0]
aggCluster2 = df.loc[df.movieId == 260, 'cluster_agg'].values[0]
aggCluster3 = df.loc[df.movieId == 1, 'cluster_agg'].values[0]
aggCluster4 = df.loc[df.movieId == 176601, 'cluster_agg'].values[0]
aggCluster5 = df.loc[df.movieId == 85788, 'cluster_agg'].values[0]

print(df[df.cluster_dbs == dbsCluster1])
print(df[df.cluster_agg == aggCluster1])

print(df[df.cluster_dbs == dbsCluster2])
print(df[df.cluster_agg == aggCluster2])

print(df[df.cluster_dbs == dbsCluster3])
print(df[df.cluster_agg == aggCluster3])

print(df[df.cluster_dbs == dbsCluster4])
print(df[df.cluster_agg == aggCluster4])

print(df[df.cluster_dbs == dbsCluster5])
print(df[df.cluster_agg == aggCluster5])

df = df[['title', 'averageRating', 'cluster_agg']].sort_values(by=['cluster_agg'])
print(df)
df.to_csv('../data/processed/movie_clusters.csv')
