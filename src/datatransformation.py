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

movies = pd.read_csv('../data/raw/movies.csv')
ratings = pd.read_csv('../data/raw/ratings.csv')
tags = pd.read_csv('../data/raw/tags.csv')

movies['genres'] = movies['genres'].str.lower()
ratings = ratings[['movieId', 'rating']]
tags = tags[['movieId', 'tag']]
tags['tag'] = tags['tag'].str.lower()

movies['year'] = movies['title'].str[-5:-1]

movies.loc[movies.year == '973)', 'year'] = '1973'
movies.loc[movies.year == '995)', 'year'] = '1995'
movies.loc[movies.year == '998)', 'year'] = '1998'
movies.loc[movies.year == '999)', 'year'] = '1999'
movies.loc[movies.year == '008)', 'year'] = '2008'
movies.loc[movies.year == '007)', 'year'] = '2007'
movies.loc[movies.year == '011)', 'year'] = '2011'
movies.loc[movies.year == '012)', 'year'] = '2012'
movies.loc[movies.year == '014)', 'year'] = '2014'

movies.loc[movies.year == 'lon ', 'year'] = '2007'
movies.loc[movies.year == 'r On', 'year'] = '2018'
movies.loc[movies.year == ' Roa', 'year'] = '2015'
movies.loc[movies.year == 'atso', 'year'] = '1980'
movies.loc[movies.year == 'imal', 'year'] = '2016'
movies.loc[movies.year == 'erso', 'year'] = '2016'
movies.loc[movies.year == 'ligh', 'year'] = '2016'
movies.loc[movies.year == 'he O', 'year'] = '2016'
movies.loc[movies.year == 'osmo', 'year'] = '2014'
movies.loc[movies.year == ' Bab', 'year'] = '2017'
movies.loc[movies.year == 'ron ', 'year'] = '2017'
movies.loc[movies.year == 'irro', 'year'] = '2011'

movies['year'] = movies['year'].astype('int32')

bins = [1899, 1909, 1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
movies['decade'] = pd.cut(movies['year'], bins, labels=labels)
years = movies.pop('year')

movies['decade'] = movies['decade'].cat.codes / 11

for i in movies['movieId']:
    movies.loc[movies.movieId == i, 'averageRating'] = ratings[ratings.movieId == i].rating.mean()
    
movies['averageRating'] = movies['averageRating'].fillna(movies['averageRating'].mean())

allTags = tags['tag'].value_counts().head(184)

movies['totalTags'] = 0
for tag, count in allTags.items():
    for i in movies['movieId']:
        movies.loc[movies.movieId == i, ('tag_' + tag)] = ((tags.movieId == i) & (tags.tag == tag)).sum()
        movies.loc[movies.movieId == i, 'totalTags'] += movies.loc[movies.movieId == i, ('tag_' + tag)]

tagCols = [col for col in movies.columns if 'tag_' in col]
for col in tagCols:
    for i in movies['movieId']:
        if movies.loc[movies.movieId == i, 'totalTags'].all() != 0:
            movies.loc[movies.movieId == i, col] /= movies.loc[movies.movieId == i, 'totalTags']
totalTags = movies.pop('totalTags')

allGenres = movies['genres'].str.split('|', expand=True).stack().value_counts()

for genre, count in allGenres.items():
    movies['genre_' + genre] = movies['genres'].str.count(genre)
    
genres = movies.pop('genres')

movies.to_csv('../data/processed/movies_transformed.csv')

df = pd.read_csv('../data/processed/movies_transformed.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df.describe())
print(df.corr())

df['averageRating'].plot.hist(title='Histogram of Average Movie Ratings')
plt.show()

df.plot.scatter(x='decade', y='averageRating', title='Average Movie Rating vs. Movie Decade')
plt.show()

df.plot.scatter(x='tag_in netflix queue', y='averageRating', title='Average Movie Rating vs. Movies Tagged in a User\'s Netflix Queue')
plt.show()

df.plot.scatter(x='genre_documentary', y='averageRating', title='Average Movie Rating vs. Movies Listed in Documentary Genre')
plt.show()
