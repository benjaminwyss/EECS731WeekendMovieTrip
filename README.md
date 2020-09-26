# EECS 731 Project 3: Weekend Movie Trip
Submission by Benjamin Wyss

## Project Overview

Examining MovieLens movie data to build a clustering model which can recommend movies similar to other movies

### Data Sets Used

MovieLens Small Datasets - Movies, Ratings, and Tags - Taken from: https://grouplens.org/datasets/movielens/ on 9/23/20

The small data sets were selected because they will reduce overall model complexity while still providing a sufficiently large number of samples (9742 movie samples)

### Results

Multiple machine learning clustering models were created and assigned clusters to movies based on their similarity to other movies in terms of ratings, time period, genres, and user tags. Out of these models, an agglomerative model generated more specific and higher quality recommendations than all other tested models. This model generated a total of 1236 clusters which contain at least two similar movies. All of the examined movies and their corresponding agglomerative clusters are recorded in the data/processed/movie_clusters.csv file which can be used to provide movie recommendations by examining movies that are clustered with movies one likes. My full process and results are documented in the notebooks folder of this project.