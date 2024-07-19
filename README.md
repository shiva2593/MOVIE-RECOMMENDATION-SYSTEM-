# MOVIE-RECOMMENDATION-SYSTEM-
# <font color="orange">Movie Recommendation System </font>

### Importing the basic libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

### Importing & Parsing the dataset as ratings and movies details

ratingData = pd.read_table('ratings.dat', 

names=['user_id', 'movie_id', 'rating', 'time'],engine='python', 

delimiter='::',encoding="ISO-8859-1")

movieData = pd.read_table('movies.dat',names=['movie_id', 'title', 

'genre'],engine='python',

 delimiter='::',encoding="ISO-8859-1")

### Basic Inspection on datasets

# Top 5 rows of movie data

movieData.head(10)

movieData.title[0]

# Top 5 rows of rating data

ratingData.head()

r,c=ratingData.shape

print("rating data having {} rows {} columns".format(r,c))

r,c=movieData.shape

print("movie data having {} rows {} columns".format(r,c))

movieData.size

ratingData.size

print("columns in the movie data: {}".format(list(movieData.columns)))

print('columns in the rating data: ',list(ratingData.columns))

len(movieData.movie_id.unique())

len(ratingData.movie_id.unique())

ratingData.info()

movieData.info()
movieData.describe()

ratingData.describe()

# Checking null values

def checknull(obj):

 return obj.isnull().sum()

movieData.apply(checknull)

ratingData.apply(checknull)

# Checking duplicate values

def checkduplicate(obj):

 return obj.duplicated().sum()

movieData.apply(checkduplicate)

ratingData.apply(checkduplicate)

### Create the ratings matrix of shape (m×u)

ratingData.movie_id.values

np.max(ratingData.movie_id.values)

ratingData.user_id.values

np.max(ratingData.user_id.values)

ratingMatrix = np.ndarray(

 shape=(np.max(ratingData.movie_id.values), 

np.max(ratingData.user_id.values)),

 dtype=np.uint8)

ratingData.movie_id.values-1

ratingData.user_id.values-1

ratingData.rating.values

ratingMatrix[ratingData.movie_id.values-1, ratingData.user_id.values-1] = 

ratingData.rating.values

print(ratingMatrix)

### Subtract Mean off - Normalization

np.mean(ratingMatrix)

np.mean(ratingMatrix, 1)

np.mean(ratingMatrix, 1).shape

np.asarray(np.mean(ratingMatrix, 1))

np.asarray(np.mean(ratingMatrix, 1)).shape
normalizedMatrix = ratingMatrix - np.asarray([(np.mean(ratingMatrix, 

1))]).T

print(normalizedMatrix)

### Computing SVD

normalizedMatrix.T

ratingMatrix.shape[0] - 1

np.sqrt(ratingMatrix.shape[0] - 1)

A = normalizedMatrix.T / np.sqrt(ratingMatrix.shape[0] - 1)

A

U, S, V = np.linalg.svd(A)

### Calculate cosine similarity, sort by most similar and return the top N

def similar(ratingData, movie_id, top_n):

 index = movie_id - 1 # Movie id starts from 1

 movie_row = ratingData[index, :]

 magnitude = np.sqrt(np.einsum('ij, ij -> i', ratingData, ratingData)) 

#Einstein summation | traditional matrix multiplication and is equivalent 

to np.matmul(a,b)

 similarity = np.dot(movie_row, ratingData.T) / (magnitude[index] * 

magnitude)

 sort_indexes = np.argsort(-similarity) #Perform an indirect sort along 

the given axis (Last axis)

 return sort_indexes[:top_n]

### Select k principal components to represent the movies, a movie_id to 

find recommendations and print the top_n results

k = int(input("enter the total number of movies: "))

print(" ")

movie_id = int(input("enter the movie id: "))

print(" ")

print("the entered movie id name is : 

{}".format(movieData.title[movie_id]))

print(" ")

top_n = int(input("ton n movies: "))

sliced = V.T[:, :k] # representative data

indexes = similar(sliced, movie_id, top_n)

print(" ")

print('Recommendations for Movie {0}: \n'.format(

movieData[movieData.movie_id == movie_id].title.values[0]))

for id in indexes + 1:

 print(movieData[movieData.movie_id == id].title.values[0])

#### <font color="red">Conclusions:</font>

<font color="green"></font>
* <font color="green">Here The Recommendation System is Developed for List 

of N Movies</font>

* <font color="green">Movie Recommendation System is Developed Based on 

Collabarating Based Recommendation</font>

* <font color="green">We Have to Give K Number of Features, Movie Id,Top N 

as Input and it Recommends Top N Movies as Output</font>

* <font color="green">These Top N Movies Recommended Using Collabarating 

Based Filtering Technique with Cosine Similarity and SVD</font>
