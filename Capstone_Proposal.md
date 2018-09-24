
Machine Learning Engineer Nanodegree
Capstone Proposal

Arun Dakua Septmeber 23rd, 2018
Proposal

Promotion of Indian Regional language movies (Movie Recommendation System for Indian movies). More than 2000 movies are launched every year in India. Most of them are aware only of Bollywood movies,which comprises of one third of the movies. There has been a lot of development in Gujarathi,Bengali,Marathi,Telugu,Tamil and Kannada movies,which could be enjoyed by audience who are exposed to only Bollywood movies. Currently there are no application or website which would suggest people of movies apart from Bollywood. The motto here is to provide a service which can recommend Indian regional movies.
Domain Background

The project is inspired by Movielens(http://movielens.org). MovieLens is a web site that helps people find movies to watch. It has hundreds of thousands of registered users. It conducts online field experiments in MovieLens in the areas of automated content recommendation, recommendation interfaces, tagging-based recommenders and interfaces, member-maintained databases, and intelligent user interface design. Currently ,an open source database for Indian movies is not available.We are working on it to collect data from IMDB and various production house.
Problem Statement

Movie recommendation system for Indian movies.Currently no recommmendation system has been created for Indian movies. The solution to use deep learning on data provided by Grouplens(https://grouplens.org/datasets/movielens/) and replicate on India Movie database after it is created.

Quantifiable : Based on ratings provided by user a user profile is created. Based on tags of the movies relevance between the movies are created. Using user profile and movies relevance recommendation are made to the user.

Measurable : We can use the existing movielens service to test the recommendation made by the engine I developed.

Replicable: The problem can be reproduced and occurs more than once.
Datasets and Inputs

Dataset is taken from Grouplens(https://grouplens.org/datasets/movielens/). Structure of files :

==> movies.csv <== movieId,title,genres 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy 2,Jumanji (1995),Adventure|Children|Fantasy 3,Grumpier Old Men (1995),Comedy|Romance 4,Waiting to Exhale (1995),Comedy|Drama|Romance

==> ratings.csv <== userId,movieId,rating,timestamp 1,110,1.0,1425941529 1,147,4.5,1425942435 1,858,5.0,1425941523 1,1221,5.0,1425941546

==> tags.csv <== userId,movieId,tag,timestamp 1,318,narrated,1425942391 20,4306,Dreamworks,1459855607 20,89302,England,1400778834 20,89302,espionage,1400778836

==> genome-tags.csv <== tagId,tag 1,007 2,007 (series) 3,18th century 4,1920s

==> genome-scores.csv <== movieId,tagId,relevance 1,1,0.024749999999999994 1,2,0.024749999999999994 1,3,0.04899999999999999 1,4,0.07750000000000001
Solution Statement

Collaberative filtering in Keras.

The idea of using deep learning is similar to that of Matrix Factorization. The idea behind matrix factorization is to represent users and items in a lower dimensional latent space . (https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. This family of methods became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in his 2006 blog post, where he shared his findings with the research community.

For deep learning implementation, we don’t need them to be matrix form, we want our model to learn the values of embedding matrix itself. The user latent features and movie latent features are looked up from the embedding matrices for specific movie-user combination. These are the input values for further linear and non-linear layers. We can pass this input to multiple relu, linear or sigmoid layers and learn the corresponding weights by any optimization algorithm (Adam, SGD, etc.).
Benchmark Model

Gouplens provides a open soure library to create recommender system LensKit(https://lenskit.org/). We can feed the same data through LensKit and observe if the recommendation provided by engine I created is in par with LensKit.
Evaluation Metrics

To evaluate accuracy of predicted ratings I will use Root Mean Squared Error (RMSE). (https://en.wikipedia.org/wiki/Root-mean-square_deviation) The root-mean-squared error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences. These deviations are called residuals when the calculations are performed over the data sample that was used for estimation and are called errors (or prediction errors) when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSD is a measure of accuracy, to compare forecasting errors of different models for a particular dataset and not between datasets, as it is scale-dependent.
Project Design

The main components of my neural network:

1.A left neural network layer that creates Users matrix. A right neural network layer that creates Movies matrix. The input to the left layer are user and rating data The input to the right layer are movies and tags data A merge layer that takes the dot product of these two vectors to return the predicted rating.

2.This code is based on the approach outlined in Alkahest’s blog post Collaborative Filtering in Keras.

3.Compile the model using Mean Squared Error (MSE) as the loss function and the AdaMax learning algorithm.

4.Split the training and test data in 80/20.

5.Train the model on different epochs. Callbacks monitor the validation loss Save the model weights each time the validation loss has improved

6.The next step is to actually predict the ratings a random user will give to a random movie.
