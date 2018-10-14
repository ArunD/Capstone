# Machine Learning Engineer Nanodegree
## Capstone Project
Arun Dakua
September 15 Oct, 2018

## I. Definition

### Project Overview
Promotion of Indian Regional language movies (Movie Recommendation System for Indian movies). More than 2000 movies are launched every year in India. Most of them are aware only of Bollywood movies,which compromises of one third of the movies. There has been a lot of development in Gujarathi,Bengali,Marathi,Telugu,Tamil and Kannada movies,which could be enjoyed by audience who are exposed to only Bollywood movies. Currently there are no application or website which would suggest people of movies apart from Bollywood. The motto here is to provide a service which can recommend Indian regional movies. 

### Problem Statement

The project is inspired by Movielens(http://movielens.org). MovieLens is a web site that helps people find movies to watch. It has hundreds of thousands of registered users. It conducts online field experiments in MovieLens in the areas of automated content recommendation, recommendation interfaces, tagging-based recommenders and interfaces, member-maintained databases, and intelligent user interface design. Currently ,an open source database for Indian movies is not available.We are working on it to collect data from IMDB and various production house. 

Movie recommendation system for Indian movies.Currently no recommmendation system has been created for Indian movies. The solution to use deep learning on data provided by Grouplens(https://grouplens.org/datasets/movielens/) and replicate on India Movie database after it is created.

### Metrics

To evaluate accuracy of predicted ratings I will use Root Mean Squared Error (RMSE). 
(https://en.wikipedia.org/wiki/Root-mean-square_deviation) The root-mean-squared error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences. These deviations are called residuals when the calculations are performed over the data sample that was used for estimation and are called errors (or prediction errors) when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSD is a measure of accuracy, to compare forecasting errors of different models for a particular dataset and not between datasets, as it is scale-dependent.

## II. Analysis

### Data Exploration

Dataset is taken from Grouplens(https://grouplens.org/datasets/movielens/). Structure of files :

==> movies.csv <== movieId,title,genres 
		   1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy 2,Jumanji (1995),Adventure|Children|Fantasy 3,Grumpier Old Men (1995),Comedy|Romance 4,Waiting to Exhale (1995),Comedy|Drama|Romance

==> ratings.csv <== userId,movieId,rating,timestamp 
		    1,110,1.0,1425941529 1,147,4.5,1425942435 1,858,5.0,1425941523 1,1221,5.0,1425941546

==> tags.csv <== userId,movieId,tag,timestamp 
		 1,318,narrated,1425942391 20,4306,Dreamworks,1459855607 20,89302,England,1400778834 20,89302,espionage,1400778836


### Exploratory Visualization

ISSUE WHAT TO PUT HERE.


### Algorithms and Techniques

Collaberative filtering in Keras.

The idea of using deep learning is similar to that of Matrix Factorization. The idea behind matrix factorization is to represent users and items in a lower dimensional latent space . (https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. This family of methods became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in his 2006 blog post, where he shared his findings with the research community.

For deep learning implementation, we don’t need them to be matrix form, we want our model to learn the values of embedding matrix itself. The user latent features and movie latent features are looked up from the embedding matrices for specific movie-user combination. These are the input values for further linear and non-linear layers. We can pass this input to multiple relu, linear or sigmoid layers and learn the corresponding weights by any optimization algorithm (Adam, SGD, etc.).

The main components of my neural network:
1.A left neural network layer that creates Users matrix. A right neural network layer that creates Movies matrix. The input to the left layer are user data.
The input to the right layer are movies data .
A merge layer that takes the dot product of these two vectors to return the predicted rating.

2.This code is based on the approach outlined in Alkahest’s blog post Collaborative Filtering in Keras.

3.Compile the model using Mean Squared Error (MSE) as the loss function and the AdaMax learning algorithm.

4.Split the training and test data in 90/10.

5.Train the model on different epochs. Callbacks monitor the validation loss Save the model weights each time the validation loss has improved

6.The next step is to actually predict the ratings a  user will give to a random movie.

7.Movies with top 10 predicted ratings are displayed to the user

### Benchmark

As the project is inspired by GroupLens,we will use the open soure library to create recommender system LensKit(https://lenskit.org/). 
Their existing project((https://github.com/lenskit/lenskit-hello).) allows anyone to create a movie recommender system on top of the code and data.
The output will be considered as the benchmark.

Comparison will be made between Deep Learning Model recommendation and Lenskit.


## III. Methodology

### Data Preprocessing

There was no data pre processing required .
The data provided by  GroupLens(http://files.grouplens.org/datasets/movielens/ml-latest.zip) is sufficient.
As the recommender system provided by GroupLens is the benchmark and we need to use the same data on both the system ,I didn't modify the data.


### Implementation

A single code file has been created Movie_Recommender_DL.py to define and train the model.
The file also includes the code for recommending movies for a user.

Workflow Diagram:




Defining the model : 
2 sequential models merged by dot product.
The 2 sequential model and dot product are used to instaniate a model.
Each sequential model has layer of 1 embedding,3 convolution and 2 fully connected layers.

Training the model :
Training and validation data is stored in the memory.
The ratio between training and validation is kept to 90/10.
Model is compiled using loss function mean square error and optimizer adamax.
While fiiting the model ,number of epochs is set to 30 and batch size to 1000.
The best weights are stored in movie-weights.h5

Movie recommendation for a user :
Model is used to predict the ratings on movie that the user has already provided for comparison 
Model is used to predict the ratings on movies that the user has not provided.
Top 10 movies with highest predicted rating value are displayed.


### Refinement

The initial model menitoned in collaberative filtering in keras(http://www.fenris.org/2016/03/07/index-html) involved only an embedded and reshape layers which gave a high root mean square error.

The error was gradually reduced by adding layers of convolution,maxpool,dropout ,flatten and fully connected layers.

right.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))
right.add(MaxPooling1D(pool_size=1))
right.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
right.add(MaxPooling1D(pool_size=1))
right.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
right.add(MaxPooling1D(pool_size=1))
right.add(Dropout(0.3))
right.add(Flatten())
right.add(Dense(500, activation='relu'))
right.add(Dropout(0.4))
right.add(Dense(10, activation='sigmoid'))


## IV. Results

### Model Evaluation and Validation

The best validation loss : 0.7970
Mean square root error : 0.898
 
The final architecure and hyperparameters were chosen because they performed the best among  the tried combinations.
Refer the figure for the workflow diagram.

For complete descritpion of the model and description please refer the figure.

1.The embedding layer receives max user id and max movie id as input to create dense vectors.The number of latent factors considered is 100.
2.The first convolutional layer has 16 filters and a stride of 2.
3.Max Pool layer with pool size of 1
4.The second convolutional layer has 32 filters and a stride of 2.
5.Max Pool layer with pool size of 1
6.The third convolutional layer has 64 filters and a stride of 2.
7.Max Pool layer with pool size of 1
8.Dropout layer of 0.3
9.Flatten the layer
10.Fully connected layer with relu activation.
11.Dropout layer of 0.4
12.Fully connected layer with sigmoid activation.
13.The same model is used for left and right layer .
14.Dot product between left and right layer to predict ratings.

### Justification

The output provided by the benchmark grouplens algorithim is provided below.

Below are the output for User Id 1.

MOVIELENS:
Top 10 movies Recommended by movielens.

Title							 Genres					Score	
C.R.A.Z.Y. (2005)					 Drama					5.63
World of Apu, The (Apur Sansar) (1959)			 Drama					5.59
All Things Fair (Lust och fägring stor) (1995)		 Drama|Romance|War			5.58
Ballad of Narayama, The (Narayama Bushiko) (1958)	 Drama					5.57
Lumumba (2000)						 Drama					5.57
Neon Bible, The (1995)					 Action|Animation|Drama|Fantasy|Sci-Fi	5.56
Jupiter's Wife (1994)					 Documentary				5.53
Some Mother's Son (1996)				 Drama					5.53
Get Real (1998)						 Drama|Romance				5.51
Harakiri (Seppuku) (1962)				 Drama					5.50

MACHINE LEARNING:
Predicted ratings for the movies that are already provided by user.

Title					Genres						Rating	Prediction
Toy Story (1995)			Adventure|Animation|Children|Comedy|Fantasy	5	4.015676
Shawshank Redemption, The (1994)	Crime|Drama					5	4.577501
Like Water for Chocolate 	 	Drama|Fantasy|Romance				5	4.035459
Natural Born Killers (1994)		Action|Crime|Thriller				5	3.507
Pulp Fiction (1994)			Comedy|Crime|Drama|Thriller			5	4.294444
Quiz Show (1994)			Drama						5	3.879982
Mission: Impossible (1996)		Action|Adventure|Mystery|Thriller		5	3.626861
Forrest Gump (1994)			Comedy|Drama|Romance|War			5	4.250167
Jurassic Park (1993)			Action|Adventure|Sci-Fi|Thriller		5	3.725909
Philadelphia (1993)			Drama						5	3.915203

Movies Recommended and predicted ratings by machine learning.

Title					Genres						Prediction
Godfather, The (1972)			Crime|Drama					4.571756
Ikiru (1952)				Drama						4.56005
Central Station (1998)			Drama						4.556155
NausicaÃ¤ of the Valley of the Wind	Adventure|Animation|Drama|Fantasy|Sci-Fi	4.53071
Band of Brothers (2001)			Action|Drama|War				4.519944
Rashomon (RashÃ´mon) (1950)		Crime|Drama|Mystery				4.518548
Grand Illusion (1937)			Drama|War					4.517025
Hedwig and the Angry Inch (2000)	Comedy|Drama|Musical				4.516345
Killing, The (1956)			Crime|Film-Noir					4.516186
Corporation, The (2003)			Documentary					4.515452

The output of 2 recommender doesn't match .
It doesn't give a clear evidence that recommendation by deep learning methods will be accepted by the user.


## V. Conclusion

### Free-Form Visualization

The model doesn't take first time user into consideration.Though we can recommend the top 20 movies that are being watched.
Users who have rated less number of movies.
If a user has rated only 1 or 2 movies recommending movies wouldn't be possible ,as the model won't be able to generate patterns.


### Reflection

There is an existing solution for movie recommendation which is provided by MovieLens(https://grouplens.org/) which is freely available and can be easily implemented in a application or website.
MovieLens is a web site that helps people find movies to watch. It has hundreds of thousands of registered users. They conduct online field experiments.In MovieLens in the areas of automated content recommendation, recommendation interfaces, tagging-based recommenders and interfaces, member-maintained databases, and intelligent user interface design.

The goal of the project is to use the data provided by MovieLens and use deep learning methods to create a movie recommender system.
The output of the project would then be compared with benchmark output provided by MovieLens.

1.The data was downloaded(https://grouplens.org/datasets/movielens/) and preprocessed.
2.A benchmark was created for deep learning recommender system using MovieLens(https://github.com/lenskit/lenskit-hello).
3.The User and movie model has 3 convolutional layers and 2 fully connected layers.
3.The model was trained using the data multiple times until a good set of parameter was found.

Though a recommender system has been created using Machine Learning ,to truly validate we would need to test with real users .
A sample of users can be recommended with GroupLens recommendation engine and other sample of users could be recommended with Machine Learning system.
This will help us understand if it truly works and how much improvement is needed.

### Improvement








