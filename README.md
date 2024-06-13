# Movie Recommendation System

## Overview
This project is part of Task 3 at Bharat Intern. It involves building a Movie Recommendation System using collaborative filtering and machine learning techniques in Python. The system utilizes the MovieLens dataset and employs the SVD algorithm from the Surprise library to recommend movies to users based on their past ratings.

## Dataset
The MovieLens dataset contains millions of ratings and tag applications applied to thousands of movies by users. For this project, the small version of the dataset (`ml-latest-small`) is used.

## Installation
To run this project, you need to have Python installed. You can install the required libraries using pip:
pip install numpy pandas scikit-learn surprise matplotlib seaborn

## Usage

### Download the Dataset
!wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

### Load the Data
import pandas as pd

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
data = pd.merge(ratings, movies, on='movieId')

## Visualizations

### Distribution of Ratings
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data['rating'], bins=30, kde=False)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

### Number of Ratings per Movie
movie_ratings_count = data.groupby('title').size().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=movie_ratings_count.head(20).values, y=movie_ratings_count.head(20).index, palette='viridis')
plt.title('Top 20 Most Rated Movies', fontsize=16)
plt.xlabel('Number of Ratings', fontsize=14)
plt.ylabel('Movie Title', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

## Model Training

### Prepare Data for Surprise Library
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split

reader = Reader(rating_scale=(0.5, 5.0))
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
trainset, testset = surprise_train_test_split(dataset, test_size=0.2)

### Train the SVD Model
svd = SVD(n_epochs=20, lr_all=0.005, reg_all=0.4)  # Adjust parameters for quicker training
svd.fit(trainset)

## Evaluation

### Evaluate the Model
from surprise import accuracy

predictions = svd.test(testset)
print(f'RMSE: {accuracy.rmse(predictions)}')

## Hyperparameter Tuning

### GridSearchCV (Optional but Time-consuming)
from surprise.model_selection import GridSearchCV

param_grid = {
    'n_epochs': [20],
    'lr_all': [0.002],
    'reg_all': [0.4, 0.5]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=2)
gs.fit(dataset)

# Best parameters
print(gs.best_params['rmse'])

# Train the best model
best_svd = gs.best_estimator['rmse']
best_svd.fit(trainset)

## Recommendations

# Get Recommendations for a Specific User
user_id = 2  # Example user ID
recommendations = get_movie_recommendations(user_id, num_recommendations=10)
print(recommendations)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.
