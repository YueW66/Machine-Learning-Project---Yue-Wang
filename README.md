# Overview
This repository contains the code and report for our machine learning assignment as part of our coursework. The goal of this assignment was to participate in a learning challenge, predict a momentary self-reported well-being score while people were playing a video game, and rank our predictions against other groups. We achieved a ranking of 5th out of 46 groups.

# Project Description
## Learning Challenge
The task was to predict the well-being scores of users based on their interaction with a game designed to lower stress and improve mental health. We were required to submit our predictions and a detailed report describing our approach, solutions, and the code used for predictions.
## Submission
Rank: 5th out of 46 groups

Platform: Codalab

# My Contributions
## Feature Engineering
Feature Importance: Conducted permutation feature importance analysis to identify the most influential features.

Feature Engineering: Engineered features such as CurrentSessionLength, Day, Time, StandardizedProgression, ResponseValue_mean, ResponseValue_median, ResponseValue_std.
## Model Development
Model Selection: Implemented and tested various models including Random Forest, Logistic Regression, Support Vector Regression (SVR), Gradient Boosting, and Stochastic Gradient Descent Regression (SGDR).

Hyperparameter Tuning: Conducted hyperparameter tuning for the Random Forest model using GridSearchCV to optimize performance based on Mean Absolute Error (MAE).
## Data Processing
Test Dataset Processing: Ensured the correct order of data in the test dataset and handled any necessary preprocessing to maintain data integrity.
## Code Cleanup
Final Code Preparation: Cleaned and organized the final code for submission, ensuring readability and functionality.
