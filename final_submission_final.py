#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.inspection import permutation_importance
import zipfile
# from sklearn.impute import SimpleImputer
# from sklearn.utils import resample
# import featuretools as ft


# # The following code can achieve our best results.

# In[ ]:


# Load the datasets
def import_data_from_csv():
    df_train = pd.read_csv('train_data.csv')
    df_test = pd.read_csv('test_data.csv')
    return df_train, df_test
train_data, test_data = import_data_from_csv()


# In[ ]:


# fill missing values
train_data['LastTaskCompleted'].fillna('Not_Saved', inplace=True)
train_data['CurrentTask'].fillna('Not_Playing', inplace=True)
train_data['LevelProgressionAmount'].fillna(0, inplace=True)

test_data['LastTaskCompleted'].fillna('Not_Saved', inplace=True)
test_data['CurrentTask'].fillna('Not_Playing', inplace=True)
test_data['LevelProgressionAmount'].fillna(0, inplace=True)


# In[ ]:


# Remove outliers from training data
def remove_outliers(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
train_data = remove_outliers(train_data)


# In[ ]:


# 'TimeUtc': According to feature "TimeUtc", we extract the date and time information and create two new features "Day" and "Time"
# "Day" should be a categorical feature and include monday, tuesday, wednesday, thursday, friday, saturday, and sunday, and then it gets encoded
# "Time" should be a categorical feature and include night, daytime, and evening, and then it gets encoded
train_data['TimeUtc'] = pd.to_datetime(train_data['TimeUtc'])
train_data['Day'] = train_data['TimeUtc'].dt.day_name()
# train_data_1['Day'] = pd.Categorical(train_data_1['Day'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
train_data['Time'] = train_data['TimeUtc'].dt.hour
train_data['Time'] = pd.cut(train_data['Time'], bins=[-1, 5, 17, 23], labels=['night', 'daytime', 'evening'])
# encode 'Day': Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
train_data['Day'] = train_data['Day'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})
train_data['Time'] = train_data['Time'].map({'night': 0, 'daytime': 1, 'evening': 2})
# change 'Day' and 'Time' to int
train_data['Day'] = train_data['Day'].astype(int)
train_data['Time'] = train_data['Time'].astype(int)


# In[ ]:


test_data['TimeUtc'] = pd.to_datetime(test_data['TimeUtc'])
test_data['Day'] = test_data['TimeUtc'].dt.day_name()

test_data['Time'] = test_data['TimeUtc'].dt.hour
test_data['Time'] = pd.cut(test_data['Time'], bins=[-1, 5, 17, 23], labels=['night', 'daytime', 'evening'])

test_data['Day'] = test_data['Day'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})
test_data['Time'] = test_data['Time'].map({'night': 0, 'daytime': 1, 'evening': 2})

test_data['Day'] = test_data['Day'].astype(int)
test_data['Time'] = test_data['Time'].astype(int)


# In[ ]:


# Standardize the numerical feature
scaler = StandardScaler()
train_data[['Day', 'Time', 'CurrentSessionLength']] = scaler.fit_transform(train_data[['Day', 'Time', 'CurrentSessionLength']])
test_data[['Day', 'Time', 'CurrentSessionLength']] = scaler.fit_transform(test_data[['Day', 'Time', 'CurrentSessionLength']])

train_data['StandardizedProgression'] = scaler.fit_transform(train_data[['LevelProgressionAmount']])
test_data['StandardizedProgression'] = scaler.fit_transform(test_data[['LevelProgressionAmount']])


# In[ ]:


# Group by 'UserID' and calculate statistical features for 'ResponseValue'
user_stats = train_data.groupby('UserID')['ResponseValue'].agg(['mean', 'median', 'std']).reset_index()
user_stats.columns = ['UserID', 'ResponseValue_mean', 'ResponseValue_median', 'ResponseValue_std']


# In[ ]:


# Merge the new features back into the original dataframe
train_data = pd.merge(train_data, user_stats, on='UserID', how='left')


# In[ ]:


# Impute NaN values in ResponseValue_std with 0
train_data['ResponseValue_std'].fillna(0, inplace=True)


# In[ ]:


train_data['UserID'] = train_data['UserID'].str.replace('p', '').astype(int)
test_data['UserID'] = test_data['UserID'].str.replace('p', '').astype(int)


# In[ ]:


# Apply categorization to LastTaskCompleted and CurrentTask
special_tasks = [
    'MarsRover', 'MARS_MARSROVER', 'WASH_MarsRover', 'RECREATIONGROUND_MINIGOLF', 'SteamLocomotive',
    'WASH_SteamLocomotive', 'DESERT_STEAMLOCOMOTIVE', 'WASH_Fountain','RECREATIONGROUND_FOUNTAIN'
]


# In[ ]:


def categorize_task(task):

    if task == 'Not_Saved' or task == 'Not_Playing':
        return 0
    elif task in special_tasks:
        return 2
    else:
        return 1


# In[ ]:


train_data['LastTaskCompleted_Category'] = train_data['LastTaskCompleted'].apply(categorize_task)
train_data['CurrentTask_Category'] = train_data['CurrentTask'].apply(categorize_task)

test_data['LastTaskCompleted_Category'] = test_data['LastTaskCompleted'].apply(categorize_task)
test_data['CurrentTask_Category'] = test_data['CurrentTask'].apply(categorize_task)

# Verify the results
print(train_data[['LastTaskCompleted', 'CurrentTask', 'LastTaskCompleted_Category', 'CurrentTask_Category']].head())

print(test_data[['LastTaskCompleted', 'CurrentTask', 'LastTaskCompleted_Category', 'CurrentTask_Category']].head())


# In[ ]:


# Group by 'UserID' and calculate statistical features for 'CurrentSessionLength' and 'LevelProgressionAmount'
CSLstats = train_data.groupby('UserID')['CurrentSessionLength'].agg(['mean', 'median', 'std', 'max']).reset_index()
CSLstats.columns = ['UserID', 'CSL_mean', 'CSL_median', 'CSL_std', 'CSL_max']
LPAstats = train_data.groupby('UserID')['LevelProgressionAmount'].agg(['mean', 'median', 'std', 'max']).reset_index()
LPAstats.columns = ['UserID', 'LPA_mean', 'LPA_median', 'LPA_std', 'LPA_max']


# In[ ]:


# Merge the new features back into the original dataframe
train_data = pd.merge(train_data, CSLstats, on='UserID', how='left')
train_data = pd.merge(train_data, LPAstats, on='UserID', how='left')


# In[ ]:


# Impute NaN values in ResponseValue_std with 0
train_data['CSL_std'].fillna(0, inplace=True)

# Impute NaN values in ResponseValue_std with 0
train_data['LPA_std'].fillna(0, inplace=True)


# In[ ]:


# Filter out ResponseValue with only one instance
value_counts = train_data['ResponseValue'].value_counts()
filtered_train_data = train_data[train_data['ResponseValue'].isin(value_counts[value_counts > 1].index)]


# In[ ]:


# give test_data a new feature 'index'
test_data['index'] = test_data.index


# In[ ]:


# Selecting data
# Identify the unique users in the training set and test set
train_users = train_data['UserID'].unique()
test_users = test_data['UserID'].unique()

# Find the users in the test set that are not in the training set
new_users = set(test_users) - set(train_users)
new_users = list(new_users)


# In[ ]:


#Filter the test data to get the rows corresponding to these new users
new_users_data = test_data[test_data['UserID'].isin(new_users)]


# In[ ]:


# Find the users that are present in both the training and test sets
shared_users = set(train_users).intersection(set(test_users))
shared_users = list(shared_users)

# Filter the test data to get the rows corresponding to these shared users
shared_users_data = test_data[test_data['UserID'].isin(shared_users)]


# In[ ]:


# select the feature 'index' out of the new_users_data and shared_users_data
new_users_data_index = new_users_data['index']
shared_users_data_index = shared_users_data['index']


# In[ ]:


# Calculate historical features from the training data
train_user_stats = train_data.groupby('UserID')['ResponseValue'].agg([
    'mean',
    'median',
    'std',
    'min',
    'max',
    'count'
]).reset_index()

# Rename columns for clarity
train_user_stats.columns = [
    'UserID',
    'ResponseValue_mean',
    'ResponseValue_median',
    'ResponseValue_std',
    'ResponseValue_min',
    'ResponseValue_max',
    'ResponseValue_count'
]

# Merge historical statistics into the shared_users_data test set
shared_users_data = pd.merge(shared_users_data, train_user_stats, on='UserID', how='left')


# In[ ]:


# Prepare features and target variable for training
target = 'ResponseValue'
features = [
    'CurrentSessionLength', 'Day', 'Time', 'StandardizedProgression',
    'ResponseValue_mean', 'ResponseValue_median', 'ResponseValue_std']

X_train = train_data[features]
y_train = train_data[target]


# In[ ]:


# Prepare features for the test set
X_test = shared_users_data[features]


# In[ ]:


# hyperparameter tuning
# Define the parameter grid
param_grid = {
    'iterations': [100, 200, 500],
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
cat_model = CatBoostRegressor(random_state=42, verbose=0)

# Define the scoring metric
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=cat_model, param_grid=param_grid, cv=skf, scoring=scorer, n_jobs=-1)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Best MAE: {best_score}')


# In[ ]:


# Define the CatBoost model with specified parameters and specify a directory for temporary files
model = CatBoostRegressor(iterations=500,
                          learning_rate=0.1,
                          depth=10,
                          loss_function='MAE')


# In[ ]:


# Cross validation
# Define the number of folds for cross-validation
num_folds = 10
cross_validator = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation and calculate negative mean absolute error (MAE)
cv_scores = cross_val_score(model, X_train, y_train, cv=cross_validator, scoring='neg_mean_absolute_error')

# Convert negative MAE scores to positive
positive_cv_scores = -cv_scores

# Print the cross-validation scores and average MAE
print(f'Cross-validation MAE scores: {positive_cv_scores}')
print(f'Average MAE: {positive_cv_scores.mean()}')


# In[ ]:


#final model on the entire training dataset
final_model = model.fit(X_train, y_train)

# predictions on the test dataset
predictions = final_model.predict(X_test)


# In[ ]:


shared_users_data['Predicted_ResponseValue'] = predictions


# In[ ]:


shared_users_prediction = shared_users_data[['index', 'Predicted_ResponseValue']]


# In[ ]:


# Calculate the mean predicted response value for shared users
mean_predicted_response = shared_users_data['Predicted_ResponseValue'].mean()


# In[ ]:


mean_predicted_response


# In[ ]:


# Ensure no SettingWithCopyWarning by explicitly making a copy of the DataFrame slice
new_users_data_1 = new_users_data.copy()


# In[ ]:


new_users_data_1['Predicted_ResponseValue'] = mean_predicted_response


# In[ ]:


new_users_prediction = new_users_data_1[['index', 'Predicted_ResponseValue']]


# In[ ]:


# concatenate the predictions for the new users and shared users
final_predictions = pd.concat([shared_users_prediction, new_users_prediction])


# In[ ]:


# sort the final predictions by the index
final_predictions_sorted = final_predictions.sort_values(by='index')


# In[ ]:


# only keep the 'Predicted_ResponseValue' column
final_predictions_sorted_ResponseValue = final_predictions_sorted['Predicted_ResponseValue']


# In[ ]:


print(final_predictions_sorted_ResponseValue)


# In[ ]:


# save the predictions to a CSV file, but delete the column name
final_predictions_sorted_ResponseValue.to_csv('predicted.csv', index=False, header=False)

with zipfile.ZipFile('predicted.zip', 'w') as z:
    z.write('predicted.csv')


# In[ ]:





# # The following code is used for other ways of data cleaning, EDA, feature engineering, modeling, and hyperparameter tuning.

# In[ ]:


# Load the datasets
def import_data_from_csv():
    df_train = pd.read_csv('train_data.csv')
    df_test = pd.read_csv('test_data.csv')
    return df_train, df_test
train_data, test_data = import_data_from_csv()


# ### data cleaning

# In[ ]:


# 1. UserID
train_data['UserID'].value_counts()


# In[ ]:


# 2. QuestionTiming
train_data['QuestionTiming'].value_counts()


# In[ ]:


# 3. TimeUtc
train_data["TimeUtc"] = pd.to_datetime(train_data["TimeUtc"])

# Find the earliest date
earliest_date = train_data["TimeUtc"].min()

# Find the most recent date
recent_date = train_data["TimeUtc"].max()

print("Earliest Date:", earliest_date)
print("Most Recent Date:", recent_date)


# In[ ]:


# extract workday and weekend
train_data['Day'] = train_data['TimeUtc'].dt.day_name()
train_data['Day'] = train_data['Day'].map({'Monday': 'Workday', 'Tuesday': 'Workday', 'Wednesday': 'Workday', 'Thursday': 'Workday', 'Friday': 'Workday', 'Saturday': 'Weekend', 'Sunday': 'Weekend'})

# extract the hour of the day
train_data['Hour'] = train_data['TimeUtc'].dt.hour

# plot the distribution of Day
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='Day')
plt.title('Distribution of Day')
plt.show()

# plot the distribution of Hour
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='Hour')
plt.title('Distribution of Hour')
plt.show()


# In[ ]:


# extract workday and weekend
train_data['Day'] = train_data['TimeUtc'].dt.day_name()

# extract the hour of the day
train_data['Hour'] = train_data['TimeUtc'].dt.hour

# change Hour into three categories: night, daytime, and evening
train_data['Hour'] = pd.cut(train_data['Hour'], bins=[-1, 5, 17, 23], labels=['night', 'daytime', 'evening'])

# plot the distribution of Day
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='Day')
plt.title('Distribution of Day')
plt.show()

# plot the distribution of Hour
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='Hour')
plt.title('Distribution of Hour')
plt.show()


# In[ ]:


# 4. CurrentGameMode
# plot the distribution of CurrentGameMode
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='CurrentGameMode')
plt.title('Distribution of CurrentGameMode')
plt.show()


# In[ ]:


# 5. CurrentTask
# plot the distribution of CurrentTask
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='CurrentTask')
plt.title('Distribution of CurrentTask')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Function to categorize CurrentTask
def categorize_current_task(task):
    vehicles = ['CAR', 'MOTORBIKE', 'BICYCLE', 'VINTAGECAR', 'FIRETRUCK', 'SUBMARINE', 'UFO', 'PLANE', 'HELICOPTER', 'TRAIN', 'LOCOMOTIVE',
                 'PWVan', 'DirtBike', 'GolfCart', 'MotorbikeSidecar', '80sRacingCar', 'FireTruck', 'CamperVan', 'PrivateJet', 'MarsRover',
                 'SUV', 'FireHelicopter', 'FortuneTellerCaravan', 'MonsterTruck', 'StuntPlane', 'SteamLocomotive', 'HOME_VAN', 'SUBURBIA_CAMPERVAN',
                 'HOME_DIRTBIKE', 'HOME_GOLFCART', 'SUBURBIA_VINTAGECAR', 'HANGAR_STUNTPLANE', 'HOME_SUV', 'HANGAR_MONSTERTRUCK',
                 'HOME_MOTORBIKESIDECAR', 'MARS_MARSROVER', 'MARINA_FISHINGBOAT', 'DESERT_UFO', 'WATER_HOME_VAN', 'TIME_HOME_MOTORBIKESIDECAR',
                 'WATER_HOME_GOLFCART', 'TIME_HOME_VAN', 'TIME_HOME_DIRTBIKE', 'WATER_FIRESTATION_FIRETRUCK', 'HOME_PENNYFARTHING','FrolicBoat',
                'SUBWAY_PLATFORM', 'SUBWAY_TRAIN', 'AIRPORT_HANGAR', 'AIRPORT_RUNWAY', 'Subway', 'Airport', 'FIRESTATION_FIRETRUCK',
                      'FIRESTATION_FIREHELICOPTER', 'AIRPORT_PRIVATEJET', 'SUBWAY_SUBWAYPLATFORM', 'FIRESTATION_FIRESTATION']

    recreational = ['RECREATIONGROUND_PLAYGROUND', 'RECREATIONGROUND_BACKYARD', 'RECREATIONGROUND_SKATEPARK', 'Playground', 'Swing', 'Slide',
                     'Seesaw', 'Roundabout', 'JungleGym', 'Sandbox', 'Stegoslide', 'ClimbingFrame', 'PlaygroundFloor', 'MerryGoRound', 'SkatePark',
                     'HelterSkelter', 'BigWheel_01', 'FAIRGROUND_MERRYGOROUND', 'RECREATIONGROUND_MINIGOLF', 'FAIRGROUND_HELTERSKELTER',
                     'FAIRGROUND_BIGWHEEL','HOME_DRILL', 'Drill', 'Stadium', 'ShoppingMall', 'Park', 'Subway', 'SubwayPlatform', 'SubwayWashroom',
                    'RECREATIONGROUND_FOUNTAIN',                     'SUBWAY_SUBWAYWASHROOM','NATIONALPARK_CAMPSITE', 'NATIONALPARK_WHEELCHAIRRAMP',
                    'NATIONALPARK_SHELTER', 'NATIONALPARK_PICNICTABLE', 'NATIONALPARK_BENCH',
                   'Restaurant', 'Shop', 'Store', 'Mall', 'NATIONALPARK_TREEHOUSE']

    buildings = ['RESIDENTIALSMALL_BACKYARD', 'RESIDENTIAL_HOME', 'RESIDENTIAL_FRONTYARD', 'RESIDENTIAL_SMALL_HOUSE', 'RESIDENTIAL_TREEHOUSE',
                    'House', 'Bungalow', 'Cottage', 'Mansion', 'ShoeHouse', 'DetachedHouse', 'TreeHouse', 'RESIDENTIALSMALL_BUNGALOW',
                    'SUBURBIA_DETACHEDHOUSE', 'MANSION_FRONT', 'NATIONALPARK_SHOEHOUSE', 'NATIONALPARK_STORYBOOKHOUSE', 'RESIDENTIALSMALL_RACINGCAR']

    special = ['ALIENBASE', 'TREASUREISLAND', 'SPACESTATION', 'MARS_MARSROVER', 'SPACEHATCH', 'AlienHatch', 'TreasureChest', 'BigStatue',
                          'SpaceStation', 'AncientStatue', 'AncientHand', 'COUNTRYSIDE_TEMPLE', 'SEATEMPLE', 'DESERT_STEAMLOCOMOTIVE',
                          'DESERT_ANCIENTSTATUE', 'DESERT_ANCIENTHAND', 'FAIRGROUND_FORTUNETELLERCARAVAN']

    other = ['WASHABLE']

    if task in vehicles:
        return 'Vehicles'
    elif task in recreational:
        return 'Recreational'
    elif task in buildings:
        return 'Residential_Areas'
    elif task in special:
        return 'Special'
    elif task in other:
        return 'Other'
    else:
        return 'Uncategorized'

# Apply the categorization functions to create new columns
train_data['CurrentTaskCategory'] = train_data['CurrentTask'].apply(categorize_current_task)


# In[ ]:


# plot the distribution of CurrentTaskCategory
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='CurrentTaskCategory')
plt.title('Distribution of CurrentTaskCategory')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# 6. CurrentSessionLength
# plot the distribution of CurrentSessionLength
plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x='CurrentSessionLength', bins=10)
plt.title('Distribution of CurrentSessionLength')
plt.show()


# In[ ]:


# 7. LastTaskCompleted
# plot the distribution of LastTaskCompleted
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='LastTaskCompleted')
plt.title('Distribution of LastTaskCompleted')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Function to categorize LastTaskCompleted
def categorize_last_task(task):
    vehicles = ['WASH_VintageCar', 'WASH_Firetruck', 'WASH_Submarine', 'WASH_Boat', 'WASH_UFO', 'WASH_Motorbike', 'WASH_Plane',
                'WASH_Helicopter', 'WASH_Train', 'WASH_Locomotive','WASH_PWVan', 'WASH_DirtBike', 'WASH_GolfCart', 'WASH_MotorbikeSidecar',
                'WASH_80sRacingCar', 'WASH_FireTruck', 'WASH_CamperVan', 'WASH_PrivateJet', 'WASH_MarsRover',
                'WASH_SUV', 'WASH_FireHelicopter', 'WASH_FortuneTellerCaravan', 'WASH_MonsterTruck', 'WASH_StuntPlane',
                'WASH_SteamLocomotive', 'WASH_Marina_FishingBoat', 'WASH_Subway']

    recreational = ['WASH_Playground', 'WASH_Swing', 'WASH_Slide', 'WASH_Seesaw', 'WASH_Roundabout',
                    'WASH_JungleGym', 'WASH_Sandbox', 'WASH_Stegoslide', 'WASH_ClimbingFrame',
                  'WASH_PlaygroundFloor', 'WASH_MerryGoRound', 'WASH_SkatePark', 'WASH_HelterSkelter',
                    'WASH_BigWheel_01','WASH_Park', 'WASH_Garden', 'WASH_Backyard', 'WASH_Porch',
                    'WASH_TreeHouse', 'WASH_Temple', 'WASH_SeaTemple', 'WASH_Fountain']

    buildings = ['WASH_House', 'WASH_Bungalow', 'WASH_Cottage', 'WASH_Restaurant', 'WASH_Shop',
                 'WASH_Store', 'WASH_Mall', 'WASH_Mansion', 'WASH_ShoeHouse', 'WASH_DetachedHouse',
                 'WASH_FireStation', 'WASH_StoryBookCottage', 'WASH_Airport', 'WASH_Stadium', 'WASH_Park',
                 'WASH_School', 'WASH_SubwayWashroom', 'WASH_SubwayPlatform']

    furniture = ['WASH_Chair', 'WASH_Table', 'WASH_Bench', 'WASH_Sofa', 'WASH_Bench_01', 'WASH_Bench_02', 'WASH_Bin_01', 'WASH_Bin']

    special = ['WASH_AlienHatch', 'WASH_TreasureChest', 'WASH_BigStatue', 'WASH_SpaceStation', 'WASH_AncientStatue',
               'WASH_Drill', 'WASH_FrolicBoat', 'WASH_AncientHand']

    other = ['WASHABLE']

    if task in vehicles:
        return 'Vehicles'
    elif task in recreational:
        return 'Playground'
    elif task in buildings:
        return 'Buildings'
    elif task in furniture:
        return 'Furniture'
    elif task in special:
        return 'Special'
    elif task in other:
        return 'Other'
    else:
        return 'Uncategorized'
    
train_data['LastTaskCategory'] = train_data['LastTaskCompleted'].apply(categorize_last_task)


# In[ ]:


# plot the distribution of LastTaskCategory
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='LastTaskCategory')
plt.title('Distribution of LastTaskCategory')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# 8. LevelProgressionAmount
# plot the distribution of LevelProgressionAmount
plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x='LevelProgressionAmount', bins=10)
plt.title('Distribution of LevelProgressionAmount')
plt.show()


# In[ ]:


# 9. QuestionType
# plot the distribution of QuestionType
plt.figure(figsize=(10, 6))
sns.countplot(data=train_data, x='QuestionType')
plt.title('Distribution of QuestionType')
plt.show()


# In[ ]:


# 10. ResponseValue
# plot the distribution of ResponseValue
plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x='ResponseValue', bins=10)
plt.title('Distribution of ResponseValue')
plt.show()


# In[ ]:


train_data.info()


# In[ ]:


# Load the datasets again
def import_data_from_csv():
    df_train = pd.read_csv('train_data.csv')
    df_test = pd.read_csv('test_data.csv')
    return df_train, df_test
train_data, test_data = import_data_from_csv()


# In[ ]:


# if fill missing values by deleting the rows with missing values
train_data_1 = train_data.copy()
train_data_1.dropna(inplace=True)
train_data_1.info()


# In[ ]:


# if delete LastTaskCompleted and then delete the rows with missing values
train_data_2 = train_data.copy()
train_data_2.drop(columns=['LastTaskCompleted'], inplace=True)
train_data_2.dropna(inplace=True)
train_data_2.info()


# In[ ]:


min_value = train_data_2['LevelProgressionAmount'].min()
max_value = train_data_2['LevelProgressionAmount'].max()
print(min_value)
print(max_value)


# In[ ]:


# Determine the percentile-based bin edges dynamically
bin_edges = [0] + train_data_2['LevelProgressionAmount'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).tolist() + [1.001]

# Generate labels for each bin
labels = [f"from {bin_edges[i]} to {bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

# Create a new column with the binned values
train_data_2['BinnedLevelProgression'] = pd.cut(train_data_2['LevelProgressionAmount'], bins=bin_edges, labels=labels, right=False)

# Display the initial result
print("Initial bin counts:")
print(train_data_2['BinnedLevelProgression'].value_counts())
print("Number of missing values:", train_data_2['BinnedLevelProgression'].isnull().sum())

# Drop two instances from the bin 'from 0.9838766360000001 to 1.001'
indices_to_drop_bin1 = train_data_2[train_data_2['BinnedLevelProgression'] == 'from 0.9838766360000001 to 1.001'].index[:2]
train_data_2 = train_data_2.drop(indices_to_drop_bin1)

# Drop three instances from the bin 'from 1.0 to 1.001'
indices_to_drop_bin2 = train_data_2[train_data_2['BinnedLevelProgression'] == 'from 1.0 to 1.001'].index[:3]
train_data_2 = train_data_2.drop(indices_to_drop_bin2)

# Recreate the binned column
train_data_2['BinnedLevelProgression'] = pd.cut(train_data_2['LevelProgressionAmount'], bins=bin_edges, labels=labels, right=False)

# Display the adjusted result and sort by bin edges to maintain order
print("Adjusted bin counts:")
print(train_data_2['BinnedLevelProgression'].value_counts().reindex(labels))
print("Number of missing values:", train_data_2['BinnedLevelProgression'].isnull().sum())


# In[ ]:


# Determine the percentile-based bin edges dynamically (because otherwise we will loose values due to floating points. Like this we're also making sure the distribution is normal)
bin_edges = [0] + train_data_2['LevelProgressionAmount'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).tolist() + [1.001]

# Generate labels for each bin
labels = [f"from {bin_edges[i]} to {bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

# Create a new column with the binned values
train_data_2['BinnedLevelProgression'] = pd.cut(train_data_2['LevelProgressionAmount'], bins=bin_edges, labels=labels, right=False)

# Display the result
print(train_data_2['BinnedLevelProgression'].value_counts())
print("Number of missing values:", train_data_2['BinnedLevelProgression'].isnull().sum())


# In[ ]:


# Define the bin edges manually to ensure correct binning, including separate bins for 0.9838 to 1.0 and 1.0 to 1.001
bin_edges = [0, 0.104285634, 0.213396196, 0.32048484899999996, 0.43458137200000005, 0.55007635, 0.662722016, 0.771927278, 0.87988618, 0.9838, 0.999999, 1.001]

# Generate labels for each bin
labels = [f"from {bin_edges[i]} to {bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

# Apply binning to the LevelProgressionAmount column in train_data
train_data_2['BinnedLevelProgression'] = pd.cut(train_data_2['LevelProgressionAmount'], bins=bin_edges, labels=labels, right=False, include_lowest=True)
# Display the result
print(train_data_2['BinnedLevelProgression'].value_counts())
print("Number of missing values in train_data:", train_data_2['BinnedLevelProgression'].isnull().sum())


# In[ ]:


# Define the custom mapping dictionary (this is because otherwise the bin "from 0.9838766360000001 to 1.001" would be labeled as 1)
custom_mapping = {
    "from 0 to 0.104285634": 0,
    "from 0.104285634 to 0.213396196": 1,
    "from 0.213396196 to 0.32048484899999996": 2,
    "from 0.32048484899999996 to 0.43458173200000005": 3,
    "from 0.43458173200000005 to 0.55007635": 4,
    "from 0.55007635 to 0.662272016": 5,
    "from 0.662272016 to 0.771927278": 6,
    "from 0.771927278 to 0.87988618": 7,
    "from 0.104285634 to 0.213396196": 8,
    "from 0.9838766360000001 to 1.001": 9
}

# Initialize LabelEncoder with custom mapping
label_encoder = LabelEncoder()
label_encoder.mapping = [{key: value} for key, value in custom_mapping.items()]

# Fit and transform the column
train_data_2['BinnedLevelProgression_encoded'] = label_encoder.fit_transform(train_data_2['BinnedLevelProgression'])

# Display the result
print(train_data_2['BinnedLevelProgression_encoded'].value_counts())


# In[ ]:


# Create a new column with the binned values
train_data_2['BinnedLevelProgression'] = pd.cut(train_data_2['LevelProgressionAmount'], bins=11, labels=False, right=False)

# Display the result
print(train_data_2['BinnedLevelProgression'].value_counts())


# In[ ]:


# if fill missing values with mode for categorical features and mean for numerical features
train_data_3 = train_data.copy()
train_data_3['CurrentGameMode'].fillna(train_data_3['CurrentGameMode'].mode()[0], inplace=True)
train_data_3['CurrentTask'].fillna(train_data_3['CurrentTask'].mode()[0], inplace=True)
train_data_3['LastTaskCompleted'].fillna(train_data_3['LastTaskCompleted'].mode()[0], inplace=True)
train_data_3['LevelProgressionAmount'].fillna(train_data_3['LevelProgressionAmount'].mean(), inplace=True)
train_data_3.info()
# check each feature
print(train_data_3['CurrentGameMode'].value_counts())
print(train_data_3['CurrentTask'].value_counts())
print(train_data_3['LastTaskCompleted'].value_counts())
print(train_data_3['LevelProgressionAmount'].describe())


# In[ ]:


# if fill missing values according to the meaning of the feature
train_data_4 = train_data.copy()
train_data_4['LastTaskCompleted'].fillna('Not_Saved', inplace=True)
train_data_4['CurrentTask'].fillna('Not_Playing', inplace=True)
train_data_4['LevelProgressionAmount'].fillna(0, inplace=True)

train_data_4['LastTaskCompleted'].fillna('Not_Saved', inplace=True)
train_data_4['CurrentTask'].fillna('Not_Playing', inplace=True)
train_data_4['LevelProgressionAmount'].fillna(0, inplace=True)


# In[ ]:


# Remove outliers from training data
def remove_outliers(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
train_data_4 = remove_outliers(train_data_4)


# ### EDA & feature engineering

# In[ ]:


# fill missing values
train_data['LastTaskCompleted'].fillna('Not_Saved', inplace=True)
train_data['CurrentTask'].fillna('Not_Playing', inplace=True)
train_data['LevelProgressionAmount'].fillna(0, inplace=True)


# In[ ]:


# Remove outliers from training data
def remove_outliers(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# In[ ]:


# Remove outliers from the train_data
train_data_no_outliers = remove_outliers(train_data)

# Visualize the before and after for all numerical columns in train_data
numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    plt.figure(figsize=(14, 6))

    # Before removing outliers
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_data, x=col)
    plt.title(f'Boxplot of {col} (Before)')

    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=col, bins=30, kde=True)
    plt.title(f'Distribution of {col} (Before)')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))

     # After removing outliers
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_data_no_outliers, x=col)
    plt.title(f'Boxplot of {col} (After)')

    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data_no_outliers, x=col, bins=30, kde=True)
    plt.title(f'Distribution of {col} (After)')

    plt.tight_layout()
    plt.show()


# In[ ]:


train_data = remove_outliers(train_data)


# In[ ]:


# 'TimeUtc': According to feature "TimeUtc", we extract the date and time information and create two new features "Day" and "Time"
# "Day" should be a categorical feature and include monday, tuesday, wednesday, thursday, friday, saturday, and sunday, and then it gets encoded
# "Time" should be a categorical feature and include night, daytime, and evening, and then it gets encoded
train_data['TimeUtc'] = pd.to_datetime(train_data['TimeUtc'])
train_data['Day'] = train_data['TimeUtc'].dt.day_name()
# train_data_1['Day'] = pd.Categorical(train_data_1['Day'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
train_data['Time'] = train_data['TimeUtc'].dt.hour
train_data['Time'] = pd.cut(train_data['Time'], bins=[-1, 5, 17, 23], labels=['night', 'daytime', 'evening'])
# encode 'Day': Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
train_data['Day'] = train_data['Day'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})
train_data['Time'] = train_data['Time'].map({'night': 0, 'daytime': 1, 'evening': 2})
# change 'Day' and 'Time' to int
train_data['Day'] = train_data['Day'].astype(int)
train_data['Time'] = train_data['Time'].astype(int)


# In[ ]:


train_data.drop(['QuestionTiming', 'CurrentGameMode'], axis=1, inplace=True)


# In[ ]:


# Standardize the numerical feature
scaler = StandardScaler()
train_data[['Day', 'Time', 'CurrentSessionLength']] = scaler.fit_transform(train_data[['Day', 'Time', 'CurrentSessionLength']])
train_data['StandardizedProgression'] = scaler.fit_transform(train_data[['LevelProgressionAmount']])


# In[ ]:


# Group by 'UserID' and calculate statistical features for 'ResponseValue'
user_stats = train_data.groupby('UserID')['ResponseValue'].agg(['mean', 'median', 'std']).reset_index()
user_stats.columns = ['UserID', 'ResponseValue_mean', 'ResponseValue_median', 'ResponseValue_std']


# In[ ]:


# Merge the new features back into the original dataframe
train_data = pd.merge(train_data, user_stats, on='UserID', how='left')


# In[ ]:


# Impute NaN values in ResponseValue_std with 0
train_data['ResponseValue_std'].fillna(0, inplace=True)


# In[ ]:


train_data['UserID'] = train_data['UserID'].str.replace('p', '').astype(int)


# In[ ]:


# Apply categorization to LastTaskCompleted and CurrentTask
special_tasks = [
    'MarsRover', 'MARS_MARSROVER', 'WASH_MarsRover', 'RECREATIONGROUND_MINIGOLF', 'SteamLocomotive',
    'WASH_SteamLocomotive', 'DESERT_STEAMLOCOMOTIVE', 'WASH_Fountain','RECREATIONGROUND_FOUNTAIN'
]


# In[ ]:


def categorize_task(task):

    if task == 'Not_Saved' or task == 'Not_Playing':
        return 0
    elif task in special_tasks:
        return 2
    else:
        return 1


# In[ ]:


train_data['LastTaskCompleted_Category'] = train_data['LastTaskCompleted'].apply(categorize_task)
train_data['CurrentTask_Category'] = train_data['CurrentTask'].apply(categorize_task)

# Verify the results
print(train_data[['LastTaskCompleted', 'CurrentTask', 'LastTaskCompleted_Category', 'CurrentTask_Category']].head())


# In[ ]:


# Group by 'UserID' and calculate statistical features for 'CurrentSessionLength' and 'LevelProgressionAmount'
CSLstats = train_data.groupby('UserID')['CurrentSessionLength'].agg(['mean', 'median', 'std', 'max']).reset_index()
CSLstats.columns = ['UserID', 'CSL_mean', 'CSL_median', 'CSL_std', 'CSL_max']
LPAstats = train_data.groupby('UserID')['LevelProgressionAmount'].agg(['mean', 'median', 'std', 'max']).reset_index()
LPAstats.columns = ['UserID', 'LPA_mean', 'LPA_median', 'LPA_std', 'LPA_max']


# In[ ]:


# Merge the new features back into the original dataframe
train_data = pd.merge(train_data, CSLstats, on='UserID', how='left')
train_data = pd.merge(train_data, LPAstats, on='UserID', how='left')


# In[ ]:


# Impute NaN values in ResponseValue_std with 0
train_data['CSL_std'].fillna(0, inplace=True)

# Impute NaN values in ResponseValue_std with 0
train_data['LPA_std'].fillna(0, inplace=True)


# In[ ]:


columns_to_drop = ['QuestionTiming', 'TimeUtc', 'CurrentGameMode', 'CurrentTask', 'LastTaskCompleted', 'QuestionType', 'LevelProgressionAmount',
                   'LastTaskCategory','CurrentTask_Category', 'LastTaskCompleted_Category']
train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')


# In[ ]:


# Ensure all features are numeric and handle any non-numeric data
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')


# In[ ]:


# Filter out ResponseValue with only one instance
value_counts = train_data['ResponseValue'].value_counts()
filtered_train_data = train_data[train_data['ResponseValue'].isin(value_counts[value_counts > 1].index)]


# In[ ]:


# split trainning data into features and target
X_2 = train_data.drop(columns='ResponseValue')
y_2 = train_data['ResponseValue']

# split the data into training and validation sets
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)


# In[ ]:


X_2.head()


# In[ ]:


# train a random forest model
rfr = RandomForestRegressor()
rfr.fit(X_train_2, y_train_2)
y_pred = rfr.predict(X_val_2)

# Apply Permutation Feature Importance
perm_importance = permutation_importance(rfr, X_val_2, y_val_2, n_repeats=10, random_state=42)

# Print the results
print("Feature importances:")
for i in perm_importance.importances_mean.argsort()[::-1]:
    print(f"{X_2.columns[i]:<30} {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")


# ### modeling & hyperparameter tuning

# In[ ]:


# split X_train and y_train into training and validation sets
X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


# 1. Dummy Regressor
dummy_regressor = DummyRegressor(strategy='mean')
dummy_regressor.fit(X_train_1, y_train_1)
y_pred = dummy_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the dummy model: {mae}')


# In[ ]:


# 2. Logistic Regression
logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_1, y_train_1)
y_pred = logistic_regression.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the logistic regression model: {mae}')


# In[ ]:


# 3. Random Forest Regressor
random_forest_regressor = RandomForestRegressor(random_state=42)
random_forest_regressor.fit(X_train_1, y_train_1)
y_pred = random_forest_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the random forest model: {mae}')


# In[ ]:


# hyperparameter tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Define the scoring metric
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=skf, scoring=scorer, n_jobs=-1, verbose=2)

# Fit Grid Search
grid_search.fit(X_train_1, y_train_1)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Best MAE: {best_score}')

# Train the model with best parameters on the full training set
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_rf.predict(X_valid_1)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_valid_1, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid_1, y_pred))

print(f'MAE of the best RandomForest model: {mae}')
print(f'RMSE of the best RandomForest model: {rmse}')


# In[ ]:


# 4. SVR
svr = SVR()
svr.fit(X_train_1, y_train_1)
y_pred = svr.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the SVR model: {mae}')


# In[ ]:


# 5. LGBM Regressor
lgbm_regressor = LGBMRegressor(random_state=42)
lgbm_regressor.fit(X_train_1, y_train_1)
y_pred = lgbm_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the LGBM model: {mae}')


# In[ ]:


# hyperparameter tuning
# Define the parameter grid
param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500]
}

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
lgbm = LGBMRegressor(random_state=42)

# Define the scoring metric
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=skf, scoring=scorer, n_jobs=-1)

# Fit Grid Search
grid_search.fit(X_train_1, y_train_1)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Best MAE: {best_score}')

# Train the model with best parameters on the full training set
best_lgbm = LGBMRegressor(**best_params, random_state=42)
best_lgbm.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_lgbm.predict(X_valid_1)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_valid_1, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid_1, y_pred))

print(f'MAE of the best LightGBM model: {mae}')
print(f'RMSE of the best LightGBM model: {rmse}')


# In[ ]:


# 6. XGB Regressor
xgb_regressor = XGBRegressor(random_state=42)
xgb_regressor.fit(X_train_1, y_train_1)
y_pred = xgb_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the XGB model: {mae}')


# In[ ]:


# hyperparameter tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
xgb_model = XGBRegressor(random_state=42)

# Define the scoring metric
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=skf, scoring=scorer, n_jobs=-1)

# Fit Grid Search
grid_search.fit(X_train_1, y_train_1)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Best MAE: {best_score}')


# In[ ]:


# 7. mlp Regressor
mlp_regressor = MLPRegressor(random_state=42)
mlp_regressor.fit(X_train_1, y_train_1)
y_pred = mlp_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the MLP model: {mae}')


# In[ ]:


# 8. Gradient Boosting Regressor
gradient_boosting_regressor = GradientBoostingRegressor(random_state=42)
gradient_boosting_regressor.fit(X_train_1, y_train_1)
y_pred = gradient_boosting_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the Gradient Boosting model: {mae}')


# In[ ]:


# standardize the numerical features
X_train_1 = scaler.fit_transform(X_train_1)
X_valid_1 = scaler.transform(X_valid_1)


# In[ ]:


# 9. SGD Regressor
sgd_regressor = SGDRegressor(random_state=42)
sgd_regressor.fit(X_train_1, y_train_1)
y_pred = sgd_regressor.predict(X_valid_1)
mae = mean_absolute_error(y_valid_1, y_pred)
print(f'MAE of the SGD model: {mae}')

