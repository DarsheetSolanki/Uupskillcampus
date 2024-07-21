# Smart City Traffic Patterns:

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime
import time
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
test = pd.read_csv('test_data.csv')
train = pd.read_csv('train_data.csv')
head = train.head(5)
tail = train.tail(5)

def data_inf(data,name):
    print('rows: ',data.shape[0],' ,columns: ',data.shape[1],' in',name,'\n')
    data.info()
    print('\n')
data_inf(train,'Train')
# display(train.head(5).append(train.tail(5)))
concatenated_df = pd.concat([head, tail])
print(concatenated_df)
head1 = test.head(5)
tail1 = test.tail(5)
data_inf(test,"Test")
concatenated_df = pd.concat([head, tail])
print(concatenated_df)
display(train.describe())
display(train.describe(include = 'object'))
print('Before Dropping ',train.shape[0])
train.drop_duplicates(keep="first", inplace=True) 
print('After Dropping ',train.shape[0])

if(train.isnull().sum().sum()==0):
    print('no missing values in train')
else:
    train.fillna(method='ffill',inplace=True)
if(test.isnull().sum().sum()==0):
    print('no missing values in test')    
else:
    test.fillna(method='ffill',inplace=True)
print('Before Converting :',train['DateTime'].dtype)
train['DateTime'] = pd.to_datetime(train['DateTime'],errors='coerce')
test['DateTime'] = pd.to_datetime(test['DateTime'])
print('After Converting :',train['DateTime'].dtype)
train.info()
# Exploring more features  
train["Year"]= train['DateTime'].dt.year  
train["Month"]= train['DateTime'].dt.month  
train["Date_no"]= train['DateTime'].dt.day  
train["Hour"]= train['DateTime'].dt.hour  
train["Day"]= train.DateTime.dt.strftime("%A")

test["Year"]= test['DateTime'].dt.year  
test["Month"]= test['DateTime'].dt.month  
test["Date_no"]= test['DateTime'].dt.day  
test["Hour"]= test['DateTime'].dt.hour  
test["Day"]= test.DateTime.dt.strftime("%A")
train.head()
test.head()
# time series plot
colors = [ "pink","blue","lightgreen","brown"]
plt.figure(figsize=(20,4),facecolor="violet")  
time_series=sns.lineplot(x=train['DateTime'],y="Vehicles",data=train, hue="Junction", palette=colors)  
time_series.set_title("DateTime vs Vehicle")  
time_series.set_ylabel("Vehicles in Number")  
time_series.set_xlabel("DateTime") 
#years of traffic at junction
plt.figure(figsize=(12,5),facecolor="violet")  
colors = [ "pink","blue","lightgreen","brown"]
count = sns.countplot(data=train, x =train["Year"], hue="Junction", palette=colors)  
count.set_title("Years of Traffic at Junctions")  
count.set_ylabel("Vehicles in numbers")  
count.set_xlabel("Date") 
#heat map
numeric_train = train.select_dtypes(include=['number'])
corrmat = numeric_train.corr() 
plt.subplots(figsize=(10,10),facecolor="violet")  
sns.heatmap(corrmat,cmap= "Pastel2",annot=True,square=True )
plt.show()
def datetounix1(df):
    # Initialising unixtime list
    unixtime = []
    
    # Running a loop for converting Date to seconds
    for date in df['DateTime']:
        unixtime.append(time.mktime(date.timetuple()))
    
    # Replacing Date with unixtime list
    df['DateTime'] = unixtime
    return(df)
train.head()
# Define a function to convert datetime objects to Unix timestamps
def datetounix1(df):
    # Convert 'DateTime' column to datetime type if it's not already
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    # Drop rows with NaT (missing) values in 'DateTime' column
    df = df.dropna(subset=['DateTime'])
    
    unixtime = []
    # Running a loop for converting Date to seconds
    for date in df['DateTime']:
        unixtime.append(time.mktime(date.timetuple()))
    
    # Replacing 'DateTime' with Unix timestamps
    df['DateTime'] = unixtime
    return df

# Assuming you have DataFrame 'train' and 'test'
train_features = datetounix1(train.drop(['Vehicles'], axis=1))
test_features = datetounix1(test)

# Store Features / Predictors in array :
X = train_features  
X_valid = test_features

# One Hot Encoding - Using Dummies :
X = pd.get_dummies(X)
X_valid = pd.get_dummies(X_valid)

# Store target 'Vehicles' in y array :
y = train['Vehicles'].to_frame()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=512)
# Convert the dataset to LightGBM data format
train_data = lgb.Dataset(X_train, label=y_train)

# Set the parameters for the LightGBM regression model
params = {
    'objective': 'regression',
    'metric': 'rmse'  # Root Mean Squared Error
}

# Train the LightGBM regression model
model = lgb.train(params,train_data, num_boost_round=100)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the evaluation metrics
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
# Create a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)
#for i in range(15880):
#    print(y_pred[i],y_test.iloc[i])
# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the evaluation metrics
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)