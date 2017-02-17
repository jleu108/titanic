
# Pandas dataframe and sea data visualization on Titanic data

# Imports
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("/Users/justinleu/Documents/PythonScripts/Titanic/data/train.csv")
test_df    = pd.read_csv("/Users/justinleu/Documents/PythonScripts/Titanic/data/test.csv")


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=test_df, ax=axis1)
sns.countplot(x='Age', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)
sns.countplot(x='Embarked', hue="Sex", data=titanic_df, ax=axis3)


# group by embarked, and get the mean for survived passengers for each value in Embarked
embarked_sum = test_df[['Embarked', 'Age']].groupby(['Embarked']).sum()
Passenger_Fare_sum = test_df[['PassengerId', 'Fare']].groupby(['PassengerId']).sum()
embarked_gp
Passenger_Fare_sum

# Get Fare for Embarked Passenger Types Q and S
fare_Q = test_df['Fare'][test_df['Embarked'] == 'Q']
fare_S = test_df['Fare'][test_df['Embarked'] == 'S']

# Get Average and Std Fare fo Embarked Passenger Types Q and S
average_fare = DataFrame([fare_Q.mean(), fare_S.mean()])
std_fare = DataFrame([fare_Q.std(), fare_S.std()])


# In[72]:

# Histogram Plot
test_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,80))

# Box Plot
average_fare.index.names = std_fare.index.names = ["Embarked"]
average_fare.plot(yerr=std_fare,kind='bar',legend=False)



# Make pandas dataframe a dic
fares_dict = Passenger_Fare_sum.to_dict(orient='dict')
fares_dict


