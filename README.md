# BreastCancer

#import libraries
import pandas as pd
import seaborn as sns


#Download dataset from Kaggle
#set kaggle API credentials
import os
os.environ['KAGGLE_USERNAME']='souhibkhalbous'
os.environ['KAGGLE_KEY'] ='2dd86d28af6214d746fd5da037724c12'


#download dataset
! kaggle datasets download  -d  uciml/breast-cancer-wisconsin-data
#unzip file
! unzip /content/breast-cancer-wisconsin-data.zip

#Load & Explore Data
#load data on dataframe
df = pd.read_csv('/content/data.csv')
#display dataframe
df.head()
#count of rows and columns
df.shape
#count number of null(empty) values
df.isna().sum()

# Drop the column with null values
df.dropna(axis=1, inplace = True)

# count of rows and columns
df.shape
#Get count of number of M or B cells in diagnosis
df['diagnosis'].value_counts()

#Label Encoding
#Get Datatypes of each column in our dataset
df.dtypes

#Encode the diagnosis values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.iloc[:,1] =labelencoder.fit_transform(df.iloc[:,1].values)

#display df
df


#Split Dataset & Feature Scaling
#Splitting the dataset into independent and dependent datasets 
X= df.iloc[:, 2:].values
Y = df.iloc[:, 1].values


#Splitting datasets into training(75%) and testing(25%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)

#Scaling the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trsin = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#print data
X_trsin

#Build a Logistic Regression Model
#build a logistic regression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

#make use of trained model to make predictions on test data
predictions= classifier.predict(X_test)


#plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(Y_test, predictions)
print(cm)
sns.heatmap(cm,annot=True)

#get accuracy score for model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))


#compare actual values and predicted values
print(predictions)
print(Y_test)
