# import the necessary packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 

data = pd.read_csv("creditcard.csv") #use name of the csv file and have the csv file in same directory as this file.

data.head() # retrives some few data rows from the start of file

# Use this line to get random data:
# data = data.sample(frac = 0.1, random_state = 48)
# This line gets 10 random rows and uses them as sample data. (basically head() but retrives random rows).
 
print(data.shape) #get the shape of data
print(data.describe()) 

# Get the number of fraud cases in dataset 

# Our dataset has a row named class. which shows whether the card is fraud or not. 
# Class:  1 means data is fraud. Class: 0 means data is valid.
fraud = data[data['Class'] == 1]  
valid = data[data['Class'] == 0] 
# Below line will calculate the ratio of the number of fraud transactions to the number of valid transactions. (outlier means minority.. here fraud transactions).
outlierFraction = len(fraud)/float(len(valid)) 
print(outlierFraction) 
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 

print("Amount details of the fraudulent transaction") 
fraud.Amount.describe() 

print("details of valid transaction") 
valid.Amount.describe() 

# Correlation matrix 
corrmat = data.corr() # This function computes the correlationn of columns in our dataset. 
# (Basically sees which data are related to each other to learn about fraud data patterns).
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) # Plots the data in a HeatMap format.
plt.show() 

# dividing the X and the Y from the dataset 
# This X and Y will be used as input in our ML Model.
X = data.drop(['Class'], axis = 1) # This removes the 'Class' column. every other column will be used as X axis. (Excludes the Y axis data. as they use the same dataset.)
Y = data["Class"] # This only uses the 'Class' column as it's data.
print(X.shape) 
print(Y.shape) 
# getting just the values for the sake of processing 
# (its a numpy array with no columns) 
xData = X.values 
yData = Y.values 

# Using Scikit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = train_test_split( 
		xData, yData, test_size = 0.2, random_state = 42) 

# Building the Random Forest Classifier (RANDOM FOREST) 
from sklearn.ensemble import RandomForestClassifier 
# random forest model creation 
rfc = RandomForestClassifier() 
rfc.fit(xTrain, yTrain) 
# predictions 
yPred = rfc.predict(xTest) 

# Evaluating the classifier 
# printing every score of the classifier 
# scoring in anything 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 

n_outliers = len(fraud) # Holds the Number of Fraud Transactions.
n_errors = (yPred != yTest).sum() # Holds the Number of errors(data where Prediction didnt match Test data)
print("The model used is Random Forest classifier") 

acc = accuracy_score(yTest, yPred) 
print("The accuracy is {}".format(acc)) 

prec = precision_score(yTest, yPred) 
print("The precision is {}".format(prec)) 

rec = recall_score(yTest, yPred) 
print("The recall is {}".format(rec)) 

f1 = f1_score(yTest, yPred) 
print("The F1-Score is {}".format(f1)) 

MCC = matthews_corrcoef(yTest, yPred) 
print("The Matthews correlation coefficient is{}".format(MCC)) 

 # printing the confusion matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(yTest, yPred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, 
			yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 

