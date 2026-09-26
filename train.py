import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Load the sonar dataset
sonar_data=pd.read_csv('Copy of sonar data.csv', header=None)

#Split the dataset into features and labels
X=sonar_data.drop(60, axis=1) #Features (input)- delete the last column(which is index 60-the label column)--column 0-59: features
Y=sonar_data.iloc[:, 60] #Labels (output)- take the last column(which is index 60-the label column)--column 60: labels

#Label encoding: Convert categorical labels (M,R) into numerical values
encoder=LabelEncoder()
Y=encoder.fit_transform(Y) #M:1, R:0

#Split the dataset into training set and test set
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=42) #80% training data and 20% test data

#Standardize the features (input) to have mean=0 and variance=1
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Train the logistic regression model
model=LogisticRegression()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)

#Evaluate the model
accuracy=accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(Y_test, Y_pred))

#Predicting a new sample
input_data= (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
input_data_as_numpy_array=np.asarray(input_data) #Convert the input data to a numpy array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1) #Reshape the array as we are predicting for one instance
input_data_standardized=scaler.transform(input_data_reshaped) #Standardize the input data

#Make prediction
prediction=model.predict(input_data_standardized) #Output will be 0 or 1 (0: Rock, 1: Mine)
prediction_label = encoder.inverse_transform(prediction) #Convert 0/1 back to original label as mine or rock
print("Prediction label:", prediction_label)

#Interpret the prediction
if prediction_label[0]=='M':
    print("The object is a mine")
else:
    print("The object is a rock")

joblib.dump(
    {
        'model': model,
        'scaler': scaler,
        'encoder': encoder
    },
    'sonar_model.pkl'
)