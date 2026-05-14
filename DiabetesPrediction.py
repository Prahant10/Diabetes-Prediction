import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm #Support vector machine model
from sklearn.metrics import accuracy_score

#loading dataset
diabetes_data = pd.read_csv('diabetes.csv')

#preprocessing
#separating data and labels
x = diabetes_data.drop(columns = 'Outcome', axis = 1)
y = diabetes_data['Outcome']

#Data Standardization
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data

# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y, random_state = 2)
#training the model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train, y_train)
#model prediction
y_pred = classifier.predict(x_test)
#accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# prediction from input data
input_data = (5,166,72,19,175,25.8,0.587,51)

# converting input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshaping
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
    print('The person is not diabetic')
else:   
     print('The person is diabetic')




