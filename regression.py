#library
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
import Werkzeung
import itdangerous
import matplotlib
import pickle
#dataset 
X= np.array([68.95,80.23,69.47,74.15,68.37,59.99,88.91,66,74.53,69.88,47.64,83.07,69.57,79.52,42.95,63.45,55.39,82.03,54.7,74.58,77.22,84.59,41.49,87.29,41.39,78.74,48.53,51.95,70.2,76.02,67.64,86.41,59.05,55.6,57.64,84.37,62.26,65.82,50.43,38.93,84.98,64.24,82.52,81.38,80.47,37.68,69.62,85.4,44.33,48.01]).reshape((-1, 1))

Y= np.array([61833.9,68441.85,59785.94,54806.18,73889.99,59761.56,53852.85,24593.33,68862,55642.32,45632.51,62491.01,51636.92,51739.63,30976,52182.23,23936.86,71511.08,31087.54,23821.72,64802.33,60015.57,32635.7,61628.72,68962.32,64828,38067.08,58295.82,32708.94,46179.97,51473.28,45593.93,25583.29,30227.98,45580.92,61389.5,56770.79,76435.3,57425.87,27508.41,57691.95,59784.18,66572.39,64929.61,57519.64,53575.48,50983.75,67058.72,52723.34,54286.1])


#call model regression
model = LinearRegression().fit(X,Y)
#save model
filename = 'model.sav'
joblib.dump(model, filename)
#load model
loaded_model = joblib.load(filename)
#prediction model
loaded_model.predict(10)
