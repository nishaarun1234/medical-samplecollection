# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:37:36 2022

@author: Shree
"""

## Home Sample Collection Streamline

import pandas as pd
import numpy as np
import seaborn as sns

samples = pd.read_excel("C:/Users/Shree/Desktop/projectnew/sampla_data_08_05_2022(final).xlsx")
samples.dtypes

samples.shape

# summarize the data to see the distribution of data
print(samples.describe())

# Identify duplicates records in the data
duplicate = samples.duplicated()
duplicate
sum(duplicate) # no duplicates appear

# check for count of null values  each column
samples.isna().sum() # no null values

samples.dtypes
## rename column name

samples = samples.rename(columns={'Cut-off Schedule':'Cut_off_Schedule'})
samples = samples.rename(columns={'Cut-off time_HH_MM':'Cut_off_time_HH_MM'})

#drop Patient_ID ,Agent_ID, Patient_Age, Patient_Gender, Mode_Of_Transport columns beacause is not giving any relative information for our analysis
samples.drop(['Patient_ID'], axis=1, inplace=True)
samples.drop(['Agent_ID'], axis=1, inplace=True)
samples.drop(['Patient_Age'], axis=1, inplace=True)
samples.drop(['Patient_Gender'], axis=1, inplace=True)
samples.drop(['Mode_Of_Transport'], axis=1, inplace=True)
samples.drop(['Test_Booking_Date'], axis=1, inplace=True)
samples.drop(['Sample_Collection_Date'], axis=1, inplace=True)

samples.columns


# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# label encode 
samples['Test_Name']= labelencoder.fit_transform(samples['Test_Name'])
samples['Sample'] = labelencoder.fit_transform(samples['Sample'])
samples['Way_Of_Storage_Of_Sample'] = labelencoder.fit_transform(samples['Way_Of_Storage_Of_Sample'])
samples['Cut_off_Schedule']= labelencoder.fit_transform(samples['Cut_off_Schedule'])
samples['Traffic_Conditions']= labelencoder.fit_transform(samples['Traffic_Conditions'])
samples['Reached_On_Time']= labelencoder.fit_transform(samples['Reached_On_Time'])

samples.dtypes

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
samples.mean()

# Measures of Dispersion / Second moment business decision 
samples.var()
samples.std()

# Third moment business decision 
samples.skew()

# Fourth moment business decision
samples.kurt()

import matplotlib.pyplot as plt
import numpy as np

samples.shape
samples.columns
#histogram

plt.hist(samples.Test_Booking_Date) # data is not normally distributed
plt.hist(samples.Test_Booking_Time_HH_MM) # data is not normally distributed
plt.hist(samples.Sample_Collection_Date) # data is not normally distributed
plt.hist(samples.Scheduled_Sample_Collection_Time_HH_MM) # data is  normally distributed
plt.hist(samples.Cut_off_time_HH_MM) # data is not normally distributed
plt.hist(samples.Agent_Location_KM) # data is not normally distributed
plt.hist(samples.Time_Taken_To_Reach_Patient_MM) # data is not normally distributed
plt.hist(samples.Time_For_Sample_Collection_MM) # data is not normally distributed
plt.hist(samples.Lab_Location_KM) # data is not normally distributed
plt.hist(samples.Time_Taken_To_Reach_Lab_MM) # data is not normally distributed

#Box plot
a = sns.boxplot(data = samples.iloc[:,[0,1,2,3,4,5]])
a = sns.boxplot(data = samples.iloc[:,[6,7,8,9,10]])
a = sns.boxplot(data = samples.iloc[:,[11,12,13]])
samples.columns

# scatter plot
plt.scatter(samples.Test_Booking_Time_HH_MM, samples.Test_Booking_Time_HH_MM)
plt.scatter(samples.Cut_off_time_HH_MM, samples.Traffic_Conditions)
plt.scatter(samples.Time_Taken_To_Reach_Patient_MM, samples.Time_For_Sample_Collection_MM)
plt.scatter(samples.Time_Taken_To_Reach_Patient_MM, samples.Time_For_Sample_Collection_MM)

import seaborn as sns

#'Sample ' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Sample'])

samples['Sample'] = winsor.fit_transform(samples[['Sample']])
sns.boxplot(samples.Sample)



#'Way_Of_Storage_Of_Sample ' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Way_Of_Storage_Of_Sample'])

samples['Way_Of_Storage_Of_Sample'] = winsor.fit_transform(samples[['Way_Of_Storage_Of_Sample']])
sns.boxplot(samples.Way_Of_Storage_Of_Sample)


samples.columns

#'Agent_Location_KM ' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Agent_Location_KM'])

samples['Agent_Location_KM'] = winsor.fit_transform(samples[['Agent_Location_KM']])
sns.boxplot(samples.Agent_Location_KM)


#'Time_For_Sample_Collection_MM ' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Time_For_Sample_Collection_MM'])

samples['Time_For_Sample_Collection_MM'] = winsor.fit_transform(samples[['Time_For_Sample_Collection_MM']])
sns.boxplot(samples.Time_For_Sample_Collection_MM)


#'Time_Taken_To_Reach_Lab_MM' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Time_Taken_To_Reach_Lab_MM'])

samples['Time_Taken_To_Reach_Lab_MM'] = winsor.fit_transform(samples[['Time_Taken_To_Reach_Lab_MM']])
sns.boxplot(samples.Time_Taken_To_Reach_Lab_MM)



#'Time_Taken_To_Reach_Patient_MM' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Time_Taken_To_Reach_Patient_MM'])

samples['Time_Taken_To_Reach_Patient_MM'] = winsor.fit_transform(samples[['Time_Taken_To_Reach_Patient_MM']])
sns.boxplot(samples['Time_Taken_To_Reach_Patient_MM'])

samples.dtypes
#'Test_Booking_Time_HH_MM ' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Test_Booking_Time_HH_MM'])

samples['Test_Booking_Time_HH_MM'] = winsor.fit_transform(samples[['Test_Booking_Time_HH_MM']])
sns.boxplot(samples['Test_Booking_Time_HH_MM'])

#'Lab_Location_KM' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Lab_Location_KM'])

samples['Lab_Location_KM'] = winsor.fit_transform(samples[['Lab_Location_KM']])
sns.boxplot(samples['Lab_Location_KM'])


#'Lab_Location_KM' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Lab_Location_KM'])

samples['Lab_Location_KM'] = winsor.fit_transform(samples[['Lab_Location_KM']])
sns.boxplot(samples['Lab_Location_KM'])

#'Reached_On_Time' column outliers treat by using winsorization 
#from feature_engine.outliers import Winsorizer
#winsor = Winsorizer(capping_method='iqr', 
                         # tail='both',
                         # fold=1.5,
                          #variables=['Reached_On_Time'])

#samples['Reached_On_Time'] = winsor.fit_transform(samples[['Reached_On_Time']])
#sns.boxplot(samples['Reached_On_Time'])


#'Cut_off_time_HH_MM' column outliers treat by using winsorization 
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['Cut_off_time_HH_MM'])

samples['Cut_off_time_HH_MM'] = winsor.fit_transform(samples[['Cut_off_time_HH_MM']])
sns.boxplot(samples['Cut_off_time_HH_MM'])


samples.dtypes

### Normalization 
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

samples = norm_func(samples)


samples = samples.iloc[:, [13,0,1,2,3,4,5,6,7,8,9,10,11,12]]

## model building
#  svm=support vector mechine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(samples, test_size = 0.20)

train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y) #0.9705882352941176

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # 0.9705882352941176


model_rbf.fit(test_X, test_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # 0.9656862745098039


# saving the model
# importing pickle
import pickle
pickle.dump(model_rbf, open('model_rbf.pkl', 'wb'))

# load the model from disk
model_rbf = pickle.load(open('model_rbf.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(samples.iloc[0:1,:13])
list_value

print(model_rbf.predict(list_value))
