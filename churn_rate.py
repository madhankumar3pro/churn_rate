# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:50:42 2023

@author: Admin
"""

import pandas as pd
import numpy as np
import pandas_profiling as pf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler                               
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
                            
train = pd.read_csv(r'C:\Users\Admin\Downloads\train_sample.csv')
test =  pd.read_csv(r'C:\Users\Admin\Downloads\test_sample.csv')

#Data understanding
train.shape
test.shape
cross_val_score
train.info()
test.info()

train.describe()
test.describe()

train.isna().sum()
test.isna().sum()

#Outliers detection

sns.boxplot(train.points_in_wallet)
sns.boxplot(train.avg_transaction_value)
sns.boxplot(train.avg_time_spent)
sns.boxplot(train.days_since_last_login)


#To na values . There many na values are there that's why.Data is very important

#Mode imputation
b = train['region_category'].mode()[0]
train['region_category'] = train['region_category'].fillna(b)

c = train['preferred_offer_types'].mode()[0]
train['preferred_offer_types'] = train['preferred_offer_types'].fillna(c)

#Mean imputation
d = train['points_in_wallet'].mean()
train['points_in_wallet'] = train['points_in_wallet'].fillna(d)

e = test['region_category'].mode()[0]
test['region_category'] = test['region_category'].fillna(e)

f = test['points_in_wallet'].mean()
test['points_in_wallet'] = test['points_in_wallet'].fillna(f)

#train =train.dropna()

#to replace outliers to upper and lower quantile

Q1 = np.percentile(train['points_in_wallet'], 25,interpolation = 'midpoint')
Q3 = np.percentile(train['points_in_wallet'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
train['points_in_wallet'] = np.where(train['points_in_wallet']>upper,upper,(np.where(train['points_in_wallet']<lower,lower,train['points_in_wallet'])))

B1 = np.percentile(train['avg_transaction_value'], 25,interpolation = 'midpoint')
B3 = np.percentile(train['avg_transaction_value'], 75,
                   interpolation = 'midpoint')
BQR = B3 - B1
Bpper = B3+1.5*BQR
Bower = B1-1.5*BQR
train['avg_transaction_value'] = np.where(train['avg_transaction_value']>Bpper,Bpper,(np.where(train['avg_transaction_value']<Bower,Bower,train['avg_transaction_value'])))



#Drop unwanted columns

train = train.drop(['Unnamed: 0','customer_id','Name','security_no','referral_id','joining_date','last_visit_time'],axis=1)
test = test.drop(['Unnamed: 0','customer_id','Name','security_no','referral_id','joining_date','last_visit_time'],axis=1)

train = train[train['gender']!='Unknown']
test = test[test['gender']!='Unknown']

#Duplicates finding
train.duplicated()=='True'
len([train.duplicated()=='True'])


#Data analysis
# pandas profiling

prof = pf.ProfileReport(train)
prof.to_file(output_file=r'C:\Users\Admin\Desktop\celebal_technical_training sheets\churn_rate.html')

prof1 = pf.ProfileReport(test)
prof1.to_file(output_file=r'C:\Users\Admin\Desktop\celebal_technical_training sheets\churn_rate_test.html')

#Data transformation

scha = ['gender', 'region_category', 'membership_category',
       'joined_through_referral', 'preferred_offer_types',
       'medium_of_operation', 'internet_option',
       'avg_frequency_login_days',
       'used_special_discount',
       'offer_application_preference', 'past_complaint', 'complaint_status',
       'feedback']

label = LabelEncoder()

for x in scha:
    train[x] = label.fit_transform(train[x])

for y in scha:
    test[y] = label.fit_transform(test[y])
    
#Scaling
col = list(train.columns)
col = col[0:18]

col1 = list(test.columns)
col1 = col1[0:18]

scale = MinMaxScaler()

train_y = train.iloc[:,18]
train_x = pd.DataFrame(scale.fit_transform(train.iloc[:,0:18]),columns=col)

test_y = test.iloc[:,18]
test_x = pd.DataFrame(scale.fit_transform(test.iloc[:,0:18]),columns=col1)

#Model building
#GradientBoostingClassifier

model = GradientBoostingClassifier(max_depth=5)
cross_val_score(model,test_x,test_y,cv = 5)

h = list(cross_val_score(model,train_x,train_y,cv = 5))

sum(h)/len(h)

#RandomForest model
clf = RandomForestClassifier(max_depth=5,min_samples_split = 11, random_state=50)

cross_val_score(clf,train.iloc[:,0:18],train.iloc[:,18],cv = 5)
cross_val_score(clf,train_x,train_y,cv = 5)

cross_val_score(clf,test.iloc[:,0:18],test.iloc[:,18],cv = 5)
cross_val_score(clf,test_x,test_y,cv = 5)



