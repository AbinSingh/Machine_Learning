# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:47:55 2018

@author: Abin
"""

import os
os.getcwd()
os.chdir('C:/Users/Abin/Desktop/R & Python Rahul Guidance/Python from Rahul/Logistic Regression')

import pandas as pd
import numpy as np
# read train data

train=pd.read_csv('Credit_Risk_Train_data.csv')

test=pd.read_csv('Credit_Risk_Test_data.csv')

# cretae new columns

train['source']='train'
test['source']='test'

# Combine datasets

fullset=pd.concat([train,test],axis=0)
print(fullset)

desc=fullset.describe()
desc

# calculate NA's
fullset.isnull().sum()

# Get the unique of a variable
fullset.apply(lambda x:len(x.unique()),axis=0)

                              # Visualization
import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
dir(plt)
ax = fig.add_subplot(1,1,1)

ax.hist(fullset['ApplicantIncome'],bins = 100)
plt.xlabel('ApplicantIncome')
plt.ylabel('Applicant')
plt.show()
                              
# Filtering categorical variables

cat_columns=[x for x in fullset.dtypes.index if fullset.dtypes[x]=='object']

# Exclude Loan ID,Loan_status,source,propertyarea and education:

cat_columns=[x for x in cat_columns if x not in ['Loan_ID','Loan_Status','source','Education','Property_Area']]

#Print frequency of categories

for col in cat_columns:
    print ('\nFrequency of Categories for variable %s'%col)
    print (fullset[col].value_counts())

                       # Data Cleaning # Imputing missing values
                          # 1. IMPUTING CATEGORICAL VARIABLE
from scipy.stats import mode 
subset=train.loc[:,['Gender','Married','Dependents','Self_Employed']] # subset from train to drop NA's
subset=subset.dropna(subset=['Gender','Married','Dependents','Self_Employed'])

for cal in cat_columns:
    miss_bool = fullset[cal].isnull()
    fullset.loc[miss_bool,cal]=mode(subset[cal]).mode[0]

# For info #d=mode(subset['Gender']) # returns mode of a category and count
    
# Check the total counts in each cat variable, it should come 981    
for col in cat_columns:
    print ('\nFrequency of Categories for variable %s'%col)
    print (sum(fullset[col].value_counts()))
    
                     # Data Cleaning # Imputing missing values
                          # 1. IMPUTING NUMERICAL VARIABLE

# Determine the average loan amount per default status
fullset.Loan_Status.value_counts()                        
Avg_Loan_Amount = train.pivot_table(values='LoanAmount', index='Loan_Status',aggfunc=np.median)
# Convert it to pandas series object
series_obj=pd.Series(Avg_Loan_Amount.LoanAmount, index=Avg_Loan_Amount.index)

#Get a boolean variable specifying missing amount
miss_bool_1 = fullset['LoanAmount'].isnull() 

fullset.loc[miss_bool_1,'LoanAmount'] = fullset.loc[miss_bool_1,'Loan_Status'].apply(lambda x:series_obj[x])
#fullset.loc[miss_bool_1,'LoanAmount']
#fullset.loc[miss_bool_1,'Loan_Status']
#series_obj['N']

                     # Loan Amount Term into categorical variable
import numpy as np
np.mean(fullset['Loan_Amount_Term']) # produces in decimal which is not at all acceptable

train['Loan_Amount_Term'].value_counts()
fullset['Loan_Amount_Term'].value_counts()
# checking its importance with loan status
Avg_Amount_Term = fullset.pivot_table(values='Loan_Amount_Term', index='Loan_Status',aggfunc=(lambda x:mode(x).count[0]))

# Combining values in Loan_Amount_Term
fullset['LA_Term']=fullset.Loan_Amount_Term
values=fullset['Loan_Amount_Term'].value_counts()
fullset['LA_Term'].value_counts()

fullset.loc[fullset['LA_Term'].isnull(),'LA_Term']=360
fullset.loc[fullset['LA_Term']<360,'LA_Term']='LT_Lesser_360'
fullset.loc[fullset['LA_Term']==480,'LA_Term']='LT_Greater_360'
fullset.loc[fullset['LA_Term']==360,'LA_Term']='LT_Equal_To_360'

fullset['LA_Term']=fullset['LA_Term'].astype(str)

                    # Outlier Detection
train_2=fullset[fullset.source =='train']                    
# Applicant Income                
train_2['ApplicantIncome'].quantile(q=[0.009])
train_2['ApplicantIncome'].quantile(q=[0.985])

fullset.loc[fullset['ApplicantIncome']<1012.925,'ApplicantIncome']=1012.925 # capping
fullset.loc[fullset['ApplicantIncome']>20582.37,'ApplicantIncome']=20582.37 # capping

# Co-Applicant Income

train_2['CoapplicantIncome'].quantile(q=[0.1,0.2,0.3,0.4,0.5,0.6])
train_2['CoapplicantIncome'].quantile(q=[0.95,0.96,0.97,0.98,0.99,1])

fullset.loc[fullset['CoapplicantIncome']>8895.89,'CoapplicantIncome']=8895.89 # capping

# Loan Amount

train['LoanAmount'].quantile(q=[0.95,0.96,0.97,0.98,0.99,1])

fullset.loc[fullset['LoanAmount']>496.36,'LoanAmount']=496.36 # capping


fullset['Credit_History'].value_counts()

# Imputing Credit History

fullset.loc[fullset['Credit_History'].isnull(),'Credit_History']=1.0

fullset['Credit_History']=fullset['Credit_History'].astype(object)


                          # DUMMY VARIABLE CREATION

# Filtering categorical variables

cat_dummy=[x for x in fullset.dtypes.index if fullset.dtypes[x]=='object']
cat_dummy

cat_dummy=[x for x in cat_dummy if x not in ['Loan_ID','source','Loan_Amount_Term']]
# dummy variable creation using get_dummies method

df_dummy = pd.DataFrame(data=fullset, columns=cat_dummy)

dummy_coded=pd.get_dummies(df_dummy,drop_first=True,dtype=float) # with K-1
dummy_coded.dtypes
# using append
#copy_df=fullset.copy()
#append_df=copy_df.append(dummy_coded,sort=True)
fullset.dtypes
dummy_fullset=pd.concat([fullset,dummy_coded],axis=1)
dummy_fullset.dtypes
# dropping ir-relevant variables

cat_dum=cat_dummy.copy()
cat_dum.append('Loan_Amount_Term')
cat_dum.append('Loan_ID')

# final dataset after droping variables
fullset3=dummy_fullset.drop(cat_dum,axis=1)
fullset3.shape
fullset3.columns

fullset3.isnull().sum()

fullset3['Intercept']=1

# SAMPLING
copy_final=fullset3.copy()
Train = copy_final.loc[copy_final['source']=="train"]
Train.shape
Train=Train.drop(['source'],axis=1)
Test = copy_final.loc[copy_final['source']=="test"]
Test=Test.drop(['source'],axis=1)

# train_x and train_y
train_x=Train.drop('Loan_Status_Y',axis=1).copy()
train_y=Train['Loan_Status_Y'].copy()
test_x=Test.drop('Loan_Status_Y',axis=1).copy()
test_y=Train['Loan_Status_Y'].copy()

train_x.shape[1]-1 # 16 column -1 = 15 
e=train_x.values
# Check for multicollinearity using VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

for i in range(train_x.shape[1]-1):
    temp_vif=variance_inflation_factor(train_x.values,i)
    print(train_x.columns[i],": ",temp_vif)


train_x_copy = train_x.copy()
#train_x = train_x_copy.copy()   
#train_x.drop(['LA_Term_LT_Greater_360', 'LA_Term_LT_Lesser_360'], axis = 1, inplace = True)
#train_x.columns

                  # MODEL BUILDING
    
import statsmodels.formula.api as sm
                   # FULL MODEL
modell = sm.Logit(train_y,train_x) 
result = modell.fit()
result.summary2()
                   # Variable Selection

import Variable_Selection
result=Variable_Selection.Backward_Elimination(result,0.1,train_y,train_x)

result.summary2()
  
      
