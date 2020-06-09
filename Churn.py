import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\Practises\\Churn")

FullRaw = pd.read_csv("Churn_Modelling.csv")

FullRaw.drop(['RowNumber','CustomerId','Surname'], axis =1, inplace =True)

FullRaw.isnull().sum()

Category_Vars = (FullRaw.dtypes =='object')
dummyDf = pd.get_dummies(FullRaw.loc[:,Category_Vars],drop_first =True)

FullRaw2 = pd.concat([FullRaw.loc[:,~Category_Vars],dummyDf], axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw2,test_size = 0.3, random_state =123)

Train_X = Train.drop(['Exited'], axis =1)
Train_Y = Train['Exited'].copy()
Test_X = Test.drop(['Exited'], axis =1)
Test_Y = Test['Exited'].copy()

from sklearn.linear_model import LogisticRegression

M1_Model = LogisticRegression(random_state=123).fit(Train_X,Train_Y)

Test_Pred = M1_Model.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_Pred,Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

from sklearn.ensemble import RandomForestClassifier

RF_Model = RandomForestClassifier(random_state=123).fit(Train_X,Train_Y)

RF_Pred = RF_Model.predict(Test_X)

RF_Con = confusion_matrix(RF_Pred,Test_Y)
sum(np.diag(RF_Con))/Test_Y.shape[0]*100

from sklearn.model_selection import GridSearchCV

n_tree= [50,75,100]
n_split = [100,200,300]
n_depth = [80,100]

my_param_grid = {'n_estimators': n_tree, 'min_samples_split': n_split, 'max_depth': n_depth}

Grid = GridSearchCV(RandomForestClassifier(random_state=123),param_grid= my_param_grid,
                    cv = 5, scoring = 'accuracy').fit(Train_X,Train_Y)

Grid = pd.DataFrame.from_dict(Grid.cv_results_)

Grid[['param_max_depth','param_min_samples_split','param_n_estimators','mean_test_score']]

import pickle

pickle.dump(RF_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

model.predict([[376,29,4,115047,4,1,0,119347,1,0,0]])




