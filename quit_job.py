import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

rf=False
knc=True  

X = pd.read_csv(r'C:\Users\mlaurenzi.CODIN\Downloads\HR_comma_sep.csv')
for elem in X:
    print elem
    print X[elem].unique()

    ordered_salary = X['salary'].unique()

X.salary = X['salary'].astype("category",
  ordered=True,
  categories=ordered_salary).cat.codes

y = X['left'].copy()
X.drop('left', axis=1, inplace=True)
    
X.sales = X.sales.map({'sales':1,'accounting':0,'hr':2,'technical':3,'support':4,
                         'management':5,'IT':6,'product_mng':7,'marketing':8,'RandD':9})

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1,
                                                    random_state=7)

mio = pd.Series([0.3, 0.7, 5, 180, 6, 0, 1, 6, 0], index=['satisfaction_level','last_evaluation','number_project',
                       'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years',
                       'sales','salary'])

if rf:
    RForest = RandomForestClassifier(n_estimators=30,max_depth=13,oob_score=True,random_state=0)
    RForest.fit(X_train, y_train) 
    score = RForest.score(X_test,y_test)
    print "Score: ", round(score*100, 3)
    score = RForest.oob_score_
    print "OOB Score: ", round(score*100, 3)  
    print type(X_test.iloc[5])
    print X_test.iloc[7]
    print RForest.predict(mio.values.reshape(1,-1))
    print RForest.score(mio.values.reshape(1,-1),[1])

#print RForest.predict(X_test.iloc[7].values.reshape(1,-1))
#print RForest.score(X_test.iloc[7].values.reshape(1,-1),[y_test.iloc[7]])
elif knc:
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train,y_train)
    score = knn.score(X_test,y_test)
    print "Score: KNC", round(score*100, 3)
    print RForest.predict(mio.values.reshape(1,-1))

sys.exit()
    

