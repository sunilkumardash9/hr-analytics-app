import pandas as pd 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/home/sunilkumardash9/Documents/hranalytics/HR_comma_sep.csv')

df.rename(columns={'Departments ':'departments'}, inplace=True)
df['salary'] = df['salary'].map({'low':0, 'medium':1, 'high':2})

enc = LabelEncoder()
df['departments'] = enc.fit_transform(df.departments)

from sklearn.model_selection import train_test_split
y = df['left']

df.drop('left', axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.15)


from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class my_classifier(BaseEstimator,):
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y=None):
        self.estimator.fit(X,y)
        return self
    def predict(self, X, y=None):
        return self.estimator.predict(X,y)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def score(self, X, y):
        return self.estimator.score(X, y)

pipe = Pipeline([ ('clf', my_classifier())])

parameters = [
             {'clf':[RandomForestClassifier()],
             'clf__n_estimators': [75, 100, 125,],
             'clf__min_samples_split': [2,4,6],
             'clf__max_depth': [5, 10, 15,]

             },
           ]
grid = GridSearchCV(pipe, parameters, cv=5, scoring='roc_auc')
grid.fit(x_train,y_train)

model = grid.best_estimator_
score = grid.best_score_

print(f'The estimator is found to be {model} with an ROC-AUC score of {score}')

y_pred = model.predict(x_test)

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred)

print(f'The ROC-AUC for test data is found to be {roc_auc}')

from joblib import dump

dump(model, 'my-model2')