# RandomForestClassifier 결정 트리들의 숲
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

wine = pd.read_csv('data/wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy() # sklearn requires numpy array
target = wine['class'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# n_jobs=-1 : cpu 최대수 활용
# n_estimators : 결정트리 갯수
rf = RandomForestClassifier(n_jobs=-1, random_state=42)

# return_train_score : train_score도 포함
scores = cross_validate(rf, X_train, y_train, return_train_score=True, n_jobs=-1)
# print(scores)
print(np.mean(scores['train_score']))
print(np.mean(scores['test_score']))

rf.fit(X_train, y_train)
print(rf.predict(X_test[:5]))
print(rf.feature_importances_)
# print(rf.score(X_train, y_train))
# print(rf.score(X_test, y_test))

