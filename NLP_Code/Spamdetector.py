from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd

data = pd.read_csv('spambase.data').values
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print(f'the classification rate for NB {model.score(Xtest,Ytest)}')

model1 = AdaBoostClassifier()
model1.fit(Xtrain, Ytrain)
print(f'the classification rate for adaboost {model1.score(Xtest,Ytest)}')


