import pandas as pd
from tables import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import tensorflow as tf

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

X = train._drop_axis(['y'], axis=1)
y = train['y']

X = np.array(X)
y = np.array(y)
X_test_set = np.array(test)

## Keras:
# create model
def create_network():
    classifier = Sequential()
    classifier.add(Dense(100, input_dim=100, activation='relu'))
    classifier.add(Dense(100, activation='relu'))
    classifier.add(Dense(100, activation='relu'))
    classifier.add(Dense(100, activation='relu'))
    classifier.add(Dense(5, activation='softmax'))

    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return classifier

neural_network = KerasClassifier(build_fn=create_network,
                                 epochs=20,
                                 verbose=0
                                 )
y_pred=[]
score=[0]
current_score=[]
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    current_score=np.mean(cross_val_score(neural_network, X_train, y_train))
    print(current_score)
    if current_score>score:
        score=current_score
        neural_network.fit(X_train, y_train)
        y_pred = neural_network.predict(X_test_set)
        print("New neural network")
        print(len(y_pred))
    else:
        pass

# output results
d={'Id': test.index, 'y': y_pred}
output=pd.DataFrame(d)
output.to_csv('output1.csv', index=False)
