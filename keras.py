
# coding: utf-8

# # A Deep Learning Approach to the Titanic Data Set
# 
# A recent obsession of mine has been the intersection of Bayesian Inference and Deep learning.  To start, I will build a Keras model.  Unfortunately Kaggle doesn't support Edward, my preferred package for Bayesian Inference so those pieces will be down outside the Kernal, though I will publish the code here once it's written.


# Data Structures
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Prediction
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
np.random.seed(606)


# The following functions encompass a data cleaning pipeline. The function preproc at the end wraps the rest so that a single function call will return the desired data set. 


def split_and_clean():
    X, y = select_features(pd.read_csv('train.csv'), test = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 606, stratify = y)
    return X_train, y_train, X_test, y_test
    
def select_features(data, test = 0):
    target = ['Survived']
    features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    dropped_features = ['Cabin', 'Ticket', 'Name']
    X = data[features].drop(dropped_features, axis=1)
    if test == 0:
        y = data[target]
        return X, y
    else:
        return X

def fix_na(data):
    na_vars = {"Age" : data.Age.mean(), "Fare" : data.Fare.mean(), "Embarked" : "C"}
    return data.fillna(na_vars)

def create_dummies(data, cat_vars, cat_types):
    cat_data = data[cat_vars].values
    for i in range(len(cat_vars)):   
        bins = LabelBinarizer().fit_transform(cat_data[:, 0].astype(cat_types[i]))
        cat_data = np.delete(cat_data, 0, axis=1)
        cat_data = np.column_stack((cat_data, bins))
    return cat_data

def standardize(data, real_vars):
    real_data = data[real_vars]
    scale = StandardScaler()
    return scale.fit_transform(real_data)

def preproc():
    # Import Data & Split
    X_train, y_train, X_test, y_test = split_and_clean()
    # Fill NAs
    X_train, X_test = fix_na(X_train), fix_na(X_test)
    # Preproc Categorical Vars
    cat_vars = ['Pclass', 'Sex', 'Embarked']
    cat_types = ['int', 'str', 'str']
    X_train_cat, X_test_cat = create_dummies(X_train, cat_vars, cat_types), create_dummies(X_test, cat_vars, cat_types)
    # Preprocess Numeric Vars
    real_vars = ['Age', 'Fare', 'SibSp', 'Parch']
    X_train_real, X_test_real = standardize(X_train, real_vars), standardize(X_test, real_vars)
    # Recombine
    X_train, X_test = np.column_stack((X_train_cat, X_train_real)), np.column_stack((X_test_cat, X_test_real))
    return X_train, np_utils.to_categorical(y_train.values), X_test, np_utils.to_categorical(y_test.values)


# Run the preproc pipeline

X_train, y_train, X_test, y_test = preproc()


# Now we can build a Keras model.  At the top we define a series of variables that we'll use in the model.

NB_EPOCH = 200
BATCH_SIZE = 32
VERBOSE = 0
NB_CLASSES = 1 # number of outputs
OPTIMIZER = Adam() # Adam optimizer
N_HIDDEN = 80 # Number of nodes in hidden layer
VALIDATION_SPLIT=0.3 # how much TRAIN is reserved for VALIDATION
FEATURES = X_train.shape[1]
DROPOUT = 0.2
LAYERS = 3

model = Sequential([
    Dense(64, input_shape=(FEATURES, ), activation='relu'),
    Dropout(DROPOUT),
])

for i in range(LAYERS):
    model.add(Dense(N_HIDDEN, activation = 'relu'))
    model.add(Dropout(DROPOUT))
model.add(Dense(2, activation = 'softmax'))
model.summary()


# ### The last thing to do is to compile and fit the model.


model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['acc'])
early_stopping = EarlyStopping(patience=40)

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH, callbacks=[early_stopping],
                      validation_split=VALIDATION_SPLIT)


score = model.evaluate(X_test, y_test, )
print("Test score:", score[0])
print('Test accuracy:', score[1])


def preproc_testing():
    X = pd.read_csv('test.csv')
    # Fill NAs
    X = fix_na(X)
    # Preproc Categorical Vars
    cat_vars = ['Pclass', 'Sex', 'Embarked']
    cat_types = ['int', 'str', 'str']
    X_cat = create_dummies(X, cat_vars, cat_types)
    # Preprocess Numeric Vars
    real_vars = ['Age', 'Fare', 'SibSp', 'Parch']
    X_real = standardize(X, real_vars)
    # Recombine
    X = np.column_stack((X_cat, X_real))
    return X


testing = preproc_testing()
prediction = model.predict_classes(testing)


submission = pd.DataFrame()
submission['PassengerId'] = pd.read_csv('test.csv').PassengerId
submission['Survived'] = prediction


submission.to_csv('keras_titanic.csv', index=False)