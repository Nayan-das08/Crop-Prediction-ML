import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def skRegression(df, x_lbl, y_lbl):
    #x_train, x_test, y_train, y_test = train_test_split(df[x_lbl], df[y_lbl])
    X = df[x_lbl]
    X = pd.DataFrame(X)
    Y = df[y_lbl]
    Y = pd.DataFrame(Y)

    lr = LinearRegression()
    lr.fit(X, Y)
    
    pred = lr.predict(X)
    skPlot(X, pred, Y)
    #sc = lr.score(x_test.values.reshape(-1,1),y_test)
    #return sc

def skPlot(X, pred, Y):
    plt.plot(X,pred, label='regression', color='r')
    plt.scatter(X,Y, label='test data')
    plt.legend()