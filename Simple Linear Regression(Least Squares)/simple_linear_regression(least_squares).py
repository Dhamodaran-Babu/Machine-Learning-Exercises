# -*- coding: utf-8 -*-
"""Simple Linear Regression(Least Squares)


# Simple Linear Regression
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Experience vs savings %.xlsx')
dataset

"""# **Data Visualisation**"""

dataset.plot(x='Experience(in months)',y='% of savings from income',kind='scatter',
            title='Experience VS % of Savings')

df=dataset.copy()
print("Test for null values in the dataset : {}".format(np.any(df.isnull().values==True)))

dependent_var=df.iloc[:,1].values
independent_var=df.iloc[:,0].values
np.set_printoptions(precision=4)

"""# Min max scaling technique"""

class Scaler:
    def __init__(self):
        self.min=None
        self.max=None
    def scale(self,data):
        if self.min is None and self.max is None:
            self.min=data.min()
            self.max=data.max()
        return (data-self.min) / (self.max-self.min)

    def reverse_scaling(self,data):
        return (data*(self.max-self.min))+self.min

xscaler=Scaler()
yscaler=Scaler()
x=xscaler.scale(independent_var)
y=yscaler.scale(dependent_var)

"""# Train Test Split"""

def splitter(x,y,train_size=0.75,seed=None):
    np.random.seed(seed)
    data=np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)
    np.random.shuffle(data)
    xtrain=data[:int(len(data)*train_size),0]
    ytrain=data[:int(len(data)*train_size),1]
    xtest=data[int(len(data)*train_size):,0]
    ytest=data[int(len(data)*train_size):,1]
    return xtrain,ytrain,xtest,ytest

xtrain,ytrain,xtest,ytest=splitter(x,y,train_size=0.85,seed=101)

"""# Method Of Least Squares"""

def least_square(x,y):
    xmean=x.mean()
    ymean=y.mean()
    num=((x-xmean)*(y-ymean)).sum(axis=0)
    den=((x-x.mean())**2).sum(axis=0)
    weight=num/den
    bias=ymean-(weight*xmean)
    
    return weight,bias

def predict(x,weight,bias):
    return (weight*x)+bias

def make_predictions_from_user():
    x=float(input("Enter the value for experience(in months)"))
    print("""!!! MAKING PREDICTIONS!!!""")
    print("The predicted % of savings for given experience {} months is : {} %"
          .format(x,yscaler.reverse_scaling(predict(xscaler.scale(x),weight,bias))))

weight,bias=least_square(xtrain,ytrain)
print("weight :{} , bias : {}".format(weight,bias))

"""# Defining the Metrics"""

def mse(true,pred):
    return np.mean((pred-true)**2)

def rmse(true,pred):
    return mse(true,pred)**0.5

def r_squared(true,pred):
    true_mean=true.mean()
    pred_mean=pred.mean()
    tot=((true-true_mean)**2).sum(axis=0)
    obs=((true-pred)**2).sum(axis=0)
    return 1-(obs/tot)

"""# Making Predictions"""

ypred=yscaler.reverse_scaling(predict(xtest,weight,bias))
ytrue=yscaler.reverse_scaling(ytest)

print("Predictions")
print(ypred)

print("MSE : ",mse(ytrue,ypred))
print("RMSE : ",rmse(ytrue,ypred))
print("MAE : ",mae(ytrue,ypred))
print("R-squared Value",r_squared(ytrue,ypred))

"""# Visualization of the fitted model"""

plt.scatter(independent_var,dependent_var,color='darkcyan',label='Actual Values')
plt.plot(independent_var,
    yscaler.reverse_scaling(predict(xscaler.scale(independent_var),weight,bias)),'--',
    color='crimson',label='Fitted Least Square line')
plt.xlabel('Experience(in months)')
plt.ylabel('Savings %')
plt.title('Experience Vs Savings(LSM model)')
plt.legend()
plt.tight_layout()
plt.show()

make_predictions_from_user()
