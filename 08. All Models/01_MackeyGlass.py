# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Kaike Alves
"""

# Import libraries
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from permetrics.regression import RegressionMetric
import statistics as st
import matplotlib.pyplot as plt
from dieboldmariano import dm_test

# Neural Network
# Model
from keras.models import Sequential, Model
# Layers
from keras.layers import Input, InputLayer, Dense, Dropout, Conv1D, GRU, LSTM, MaxPooling1D, Flatten, SimpleRNN
#from tensorflow.keras.layers import Input
from tcn import TCN
# Optimizers
from keras.optimizers import SGD, Adam
# Save network
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys

# Models
sys.path.append(r'Models')
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lssvr import LSSVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from eTS import eTS
from Simpl_eTS import Simpl_eTS
from exTS import exTS
from ePL import ePL
from eMG import eMG
from ePL_plus import ePL_plus
from ePL_KRLS_DISCO import ePL_KRLS_DISCO
from NMFIS import NMFIS
from NTSK import NTSK
from GEN_NMFIS import GEN_NMFIS
from GEN_NTSK import GEN_NTSK


# Including to the path another fold
sys.path.append(r'Functions')
# Import the serie generator
from MackeyGlassGenerator import MackeyGlass


#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "MackeyGlass"

# The theory
# Mackey-Glass time series refers to the following, delayed differential equation:
    
# dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)


# Input parameters
a        = 0.2;     # value for a in eq (1)
b        = 0.1;     # value for b in eq (1)
tau      = 17;		# delay constant in eq (1)
x0       = 1.2;		# initial condition: x(t=0)=x0
sample_n = 6000;	# total no. of samples, excluding the given initial condition

# MG = mackey_glass(N, a = a, b = b, c = c, d = d, e = e, initial = initial)
MG = MackeyGlass(a = a, b = b, tau = tau, x0 = x0, sample_n = sample_n)

def Create_lag(data, ncols, lag, lag_output = None):
    X = np.array(data[lag*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[lag*i:lag*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if lag_output == None:
        return X_new
    else:
        y = np.array(data[lag*(ncols-1)+lag_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y

# Defining the atributes and the target value
X, y = Create_lag(MG, ncols = 4, lag = 6, lag_output = 85)

# Spliting the data into train and test
X_train, X_test = X[201:3201,:], X[5001:5501,:]
y_train, y_test = y[201:3201,:], y[5001:5501,:]

# Spliting the data into train and test
n = X_train.shape[0]
val_size = round(n*0.8)
X_train, X_val = X[:val_size,:], X[val_size:,:]
y_train, y_val = y[:val_size], y[val_size:]

# Fixing y shape
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()


# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape the inputs for deep learning
X_train_DL = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_DL = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_DL = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()


#-----------------------------------------------------------------------------
# Initialize dataframe
#-----------------------------------------------------------------------------

# Summary
columns = ["Model_Name", "NRMSE", "NDEI", "MAPE", "Rules", "Best_Params"]
results = pd.DataFrame(columns = columns)

# Predictions
predictions = pd.DataFrame()
predictions['y_test'] = y_test


#-----------------------------------------------------------------------------
# Classic Models
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# KNN
#-----------------------------------------------------------------------------

Model_Name = "KNN"

# Define Grid Search parameters
parameters = {'n_neighbors': [2, 3, 5, 10, 20]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    KNN = KNeighborsRegressor(**param)
    KNN.fit(X_train,y_train)
    # Make predictions
    y_pred = KNN.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_KNN_params = param

# Optimized parameters
KNN = KNeighborsRegressor(**best_KNN_params)
KNN.fit(X_train,y_train)
# Make predictions
y_pred = KNN.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
KNN_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

##print(f'\n{best_KNN_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_KNN_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Regression Tree
#-----------------------------------------------------------------------------

Model_Name = "Regression Tree"

# Define Grid Search parameters
parameters = {'max_depth': [2, 4, 6, 8, 10, 20, 30, 50, 100]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    RT = DecisionTreeRegressor(**param)
    RT.fit(X_train,y_train)
    # Make predictions
    y_pred = RT.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_RT_params = param

# Optimized parameters
RT = DecisionTreeRegressor(**best_RT_params)
RT.fit(X_train,y_train)
# Make predictions
y_pred = RT.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
RT_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_RT_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Random Forest
#-----------------------------------------------------------------------------

Model_Name = "Random Forest"

# Define Grid Search parameters
parameters = {'n_estimators': [50, 250, 500, 1000, 1500, 2000],'max_depth': [10, 20, 30, 40, 50, 70,100, None],'bootstrap': [True, False]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    RF = RandomForestRegressor(**param)
    RF.fit(X_train,y_train)
    # Make predictions
    y_pred = RF.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_RF_params = param

# Optimized parameters
RF = RandomForestRegressor(**best_RF_params)
RF.fit(X_train,y_train)
# Make predictions
y_pred = RF.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
RF_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

#print(f'\n{best_RF_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_RF_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# SVM
#-----------------------------------------------------------------------------

Model_Name = "SVM"

# Define Grid Search parameters
parameters1 = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C':[0.01,0.1,1,10], 'gamma': [0.01,0.5,1,10,50]}

grid1 = ParameterGrid(parameters1)

lower_rmse = np.inf
for param in grid1:
    
    #print(param)
    
    # Optimize parameters
    SVM = SVR(**param)
    SVM.fit(X_train,y_train)
    # Make predictions
    y_pred = SVM.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_SVM_params = param

# # Use best parameters1 to save time looking for best parameters2
# parameters2 = {'kernel': ['poly'], 'C':[best_SVM_params['C']], 'gamma': [best_SVM_params['gamma']], 'degree': [2,3]}

# grid2 = ParameterGrid(parameters2)

# for param in grid2:
    
#     #print(param)
    
#     # Optimize parameters
#     SVM = SVR(**param)
#     SVM.fit(X_train,y_train)
#     # Make predictions
#     y_pred = SVM.predict(X_val)
#     
    
#     # Calculating the error metrics
#     # Compute the Root Mean Square Error
#     RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
#     if RMSE < lower_rmse:
#         lower_rmse = RMSE
#         best_SVM_params = param

# Optimized parameters
SVM = SVR(**best_SVM_params)
SVM.fit(X_train,y_train)
# Make predictions
y_pred = SVM.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)

SVM_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

#print(f'\n{best_SVM_params}')
   
# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_SVM_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# LS-SVM
#-----------------------------------------------------------------------------

Model_Name = "LS-SVM"

# Optimize parameters
LS_SVR = LSSVR(kernel='linear')
LS_SVR.fit(X_train,y_train)
# Make predictions
y_pred = LS_SVR.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
LS_SVR_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, ""]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Gradient Boosting
#-----------------------------------------------------------------------------

Model_Name = "GB"

# Define Grid Search parameters
parameters = {'learning_rate':[0.01, 0.05, 0.1, 0.5, 0.9], 'n_estimators': [2, 4, 8, 16, 32, 64, 100, 200]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    GB = GradientBoostingRegressor(**param)
    GB.fit(X_train,y_train)
    # Make predictions
    y_pred = GB.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GB_params = param

# Optimized parameters
GB = GradientBoostingRegressor(**best_GB_params)
GB.fit(X_train,y_train)
# Make predictions
y_pred = GB.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)

GB_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

#print(f'\n{best_GB_params}')
   
# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_GB_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# XGBoost
# -----------------------------------------------------------------------------

Model_Name = "XGBoost"

# Define Grid Search parameters
parameters = {'n_estimators':[250, 500], 'min_child_weight':[2,5], 
              'gamma':[i/10.0 for i in range(3,6)], 'max_depth': [2, 3, 4, 6, 7], 
              'eval_metric':['rmse'],'eta':[i/10.0 for i in range(3,6)]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    XGBoost = XGBRegressor(**param)
    XGBoost.fit(X_train,y_train)
    # Make predictions
    y_pred = XGBoost.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_XGBoost_params = param

# Optimized parameters
XGBoost = XGBRegressor(**best_XGBoost_params)
XGBoost.fit(X_train,y_train)
# Make predictions
y_pred = XGBoost.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
XGBoost_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

#print(f'\n{best_XGBoost_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_XGBoost_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Light-GBM Regressor
#-----------------------------------------------------------------------------

Model_Name = "LGBM"

# Define Grid Search parameters
parameters = {'learning_rate':[0.01, 0.05, 0.1, 0.5, 0.9],'verbosity':[-1]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    LGBM = LGBMRegressor(**param)
    LGBM.fit(X_train,y_train)
    # Make predictions
    y_pred = LGBM.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_LGBM_params = param

# Optimized parameters
LGBM = LGBMRegressor(**best_LGBM_params)
LGBM.fit(X_train,y_train)
# Make predictions
y_pred = LGBM.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
LGBM_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

#print(f'\n{best_LGBM_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_LGBM_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Deep Learning
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# MLP
#-----------------------------------------------------------------------------


Model_Name = "MLP"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, activation = "relu", learning_rate=3e-3, input_shape=[4]):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_MLP_parameters = {'n_hidden':2, 'n_neurons':77, 'activation':"relu", 'learning_rate': 0.145, 'input_shape': X_train_DL.shape[1]}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_MLP_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
MLP_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_MLP_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# CNN
#-----------------------------------------------------------------------------


Model_Name = "CNN"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_CNN_parameters = {'n_hidden':3, 'n_neurons':97, 'learning_rate': 0.006}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_CNN_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
CNN_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_CNN_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# RNN
#-----------------------------------------------------------------------------

Model_Name = "RNN"

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3):
    model = Sequential()
    if n_hidden == 1:
        model.add(SimpleRNN(n_neurons, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    elif n_hidden == 2:
        model.add(SimpleRNN(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        model.add(SimpleRNN(n_neurons))
    else:
        model.add(SimpleRNN(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        for layer in range(1, n_hidden-1):
            model.add(SimpleRNN(n_neurons, return_sequences=True))
        model.add(SimpleRNN(n_neurons))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_RNN_parameters = {'n_hidden':5, 'n_neurons':21, 'learning_rate': 0.003}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_RNN_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
RNN_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_RNN_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# LSTM
#-----------------------------------------------------------------------------

Model_Name = "LSTM"

# Define the function to create models for the optimization method
def build_model(n_neurons=30, n_lstm_hidden = 1, neurons_dense = 30, dropout_rate = 0, n_dense_hidden = 1, learning_rate=3e-3):
    model = Sequential()
    if n_lstm_hidden == 1:
        model.add(LSTM(n_neurons, return_sequences=False, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    elif n_lstm_hidden == 2:
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        model.add(LSTM(n_neurons, return_sequences=False))
    else:
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
        for layer in range(1, n_lstm_hidden-1):
            model.add(LSTM(n_neurons, return_sequences=True))
        model.add(LSTM(n_neurons, return_sequences=False))
    for dense_layer in range(n_dense_hidden):
        model.add(Dense(neurons_dense))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_LSTM_parameters = {'n_neurons':83, 'n_lstm_hidden':4, 'neurons_dense': 1, 'dropout_rate': 0, 'n_dense_hidden':2, 'learning_rate': 0.006}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_LSTM_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
LSTM_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_LSTM_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# GRU
#-----------------------------------------------------------------------------

Model_Name = "GRU"

# Define the function to create models for the optimization method
def build_model(filters = 64, kernel_size = 2, strides = 1, n_neurons=30, n_gru_hidden = 1, neurons_dense = 30, dropout_rate = 0, n_dense_hidden = 1, learning_rate=3e-3):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="valid", input_shape=(X_train_DL.shape[1],X_train_DL.shape[2])))
    if n_gru_hidden == 1:
        model.add(GRU(n_neurons, return_sequences=False))
    elif n_gru_hidden == 2:
        model.add(GRU(n_neurons, return_sequences=True))
        model.add(GRU(n_neurons, return_sequences=False))
    else:
        #model.add(keras.layers.GRU(n_neurons, return_sequences=True))
        for layer in range(n_gru_hidden-1):
            model.add(GRU(n_neurons, return_sequences=True))
        model.add(GRU(n_neurons, return_sequences=False))
    for dense_layer in range(n_dense_hidden):
        model.add(Dense(neurons_dense))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_GRU_parameters = {'filters':4, 'kernel_size':2, 'strides': 2, 'n_neurons': 76, 'n_gru_hidden':4, 'neurons_dense': 0, 'dropout_rate': 0, 'n_dense_hidden': 0, 'learning_rate': 0.016}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_GRU_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
GRU_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_GRU_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# TCN
#-----------------------------------------------------------------------------


Model_Name = "TCN"

# Define the function to create models for the optimization method
def build_model(kernel_size = 1, dilations = [1], n_tcn_hidden = 1, neurons = 30, dropout_rate = 0, n_dense_hidden = 1, learning_rate=3e-3):
    
    i = Input(batch_shape=X_train_DL.shape)
    if n_tcn_hidden == 1:
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=False)(i)
    elif n_tcn_hidden == 2:
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=True)(i)
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=False)(o)
    else:
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=True)(i)
        for layer in range(1,n_tcn_hidden-1):
            o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=True)(o)
        o = TCN(kernel_size = kernel_size, dilations = dilations, return_sequences=False)(o)
    for dense_layer in range(n_dense_hidden):
        o = Dense(neurons)(o)
        o = Dropout(dropout_rate)(o)
    o = Dense(1)(o)
    
    model = Model(inputs=[i], outputs=[o])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_TCN_parameters = {'kernel_size':4, 'dilations':[1,2,4], 'n_tcn_hidden':4, 'neurons': 0, 'dropout_rate': 0, 'n_dense_hidden': 0, 'learning_rate': 0.002}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_TCN_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
TCN_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_TCN_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# WaveNet
#-----------------------------------------------------------------------------

Model_Name = "WaveNet"

# Define the function to create models for the optimization method
def build_model(dilation_rate=(1, 2, 4, 8), repeat=2, learning_rate=3e-3):
    model = Sequential()
    model.add(InputLayer(input_shape=[X_train_DL.shape[1],X_train_DL.shape[2]]))
    for rate in dilation_rate * repeat:
        model.add(Conv1D(filters=20, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
    model.add(Conv1D(filters=10, kernel_size=1))
    model.add(Flatten())
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Set the parameters for the network
best_WaveNet_parameters = {'dilation_rate':(1,2,4), 'repeat':2, 'learning_rate': 0.007}

# Checkpoint for early stopping
checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{Serie}_{Model_Name}.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20,restore_best_weights=True)

# Define the model
model = build_model(**best_WaveNet_parameters)

# Fit the model
history = model.fit(X_train_DL, y_train, epochs = 1000, validation_data=(X_val_DL, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

# Make predictions
y_pred = model.predict(X_test_DL)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred.ravel()

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
Rules = "-"
print("Rules:", Rules)
   
WaveNet_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_WaveNet_parameters]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# evolving Fuzzy Systems
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# eTS
#-----------------------------------------------------------------------------


Model_Name = "eTS"

# Define Grid Search parameters
parameters = {'InitialOmega': [50, 100, 250, 500, 750, 1000, 10000], 'r': [0.1, 0.3, 0.5, 0.7, 0.9, 1., 5, 10, 50]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = eTS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_eTS_params = param

# Optimized parameters
model = eTS(**best_eTS_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])

   
eTS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_eTS_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_eTS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# Simpl_eTS
#-----------------------------------------------------------------------------


Model_Name = "Simpl_eTS"

# Define Grid Search parameters
parameters = {'InitialOmega': [50, 250, 500, 750, 1000], 'r': [0.1, 0.3, 0.5, 0.7]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = Simpl_eTS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_Simpl_eTS_params = param

# Optimized parameters
model = Simpl_eTS(**best_Simpl_eTS_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])
   
Simpl_eTS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_Simpl_eTS_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_Simpl_eTS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# exTS
#-----------------------------------------------------------------------------


Model_Name = "exTS"

# Define Grid Search parameters
parameters = {'InitialOmega': [50, 250, 500, 750, 1000], 'mu_threshold': [0.1, 0.3, 0.5, 0.7]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = exTS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_exTS_params = param

# Optimized parameters
model = exTS(**best_exTS_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])
   
exTS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_exTS_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_exTS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# ePL
#-----------------------------------------------------------------------------


Model_Name = "ePL"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01, 0.1, 0.5, 0.9], 'beta': [0.001, 0.005, 0.01, 0.1, 0.2], 'lambda1': [0.001, 0.01, 0.1], 's': [100, 10000], 'r': [0.1, 0.25, 0.5, 0.75]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = ePL(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_ePL_params = param

# Optimized parameters
model = ePL(**best_ePL_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])
   
ePL_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_ePL_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_ePL_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# eMG
#-----------------------------------------------------------------------------


Model_Name = "eMG"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01], 'lambda1': [0.1, 0.5], 'w': [10, 50], 'sigma': [0.001, 0.003], 'omega': [10**4]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = eMG(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_eMG_params = param

# Optimized parameters
model = eMG(**best_eMG_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])
   
eMG_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_eMG_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_eMG_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# ePL+
#-----------------------------------------------------------------------------


Model_Name = "ePL_plus"

# Define Grid Search parameters
parameters = {'alpha': [0.001, 0.01, 0.1], 'beta': [0.01, 0.1, 0.25], 'lambda1': [0.25, 0.5, 0.75], 'omega': [100, 10000], 'sigma': [0.1, 0.25, 0.5], 'e_Utility': [0.03, 0.05], 'pi': [0.3, 0.5]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = ePL_plus(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_ePL_plus_params = param

# Optimized parameters
model = ePL_plus(**best_ePL_plus_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])
   
ePL_plus_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_ePL_plus_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_ePL_plus_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


#-----------------------------------------------------------------------------
# ePL-KRLS-DISCO
#-----------------------------------------------------------------------------


Model_Name = "ePL_KRLS_DISCO"

# Define Grid Search parameters
parameters = {'alpha': [0.05, 0.1], 'beta': [0.01, 0.1, 0.25], 'lambda1': [0.0000001, 0.001], 'sigma': [0.5, 1, 10, 50], 'e_utility': [0.03, 0.05]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = ePL_KRLS_DISCO(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_ePL_KRLS_DISCO_params = param

# Optimized parameters
model = ePL_KRLS_DISCO(**best_ePL_KRLS_DISCO_params)
OutputTraining, Rules = model.fit(X_train,y_train)
# Make predictions
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", Rules[-1])
   
ePL_KRLS_DISCO_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules[-1]}'

#print(f'\n{best_ePL_KRLS_DISCO_params}')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules[-1], best_ePL_KRLS_DISCO_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# Proposed Models
#-----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# NMFIS
# -----------------------------------------------------------------------------


Model_Name = "NMFIS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20)}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NMFIS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_NMFIS_params = param

# Initialize the model
model = NMFIS(**best_NMFIS_params)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Compute the number of final rules
Rules = model.parameters.shape[0]
print("Rules:", Rules)

# Save the model to excel
model.parameters.to_excel(f'Model Summary/{Model_Name}_parameters.xlsx')

NMFIS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_NMFIS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "NTSK-RLS"

# Set hyperparameters range
parameters = {'n_clusters':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'RLS_option':[1]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_NTSK_RLS_params = param

# Initialize the model
model = NTSK(**best_NTSK_RLS_params)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Compute the number of final rules
Rules = model.parameters.shape[0]
print("Rules:", Rules)  

NTSK_RLS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Save the model to excel
model.parameters.to_excel(f'Model Summary/{Model_Name}_parameters.xlsx')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_NTSK_RLS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "NTSK-wRLS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20), 'RLS_option':[2]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_NTSK_wRLS_params = param

# Initialize the model
model = NTSK(**best_NTSK_wRLS_params)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Compute the number of final rules
Rules = model.parameters.shape[0]
print("Rules:", Rules)  

# Save the model to excel
model.parameters.to_excel(f'Model Summary/{Model_Name}_parameters.xlsx')

NTSK_wRLS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_NTSK_wRLS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# GEN-NMFIS
# -----------------------------------------------------------------------------


Model_Name = "GEN-NMFIS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20)}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NMFIS(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GEN_NMFIS_params = param

# Initialize the model
model = GEN_NMFIS(**best_GEN_NMFIS_params)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Compute the number of final rules
Rules = model.model.parameters.shape[0]
print("Rules:", Rules)

# Save the model to excel
model.model.parameters.to_excel(f'Model Summary/{Model_Name}_parameters.xlsx')

GEN_NMFIS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_GEN_NMFIS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# GEN-NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-RLS"

# Set hyperparameters range
parameters = {'n_clusters':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'RLS_option':[1]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GEN_NTSK_RLS_params = param

# Initialize the model
model = GEN_NTSK(**best_GEN_NTSK_RLS_params)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Compute the number of final rules
Rules = model.model.parameters.shape[0]
print("Rules:", Rules)  

GEN_NTSK_RLS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Save the model to excel
model.model.parameters.to_excel(f'Model Summary/{Model_Name}_parameters.xlsx')

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_GEN_NTSK_RLS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)


# -----------------------------------------------------------------------------
# GEN-NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-wRLS"

# Set hyperparameters range
parameters = {'n_clusters':range(1,20), 'RLS_option':[2]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GEN_NTSK_wRLS_params = param

# Initialize the model
model = GEN_NTSK(**best_GEN_NTSK_wRLS_params)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)

# Save predictions to dataframe
predictions[f'{Model_Name}'] = y_pred

# Model name
print(f'\nResults for {Model_Name}: \n')

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RegressionMetric(y_test, y_pred).normalized_root_mean_square_error()
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(np.asfarray(y_test.flatten()))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Compute the number of final rules
Rules = model.model.parameters.shape[0]
print("Rules:", Rules)  

# Save the model to excel
model.model.parameters.to_excel(f'Model Summary/{Model_Name}_parameters.xlsx')

GEN_NTSK_wRLS_ = f'{Model_Name} & {NRMSE:.2f} & {NDEI:.2f} & {MAPE:.2f} & {Rules}'

# Store results to dataframe
newrow = pd.DataFrame([[Model_Name, NRMSE, NDEI, MAPE, Rules, best_GEN_NTSK_wRLS_params]], columns=columns)
results = pd.concat([results, newrow], ignore_index=True)



#-----------------------------------------------------------------------------
# Save results to excel
#-----------------------------------------------------------------------------


# Save results
results.to_excel(f'Results/Results_{Serie}.xlsx')

# Save predictions
predictions.to_excel(f'Predictions/Predictions_{Serie}.xlsx')


#-----------------------------------------------------------------------------
# Print results
#-----------------------------------------------------------------------------


# Print the results
for i in results.index:
    print(f'\n{i}:')
    print(f'{results.loc[i,"Model_Name"]} & {results.loc[i,"NRMSE"]:.5f} & {results.loc[i,"NDEI"]:.5f} & {results.loc[i,"MAPE"]:.5f} & {results.loc[i,"Rules"]}')
    print(f'{results.loc[i,"Best_Params"]}')
    

#-----------------------------------------------------------------------------
# DM test
#-----------------------------------------------------------------------------


# Columns
col = predictions.columns

print("\n")
# Write columns name
for i in col:
    print(i, end=" & ")

print("\n")

print(f'Model & {col[-3]} & {col[-2]} & {col[-1]} ')
for i in range(1, len(col)-6):
    print("\n")
    
    dm, pvalue1 = dm_test(predictions[col[0]].values, predictions[col[-6]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue2 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-6]].values, one_sided=True)
    
    dm, pvalue3 = dm_test(predictions[col[0]].values, predictions[col[-5]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue4 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-5]].values, one_sided=True)

    dm, pvalue5 = dm_test(predictions[col[0]].values, predictions[col[-4]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue6 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-4]].values, one_sided=True)
    
    dm, pvalue7 = dm_test(predictions[col[0]].values, predictions[col[-3]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue8 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-3]].values, one_sided=True)
    
    dm, pvalue9 = dm_test(predictions[col[0]].values, predictions[col[-2]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue10 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-2]].values, one_sided=True)

    dm, pvalue11 = dm_test(predictions[col[0]].values, predictions[col[-1]].values, predictions[col[i]].values, one_sided=True)
    dm, pvalue12 = dm_test(predictions[col[0]].values, predictions[col[i]].values, predictions[col[-1]].values, one_sided=True)

    print(f'{col[i]} & {pvalue1:.2f} & {pvalue3:.2f} & {pvalue5:.2f} & {pvalue7:.2f} & {pvalue9:.2f} & {pvalue11:.2f}')


