# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
from scipy.stats import norm

class NMFIS:
    def __init__(self, n_clusters = 5):
        self.hyperparameters = pd.DataFrame({'n_clusters':[n_clusters]})
        self.parameters = pd.DataFrame(columns = ['Center', 'std', 'y_mean', 'y_std', 'NumObservations', 'mu', 'firing_min', 'firing_max', 'firing_prod'])
        #boself.parameters1 = pd.DataFrame(columns = ['Center', 'std', 'y_mean', 'y_std', 'NumObservations', 'mu', 'firing_min', 'firing_max', 'firing_prod'])
        self.n_clusters = n_clusters
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        self.OutputTestPhaseInterval = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        self.ResidualTestInterval = np.array([])
         
    def fit(self, X, y):
        # Prepare the output
        y = y.reshape(-1,1)
        # Concatenate X with y
        Data = np.concatenate((X, y.reshape(-1,1), np.zeros((X.shape[0], 1))), axis=1)
        # Compute the number of attributes
        m = X.shape[1]
        # Compute the number of samples
        n = X.shape[0]
        # Compute the width of each interval
        self.ymin = min(Data[:, m])
        self.ymax = max(Data[:, m])
        self.region = ( self.ymax - self.ymin ) / ( self.hyperparameters.loc[0, 'n_clusters'] )
        # Compute the cluster of the inpute
        for row in range(1, n):
            if Data[row, m] < self.ymax:
                rule = int( ( Data[row, m] - self.ymin ) / self.region )
                Data[row, m + 1] = rule
            else:
                rule = int( ( Data[row, m] - self.ymin ) / self.region )
                Data[row, m + 1] = rule - 1
        # Create a dataframe from the array
        df = pd.DataFrame(Data)
        empty = []
        # Initializing the rules
        for rule in range(self.hyperparameters.loc[0, 'n_clusters']):
            if df[df[m + 1] == rule].shape[0] == 0:
                empty.append(rule)
            dfnew = df[df[m + 1] == rule]
            center = dfnew.loc[:,:m-1].mean().values.reshape(-1,1)
            std = dfnew.loc[:,:m-1].std().values.reshape(-1,1)
            y_center = dfnew.loc[:,m].mean()
            y_std = dfnew.loc[:,m].std()
            num_obs = dfnew.loc[:,m].shape[0]
            if np.isnan(std).any:
                std[np.isnan(std)] = 1.
            if 0. in std:
                std[std == 0.] = 1.
            if np.isnan(y_std):
                y_std = 1.
            if y_std == 0.:
                y_std = 1.
            if rule == 0:
                # Initialize the first rule
                self.Initialize_First_Cluster(y[0], center, std, y_center, y_std, num_obs)
            else:
                # Initialize the first rule
                self.Initialize_Cluster(y[0], center, std, y_center, y_std, num_obs)
        if empty != 0:
            self.parameters = self.parameters.drop(empty)
        # Compute the output in the training phase
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the membership function
            self.mu(x)
            # Compute the firing degree for x
            self.firing()
            # Compute the output
            Output = sum( self.parameters['y_mean'] * self.parameters['firing_prod'] ) / sum( self.parameters['firing_prod'] )
            # Store the output
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
        # Return the predictions
        return self.OutputTrainingPhase
            
    def predict(self, X):
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the membership function
            self.mu(x)
            # Compute the firing degree for x
            self.firing()
            # Compute the output
            if sum( self.parameters['firing_prod']) == 0:
                Output = 0
            else:
                Output = sum( self.parameters['y_mean'] * self.parameters['firing_prod'] ) / sum( self.parameters['firing_prod'] )
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase
    
    def predict_interval(self, X, alpha=0.05):
        # Compute the Critical Value
        cv = norm.ppf(1 - alpha/2)
        self.OutputTestPhaseInterval = np.zeros((X.shape[0],2))
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the membership function
            self.mu(x)
            # Compute the firing degree for x
            self.firing()
            # Compute the output
            Output = sum( self.parameters['y_mean'] * self.parameters['firing_prod'] ) / sum( self.parameters['firing_prod'] )
            #Sigma = np.sqrt( ( sum( (self.parameters['y_std']**2) * ( self.parameters['NumObservations'] - 1 ) * ( self.parameters['firing_prod']  / sum( self.parameters['firing_prod'] ) ) ) ) / ( sum( self.parameters['NumObservations'] ) - self.parameters.shape[0] ) )
            Sigma = np.sqrt ( sum( (self.parameters['y_std']**2) * ( self.parameters['NumObservations'] - 1 ) ) / ( sum( self.parameters['NumObservations'] ) - 1 ) )
            # Compute the Confidence Interval
            Low = Output - cv * Sigma
            High = Output + cv * Sigma
            Interval = np.array([Low,High])
            # Store the output
            self.OutputTestPhaseInterval[k,] = Interval

        return self.OutputTestPhaseInterval
    
    def Initialize_First_Cluster(self, y, center, std, y_center, y_std, num_obs):
        self.parameters = pd.DataFrame([[center, std, y_center, y_std, num_obs, np.array([]), 1., 1., 1.]], columns = ['Center', 'std', 'y_mean', 'y_std', 'NumObservations', 'mu', 'firing_min', 'firing_max', 'firing_prod'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, y, center, std, y_center, y_std, num_obs):
        NewRow = pd.DataFrame([[center, std, y_center, y_std, num_obs, np.array([]), 1., 1., 1.]], columns = ['Center', 'std', 'y_mean', 'y_std', 'NumObservations', 'mu', 'firing_min', 'firing_max', 'firing_prod'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)

    def mu(self, x):
        for row in self.parameters.index:
            self.parameters.at[row, 'mu'] = np.exp( - ( x - self.parameters.loc[row, 'Center'] )**2 / ( 2 * self.parameters.loc[row, 'std']**2 ) ).reshape(-1,1)
            
    def firing(self):
        for row in self.parameters.index:
            self.parameters.at[row, 'firing_min'] = self.parameters.loc[row, 'mu'].min().item()
            self.parameters.at[row, 'firing_max'] = self.parameters.loc[row, 'mu'].max().item()
            self.parameters.at[row, 'firing_prod'] = self.parameters.loc[row, 'mu'].prod().item()