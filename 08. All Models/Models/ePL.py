# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import math

class ePL:
    def __init__(self, alpha = 0.001, beta = 0.1, lambda1 = 0.35, tau = np.nan, s = 1000, r = 0.25):
        if tau is np.nan:
            self.hyperparameters = pd.DataFrame({'alpha':[alpha],'beta':[beta], 'lambda1':[lambda1], 'tau':[beta], 's':[s], 'r':[r]})
        else:
            self.hyperparameters = pd.DataFrame({'alpha':[alpha],'beta':[beta], 'lambda1':[lambda1], 'tau':[tau], 's':[s], 'r':[r]})
        self.parameters = pd.DataFrame(columns = ['Center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'mu'])
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
         
    def fit(self, X, y):
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0])
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.hyperparameters.loc[0, 'tau']:
                self.Initialize_Cluster(x, y[k], k+1)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
            if self.parameters.shape[0] > 1:
                self.Similarity_Index()
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            # Compute firing degree
            self.mu(x)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'mu'] * xe.T @ self.parameters.loc[row, 'Gamma']
            Output = Output / sum(self.parameters['mu'])
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        return self.OutputTrainingPhase, self.rules
    
    def evolve(self, X, y):
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.hyperparameters.loc[0, 'tau']:
                self.Initialize_Cluster(x, y[k], k+1)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
            if self.parameters.shape[0] > 1:
                self.Similarity_Index()
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            # Compute firing degree
            self.mu(x)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'mu'] * xe.T @ self.parameters.loc[row, 'Gamma']
            Output = Output / sum(self.parameters['mu'])
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            
    def predict(self, X):
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute firing degree
            self.mu(x)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'mu'] * xe.T @ self.parameters.loc[row, 'Gamma']
            Output = Output / sum(self.parameters['mu'])
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase[-X.shape[0]:]
        
    def Initialize_First_Cluster(self, x, y):
        self.parameters = pd.DataFrame([[x, self.hyperparameters.loc[0, 's'] * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., 1., 1., 1.]], columns = ['Center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'mu'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, x, y, k):
        NewRow = pd.DataFrame([[x, self.hyperparameters.loc[0, 's'] * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., k, 1., 1.]], columns = ['Center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'mu'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)

    def Compatibility_Measure(self, x, i):
        self.parameters.at[i, 'CompatibilityMeasure'] = (1 - (np.linalg.norm(x - self.parameters.loc[i, 'Center']))/x.shape[0] )
            
    def Arousal_Index(self, i):
        self.parameters.at[i, 'ArousalIndex'] = self.parameters.loc[i, 'ArousalIndex'] + self.hyperparameters.loc[0, 'beta']*(1 - self.parameters.loc[i, 'CompatibilityMeasure'] - self.parameters.loc[i, 'ArousalIndex'])
    
    def mu(self, x):
        for row in self.parameters.index:
            self.parameters.at[row, 'mu'] = math.exp( - self.hyperparameters.loc[0, 'r'] * np.linalg.norm(self.parameters.loc[row, 'Center'] - x ) )
           
    def Rule_Update(self, x, y, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
        # Update the cluster center
        self.parameters.at[i, 'Center'] = self.parameters.loc[i, 'Center'] + (self.hyperparameters.loc[0, 'alpha']*(self.parameters.loc[i, 'CompatibilityMeasure'])**(1 - self.hyperparameters.loc[0, 'alpha']))*(x - self.parameters.loc[i, 'Center'])
          
        
    def Similarity_Index(self):
        for i in range(self.parameters.shape[0] - 1):
            l = []
			#if i < len(self.clusters) - 1:
            for j in range(i + 1, self.parameters.shape[0]):
                vi, vj = self.parameters.iloc[i,0], self.parameters.iloc[j,0]
                compat_ij = (1 - ((np.linalg.norm(vi - vj))))
                if compat_ij >= self.hyperparameters.loc[0, 'lambda1']:
                    self.parameters.at[self.parameters.index[j], 'Center'] = ( (self.parameters.loc[self.parameters.index[i], 'Center'] + self.parameters.loc[self.parameters.index[j], 'Center']) / 2)
                    self.parameters.at[self.parameters.index[j], 'P'] = ( (self.parameters.loc[self.parameters.index[i], 'P'] + self.parameters.loc[self.parameters.index[j], 'P']) / 2)
                    self.parameters.at[self.parameters.index[j], 'Gamma'] = np.array((self.parameters.loc[self.parameters.index[i], 'Gamma'] + self.parameters.loc[self.parameters.index[j], 'Gamma']) / 2)
                    l.append(int(i))

        self.parameters.drop(index=self.parameters.index[l,], inplace=True)

    def RLS(self, x, y, xe):
        for row in self.parameters.index:
            self.parameters.at[row, 'P'] = self.parameters.loc[row, 'P'] - ((self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + xe.T @ self.parameters.loc[row, 'P'] @ xe))
            self.parameters.at[row, 'Gamma'] = self.parameters.loc[row, 'Gamma'] + (self.parameters.loc[row, 'P'] @ xe * (y - xe.T @ self.parameters.loc[row, 'Gamma']))