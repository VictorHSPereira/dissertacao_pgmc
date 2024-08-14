# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 06:10:14 2024

@author: kaike
"""

# Importing libraries
import math
import pygad
import numpy as np
from NMFIS import NMFIS
import statistics as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from permetrics.regression import RegressionMetric

class GEN_NMFIS:
    def __init__(self, n_clusters = 5, num_generations = 10, num_parents_mating = 10, sol_per_pop = 20, error_metric = "RMSE", print_information=False):
        self.n_clusters = n_clusters
        self.error_metric = error_metric
        # Number of generations
        self.num_generations = num_generations
        # Number of solutions to be selected as parents in the mating pool
        self.num_parents_mating = num_parents_mating
        # Number of solutions in the population
        self.sol_per_pop = sol_per_pop

        # Inferior limit
        self.init_range_low = 0
        # Superior limit 2 (not including)
        self.init_range_high = 2
        
        # Print information
        self.print_information = print_information
        
        # Models` Data
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])
        self.last_fitness = None
        self.model = None
        self.columns = None
        self.selected_cols = None

    def fit(self, X, y):
        
        y = y.ravel()
        # Define data
        self.X_train = X
        self.y_train = y
        
        # Number of genes
        num_genes = self.X_train.shape[1]
        
        def on_generation(ga_instance):
            if self.print_information == True:
                print(f"Generation = {ga_instance.generations_completed}")
                print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
                print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")
            self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=self.init_range_low,
                               init_range_high=self.init_range_high,
                               fitness_func=self.genetic_algorithm,
                               on_generation=on_generation,
                               suppress_warnings=True,
                               gene_type=int)
        
        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        
        # Returning the details of the best solution.
        self.solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        if self.print_information == True:
            print(f"Parameters of the best solution : {self.solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Index of the best solution : {solution_idx}")
            
            if ga_instance.best_solution_generation != -1:
                print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
        
        # Saving the GA instance.
        # The filename to which the instance is saved. The name is without extension
        filename = 'Results_Genetic_Algorithm' 
        ga_instance.save(filename=filename)
        
        # # Use the next function to load the saved GA instance.
        # loaded_ga_instance = pygad.load(filename=filename)
        # loaded_ga_instance.plot_fitness()
        
        # Selected cols
        selected_cols = self.solution
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        # Define the columns
        X = X[:,selected_cols]
        
        # Initializing the model
        self.model = NMFIS(n_clusters = self.n_clusters)
        # Train the model
        self.model.fit(X, y)
        
        return self.model
    
    def predict(self, X):
        
        # Selected cols
        selected_cols = self.solution
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        # Define the columns
        X_test = X[:,selected_cols]
        
        # Perform predictions
        y_pred = self.model.predict(X_test)
        
        return y_pred
        
    def genetic_algorithm(self, ga_instance, selected_cols, selected_cols_idx):
        
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        if True not in selected_cols:
            RMSE = np.inf
            NRMSE = np.inf
            NDEI = np.inf
            MAE = np.inf
            MAPE = np.inf
            CPPM = -np.inf
            
        else:
            
            # Separate into train and test
            n = self.X_train.shape[0]
            train = round(n * 0.75)
            
            # Define the columns
            X_train = self.X_train[:train,selected_cols]
            X_val = self.X_train[train:,selected_cols]
            y_train = self.y_train[:train]
            y_val = self.y_train[train:]
            # Initializing the model
            self.model = NMFIS(n_clusters = self.n_clusters)
            # Train the model
            self.model.fit(X_train, y_train)
            # Test the model
            y_pred = self.model.predict(X_val)
            
            # Calculating the error metrics
            # Compute the Root Mean Square Error
            RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
            # Compute the Normalized Root Mean Square Error
            NRMSE = RegressionMetric(y_val, y_pred).normalized_root_mean_square_error()
            # Compute the Non-Dimensional Error Index
            NDEI= RMSE/st.stdev(np.asfarray(y_val))
            # Compute the Mean Absolute Error
            MAE = mean_absolute_error(y_val, y_pred)
            # Compute the Mean Absolute Percentage Error
            MAPE = mean_absolute_percentage_error(y_val, y_pred)
            # Count number of times the model predict a correct increase or decrease
            # Actual variation
            next_y = y_val[1:]
            current_y = y_val[:-1]
            actual_variation = (next_y - current_y) > 0.
            
            # Predicted variation
            next_y_pred = y_pred[1:]
            current_y_pred = y_pred[:-1]
            pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()
    
            # Right?
            correct = actual_variation == pred_variation
            # Correct Percentual Predictions of Movement
            CPPM = (sum(correct).item()/correct.shape[0])*100
        
        if self.error_metric == "RMSE":
            return -RMSE
        
        if self.error_metric == "NRMSE":
            return -NRMSE
        
        if self.error_metric == "NDEI":
            return -NDEI
        
        if self.error_metric == "MAE":
            return -MAE
        
        if self.error_metric == "MAPE":
            return -MAPE
        
        if self.error_metric == "CPPM":
            return CPPM