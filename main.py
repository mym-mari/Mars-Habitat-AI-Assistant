import time 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Constants for logging and reading intervals and file extension
LOG_INTERVAL = 10  # Time interval in seconds for logging data to CSV
READ_INTERVAL = 1  # Time interval in seconds for reading new sensor data
FILE_EXTENSION = ".csv"  # File extension for the output files

# Variables to keep track of the last logged and read times
lastLoggedTime = 0
lastReadTime = 0

# Class to monitor sensor data and make predictions
class ai_mars: 
    SENSORNOISE = 0.5  # Standard deviation of the noise to be added to sensor readings

    def __init__(self, max_values, variables, filenames, positions, model_type, n_estimators):
        # Initialize instance variables
        self.max_values = max_values  # Maximum allowed values for each monitored variable
        self.variables = variables  # Names of the monitored variables
        self.filenames = [s + FILE_EXTENSION for s in filenames]  # Output file names
        self.positions = positions  # Positions of the monitored variables in the input data
        self.model_type = model_type  # Type of predictive model ('forest' or other)
        self.n_estimators = n_estimators  # Number of estimators for Random Forest model
        self.readCounter = 0  # Counter for how many readings have been taken
        self.lastReading = [0,0,0]  # Last readings for each variable
        
        # Load initial data for the monitored variables
        self.data = [pd.read_csv("./monitoring_data.csv", usecols=['Time', var]) for var in self.variables]

        # Store the initial values of the monitored variables
        self.df_variable = [self.data[index].iloc[0:len(self.data[index])] for index, pos in enumerate(self.positions)] 
        
        # DataFrames to store results and predictions for each variable
        self.df_results = [pd.DataFrame(columns=['Time', var]) for var in self.variables]
        self.predictions = [pd.DataFrame(columns=['Time', var]) for var in self.variables]
        
    def storing_data(self): 
        # Store data from the last reading and make predictions
        for i in range(len(self.df_variable)):
            # Check if the last reading exceeds the maximum allowed value
            if self.lastReading[i][self.variables[i]] > self.max_values[i]: 
                print(f'Exceeded {self.variables[i]} in {self.lastReading[i][self.variables[i]]} at {self.lastReading[i]["Time"]}')
                
            # Create a new DataFrame row with the latest reading
            new_row = pd.DataFrame([[self.lastReading[i]['Time'], self.lastReading[i][self.variables[i]]]], columns=['Time', self.variables[i]])

            # Append the new row to the results DataFrame
            self.df_results[i] = pd.concat([self.df_results[i], new_row], ignore_index=True) 
            
            # Initialize the predictive model (Random Forest or Linear Regression)
            if self.model_type == 'forest':
                self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42) 
            else:    
                self.model = LinearRegression() 
            
            # Train the model using existing data
            X = self.df_variable[i]['Time'].values.reshape(-1, 1)    
            y = self.df_variable[i][self.variables[i]].values
            self.model.fit(X, y)

            # Make predictions based on the trained model
            self.predict(i)
            
    def plotting(self): 
        # Plot actual vs. predicted data for each monitored variable
        for i in range(len(self.variables)):
            plt.figure(figsize=(12, 6))
            
            # Plot actual data
            plt.plot(self.df_results[i]['Time'], self.df_results[i][self.variables[i]], 
                     label='Actual Data', color='blue', marker='o', markersize=4)
            
            # Plot predicted data if it exists
            if not self.predictions[i].empty:
                plt.plot(self.predictions[i]['Time'], self.predictions[i][self.variables[i]], 
                         label='Predicted Data', color='orange', linestyle='--', marker='x', markersize=6)
            
            # Set labels and title for the plot
            plt.xlabel('Time', fontsize=14)
            plt.ylabel(self.variables[i], fontsize=14)
            plt.title(f'Actual vs Predicted {self.variables[i]}', fontsize=16)
            plt.legend()  # Show legend
            plt.grid(True)  # Add a grid for better readability
            plt.tight_layout()  # Adjust layout
            plt.show()  # Show the plot

    def logData(self):
        # Save results DataFrames to CSV files if the logging interval has elapsed
        for i in range(len(self.df_results)):
            self.df_results[i].to_csv(self.filenames[i], index=False)

    def predict(self, i):
        # Predict the next value for the monitored variable
        next_time = self.lastReading[i]['Time'] + 0.01  
        prediction = self.model.predict(np.array([[next_time]]))[0]
        new_prediction_row = pd.DataFrame([[next_time, prediction]], columns=['Time', self.variables[i]])
        self.predictions[i] = pd.concat([self.predictions[i], new_prediction_row], ignore_index=True)

    def readTemperature(self):
        # Read new temperature data and add sensor noise
        if(self.readCounter > 99): return 0  # Stop if 100 readings have been taken
        for i in range(3):
            self.lastReading[i] = self.df_variable[i].iloc[self.readCounter]  # Get last reading
            # Add noise to the reading
            self.lastReading[i][self.variables[i]] += np.random.normal(loc=0, scale=self.SENSORNOISE) 
        self.readCounter += 1  # Increment the read counter
        return 1

def readDelay():
    # Check if enough time has passed since the last read
    global lastReadTime
    if time.time() - lastReadTime >= READ_INTERVAL:
        lastReadTime = time.time() 
        return 1
    else: 
        return 0

def logDelay():
    # Check if enough time has passed since the last log
    global lastLoggedTime
    if time.time() - lastLoggedTime >= LOG_INTERVAL:
        lastLoggedTime = time.time() 
        return 1
    else: 
        return 0

# Configuration for monitoring variables
variableMax = [3.5, 3.9, 0.04]  # Maximum allowed values for each variable (thresholds)
variableNames = ["Temperature", "Oxygen", "Energy"]  # Names of the monitored variables
variablePosition = [2, 3, 4]  # Positions of the monitored variables in the input data
outputFilesNames = ["tempResults", "OxResults", "EgyResults"]  # Output file names for results

if __name__ == '__main__':
    # Main execution block
    ai1 = ai_mars(variableMax, variableNames, outputFilesNames, variablePosition, model_type = 'forest', n_estimators= 200)  # Create an instance of ai_mars
    run = 1  # Flag to control the main loop
    while(run):
        if(readDelay()):  # Check if it's time to read new data
            if(not(ai1.readTemperature())):  # Read new temperature data
                run = 0  # Stop running if no more readings are available
                break
            ai1.storing_data()  # Store the new data
            
        if(logDelay()):  # Check if it's time to log data
            ai1.logData()  # Log data to CSV files
            
    ai1.plotting()  # Plot the results and predictions
    print("Killed")  # Print message when the program ends