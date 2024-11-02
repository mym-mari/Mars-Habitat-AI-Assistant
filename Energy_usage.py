import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
import time 
INTERVAL = 1
IMPORTING_INTERVAL = 10

#variable  = nombre de la variable a monitorear
#position  = posicion de la variable a monitorear
#filename = una variable tipo string
#max_value = valor máximo que tomaría la variable para que retorne una advertencia
#n_estimators = parámetro para un modelo
#model_type = modelo predictivo

class ai_mars: #class ai, monitors the data, it saves it each timestep into a csv file and predicts new data
    def __init__(self, max_value, variable, filename, position, model_type, n_estimators):
        self.max_value = max_value
        self.variable = variable 
        self.filename = filename + '.csv'
        self.position = position
        self.model_type = model_type
        self.n_estimators = n_estimators
        
        self.data = pd.read_csv("/mnt/c/Users/maria/Desktop/Cuarto semestre/Programación/monitoring_data.csv", usecols=['time', self.variable])
        #for general use, put in the first argument the location of the file (that i will attach in the mail)
        
        self.df_variable = self.data.iloc[::position] #takes the variable of interest and storages it 
        
        self.df_results = pd.DataFrame(columns= ['time',self.variable]) #empty columns for later adding data, empty dataframe
        
        self.predictions = [] #creates an empty array for predictions in the variable of interest
        self.prediction_times = [] #creates an empty array for time
        
    def storing_data(self): 
       last_saved_time = time.time()
                   
       for _, fila in self.df_variable.iterrows(): #a for cicle that iterates each value of the variable of interest
           
           if fila [self.variable] > self.max_value: #conditional for checking if the value exceeds certain number
               print(f'Exceeded {self.variable} in {fila[self.variable]} at {fila["time"]}') #prints the exceeded value at a certain moment
               
           new_row = pd.DataFrame([[fila['time'], fila[self.variable]]], columns=['time', self.variable]) #creates a new row, of a dataframe with the analyzed values
           self.df_results = pd.concat([self.df_results, new_row], ignore_index=True) #adds the row to the empty array
           
           #ADDING A PREDICTIVE MODEL ai helped with the aproximation, can either choose a linear model or a forest model
           if self.model_type == 'forest':
               self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42) #calls a function of random forests
           else:    
               self.model = LinearRegression() #calls a function of linear regression
          
           
           #trains both models 
           X = self.df_variable['time'].values.reshape(-1, 1)
           y = self.df_variable[self.variable].values
           self.model.fit(X, y)
           
           #predicts for next time
           next_time = fila['time'] + 0.01  
           prediction = self.model.predict(np.array([[next_time]]))[0]
           self.predictions.append(prediction) 
           self.prediction_times.append(next_time)
           
           
           print(new_row)
           print(f"Prediction for time {next_time}: {prediction}")
           
           #Converts the df.results into a csv file and exports it one the timestep has passed
           if time.time() - last_saved_time >= IMPORTING_INTERVAL:
                self.df_results.to_csv(self.filename, index=False)
                print(f'Partial results saved to {self.filename} at interval')
                last_saved_time = time.time() 
                
           time.sleep(INTERVAL)
           print(self.df_results)
           
       print(self.df_results)
       self.plotting()
       
            
    def plotting(self): #plots both things (the original data and the predictions) in the same graphic
       plt.figure(figsize=(10, 5))
       plt.plot(self.df_results['time'], self.df_results[self.variable], label='Actual Data', color='blue')
       plt.plot(self.prediction_times, self.predictions, label='Predicted Data', color='orange', linestyle='--')
       plt.xlabel('Time')
       plt.ylabel(self.variable)
       plt.title(f'Actual vs Predicted {self.variable}')
       plt.legend()
       plt.show()



data1 = ai_mars(max_value = 4.0, variable = 'temperature', filename = 'temperature_results', position = 2, model_type = 'forest', n_estimators= 200)
data1.storing_data()
data1.plotting()

