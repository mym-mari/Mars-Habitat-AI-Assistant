import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import time 
INTERVAL = 1
IMPORTING_INTERVAL = 10

#variable  = nombre de la variable a monitorear
#position  = posicion de la variable a monitorear
#filename = una variable tipo string

class ai_mars: #class ai, monitors the data, it saves it each timestep into a csv file and predicts new data
    def __init__(self, max_value, variable, filename, position):
        self.max_value = max_value
        self.variable = variable 
        self.filename = filename + '.csv'
        self.position = position
        
        self.data = pd.read_csv("/mnt/c/Users/maria/Desktop/Cuarto semestre/Programación/monitoring_data.csv", usecols=['time', self.variable])
        #for general use, put in the first argument the location of the file (that i will attach in the mail)
        self.df_variable = self.data.iloc[::position] #takes the variable of interest and storages it 
        
        self.df_results = pd.DataFrame(columns= ['time',self.variable]) #empty columns for later adding data, empty dataframe
        
        self.model = LinearRegression() #calls a function of linear regression
        self.predictions = [] #creates an empty array for predictions in the variable of interest
        self.prediction_times = [] #creates an empty array for time
        
    def storing_data(self): 
       last_saved_time = time.time()
                   
       for _, fila in self.df_variable.iterrows(): #a for cicle that iterates each value of the variable of interest
           
           if fila [self.variable] > self.max_value: #conditional for checking if the value exceeds certain number
               print(f'Exceeded {self.variable} in {fila[self.variable]} at {fila["time"]}') #prints the exceeded value at a certain moment
           new_row = pd.DataFrame([[fila['time'], fila[self.variable]]], columns=['time', self.variable]) #creates a new row, of a dataframe with the analyzed values
           self.df_results = pd.concat([self.df_results, new_row], ignore_index=True) #adds the row to the empty array
           
           X = self.df_variable['time'].values.reshape(-1, 1)
           y = self.df_variable[self.variable].values
           self.model.fit(X, y)
           next_time = fila['time'] + 1  # Ajusta según corresponda tu intervalo de tiempo
           prediction = self.model.predict(np.array([[next_time]]))[0]
           self.predictions.append(prediction)
           self.prediction_times.append(next_time)
           
           
           print(new_row)
           print(f"Predicción para el tiempo {next_time}: {prediction}")
           
           if time.time() - last_saved_time >= IMPORTING_INTERVAL:
                self.df_results.to_csv(self.filename, index=False)
                print(f'Partial results saved to {self.filename} at interval')
                last_saved_time = time.time() 
                
           time.sleep(INTERVAL)
           self.df_results.to_csv(self.filename, index=False)
           print(self.df_results)
           
       self.df_results.to_csv(self.filename, index=False) 
       print(self.df_results)
       self.plotting()
       
       

    def absolute_manual_error(self):
        y_true = self.df_variable[self.variable].values
        y_pred = self.model.predict(self.df_variable['time'].values.reshape(-1, 1))
        manual_mae = sum(abs(y_true - y_pred)) / len(y_true)
        print(f"Error absoluto medio manual: {manual_mae}")
    
        
    def plotting(self):
       plt.figure(figsize=(10, 5))
       plt.plot(self.df_results['time'], self.df_results[self.variable], label='Actual Data', color='blue')
       plt.plot(self.prediction_times, self.predictions, label='Predicted Data', color='orange', linestyle='--')
       plt.xlabel('Time')
       plt.ylabel(self.variable)
       plt.title(f'Actual vs Predicted {self.variable}')
       plt.legend()
       plt.show()



data1 = ai_mars(max_value = 4.0, variable = 'temperature', filename = 'temperature_results', position = 2)
data1.storing_data()
data1.absolute_manual_error()
data1.plotting()

