import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import time 
INTERVAL = 1
IMPORTING_INTERVAL = 10

#creating a class of monitoring, it will read data, save it into an array, converted then to a csv each hour 
#variable  = nombre de la variable a monitorear
#position  = posicion de la variable a monitorear
#filename = una variable tipo string

class data_monitoring:
    def __init__(self, max_value, variable, predictive_model, filename, position):
        self.max_value = max_value
        self.variable = variable 
        self.filename = filename + '.csv'
        self.position = position
        
        self.data = pd.read_csv('monitoring_data.csv', usecols=['time', self.variable])
        
        self.df_variable = self.data.iloc[::position] 
        
        self.df_results = pd.DataFrame(columns= ['time',self.variable]) #empty columns 
        
        self.model = LinearRegression()
        self.predictions = [] 
        self.prediction_times = []
        
    def storing_data(self): 
       last_saved_time = time.time()
                   
       for _, fila in self.df_variable.iterrows():
           
           if fila [self.variable] > self.max_value:
               print(f'Exceeded {self.variable} in {fila[self.variable]} at {fila["time"]}')
           new_row = pd.DataFrame([[fila['time'], fila[self.variable]]], columns=['time', self.variable])
           self.df_results = pd.concat([self.df_results, new_row], ignore_index=True)
           
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
           
       self.df_results.to_csv(self.filename, index=False) #guarda el archivo en 
       print(self.df_results)
       self.plotting()
       
       

    def absolute_manual_error(self):
        y_true = self.df_variable[self.variable].values
        y_pred = self.model.predict(self.df_variable['time'].values.reshape(-1, 1))
        manual_mae = sum(abs(y_true - y_pred)) / len(y_true)
        print(f"Error absoluto medio manual: {manual_mae}")
    
        
    def plotting(self):
       plt.figure(figsize=(10, 5))
       # Plot actual data
       plt.plot(self.df_results['time'], self.df_results[self.variable], label='Actual Data', color='blue')
       # Plot predictions
       plt.plot(self.prediction_times, self.predictions, label='Predicted Data', color='orange', linestyle='--')
       plt.xlabel('Time')
       plt.ylabel(self.variable)
       plt.title(f'Actual vs Predicted {self.variable}')
       plt.legend()
       plt.show()



data1 = data_monitoring(max_value = 4.0, variable = 'temperature', predictive_model= None, filename = 'temperature_results', position = 2)
data1.storing_data()
data1.absolute_manual_error()
data1.plotting()

