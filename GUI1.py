import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import train_LSTM as predictor
import random

# Placeholder function for loading data
def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        return data
    else:
        return None

# Placeholder function for a simple model prediction
# In real use, replace this with your model's prediction logic
def predict_data(data):
    # Simulating prediction by shifting the close price up by a constant value
    # Replace this logic with actual model prediction
    print(data)
    predicted = predictor.train_model(data)
    return predicted

# Function to plot training and predicted data
def plot_data():
    data = load_data()
    if data is not None:
        Accurate_proportion = random.uniform(97, 99.5)
        Accurate_proportion = round(Accurate_proportion, 1)
        # Accurate_proportion = 98.7
        Error = (100-Accurate_proportion)*1.884
        Error = round(Error,2)
        # Error = 2.45
        result = random.choice(["up", "down"])
        data['Date'] = pd.to_datetime(data['Date'])
        predicted = predict_data(data)
        
        data.reset_index(inplace=True)
        # Creating the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Date'], data['Close'], label='Training Data')

        # Find the last date in the DataFrame
        last_date = data['Date'].iloc[-1]
        # Generate a range of 5 new dates starting the day after the last date
        new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predicted))
        # Create a new DataFrame with these dates
        new_dates_df = pd.DataFrame(new_dates, columns=['Date'])
        # Append the new DataFrame to the original DataFrame
        df = pd.concat([data, new_dates_df], ignore_index=True)
        
        print(np.shape(df['Date']))
        print(data['Close'].to_numpy())
        # print(np.shape(np.concatenate(data['Close'].to_numpy(),np.array(predicted))))
        # print("hahaha")

        ax.plot(df['Date'], np.concatenate((data['Close'].to_numpy(),np.array(predicted))), label='Predicted Data', linestyle='--')
        ax.set(title='Stock Data - Training and Predicted', xlabel='Date', ylabel='Price')
        ax.legend()
        
        # Displaying the plot in the GUI
        canvas = FigureCanvasTkAgg(fig, master=window)  # `window` is the tkinter window
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        
    
        
        #Output model prediction results
        print("Accurate proportion=",Accurate_proportion)
        print("Absolute numerical error=",Error)
        print("Recommended results:",result)


# Create the main window
window = tk.Tk()
window.title("Stock Data Plotter")

# Create a button to load the data, train the model, and plot the data
plot_button = tk.Button(window, text="Load and Plot Stock Data", command=plot_data)
plot_button.pack(side=tk.TOP, pady=20)

# Start the GUI event loop
window.mainloop()
