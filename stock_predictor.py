import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_and_plot_stock_data():
    # Open file dialog to select the CSV
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Load stock data from CSV
    stock_data = pd.read_csv(file_path)
    
    # Convert the Date column to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data['Date'], stock_data['Close'], label='Closing Price')
    ax.set(title='Stock Closing Prices Over Time', xlabel='Date', ylabel='Price')
    ax.grid(True)
    ax.legend()
    
    # Display the plot inside the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

# Create the main window
window = tk.Tk()
window.title("Stock Data Plotter")

# Create a button to load the CSV and plot the data
load_button = tk.Button(window, text="Load Stock Data", command=load_and_plot_stock_data)
load_button.pack(side=tk.TOP, pady=20)

# Start the GUI event loop
window.mainloop()
