import customtkinter
import pandas as pd
import numpy as np
from tkinter import filedialog, messagebox
import joblib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
from scipy.stats import skew, kurtosis

# Load model and scaler from joblib
model = joblib.load('logistic_model.joblib') # Import model to the app
scaler = joblib.load('scaler.joblib') # Import scaler for preprocessing of input

customtkinter.set_appearance_mode("light")  # Set app theme
customtkinter.set_default_color_theme("blue") # Set app colour

# Reference: https://www.youtube.com/watch?v=iM3kjbbKHQU (for custom tkinter GUIs)

def moving_average_filter(data, window_size=5): #  Define the moving average filter
    filtered_data = np.zeros_like(data) # Create an array to store the filtered data (default is zeros)
    for segment in range(data.shape[0]):
        for feature in range(data.shape[2]):
            df = pd.DataFrame(data[segment, :, feature])  # Convert the time series for the current segment and feature into a DataFrame
            rolling_mean = df.rolling(window_size, center=True, min_periods=1).mean() # Apply rolling mean
            filtered_data[segment, :, feature] = rolling_mean.to_numpy().flatten() # Store the results back into the filtered_data array
    return filtered_data
    
def extract_features_from_segment(segment_data): # Define the feature extraction function
    features = []
    for axis_data in segment_data.T:  # Transpose to iterate over axes
        # Time-domain features
        axis_features = {
            'mean': np.mean(axis_data),
            'std_dev': np.std(axis_data),
            'max': np.max(axis_data),
            'min': np.min(axis_data),
            'median': np.median(axis_data),
            'range': np.max(axis_data) - np.min(axis_data),
            'iqr': np.percentile(axis_data, 75) - np.percentile(axis_data, 25),
            'variance': np.var(axis_data),
            'skewness': skew(axis_data),
            'kurtosis': kurtosis(axis_data),
        }
        features.append(axis_features)
    
    # Flatten the list of dictionaries into a single dictionary
    flattened_features = {}
    for i, axis_feature in enumerate(features):
        for key, value in axis_feature.items():
            flattened_features[f'axis_{i}_{key}'] = value
    
    return flattened_features
    
def segment_into_windows(data, window_size=5):
    sampling_rate = data['Time (s)'].diff().median() # Sample rate is necessary to determine the number of samples per window
    samples_per_window = int(window_size / sampling_rate) # Calculate the number of samples per window
    segmented_data = [] # Create an empty list to store the segmented data
    for start in range(0, len(data), samples_per_window): # Iterate over the data with a step size of samples_per_window
        end = start + samples_per_window
        if end <= len(data): # Will drop the data that does not fill a window
            segmented_data.append(data.iloc[start:end]) # Append the data to the list
    return segmented_data

class PredictionApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title('Walking or Jumping Prediction')
        self.geometry('600x600')  # App display size

        # Initialize GUI components and plot area
        self.initialize_gui_components()
        self.create_plot_area()

    def initialize_gui_components(self):
        self.frame = customtkinter.CTkFrame(master=self) # Create a frame to hold the components
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)

        self.label = customtkinter.CTkLabel(master=self.frame, text="Were you walking or jumping?", font=("Roboto", 18)) # Create a label
        self.label.pack(pady=12, padx=10)

        self.load_button = customtkinter.CTkButton(master=self.frame, text="Load CSV", command=self.load_csv) # Create a button for loading csv
        self.load_button.pack(pady=12, padx=10)

        self.save_button = customtkinter.CTkButton(master=self.frame, text="Specify Output CSV Location", command=self.specify_output_csv) # Create a button for output location
        self.save_button.pack(pady=12, padx=10)

        self.predict_button = customtkinter.CTkButton(master=self.frame, text="Predict and Save Output", command=self.predict_and_save) # Create a button for prediction
        self.predict_button.pack(pady=12, padx=10)

    def create_plot_area(self): # Create plot initially
        mplstyle.use('seaborn-v0_8-darkgrid')
        
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor="#dbdbdb") # plot colour to match background
        self.plot = self.fig.add_subplot(1, 1, 1)
        
        # Add "Awaiting Input" text in the middle of the plot
        self.plot.text(0.5, 0.5, 'Awaiting Input', fontsize=12, ha='center', va='center', transform=self.plot.transAxes, color='gray')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame) # Reference: https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/ (for embedded graphs)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=customtkinter.BOTTOM, fill=customtkinter.BOTH, expand=True)

    def update_plot(self, predictions_labels): # Update plot upon input data
        categories = ['Walking', 'Jumping'] # Split between two categories
        counts = [predictions_labels.count(cat) for cat in categories]

        self.plot.clear()
        self.plot.bar(categories, counts, color=['darkblue', 'lightblue'])
        self.plot.set_title('Classification Results')
        self.plot.set_xlabel('Category')
        self.plot.set_ylabel('Count (5s Segments)')
        self.canvas.draw()

    def load_csv(self): # Load the file
        self.csv_file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if self.csv_file_path:
            messagebox.showinfo("Information", "CSV Loaded Successfully")

    def specify_output_csv(self): # Specify the output path
        self.output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[('CSV Files', '*.csv')])
        if self.output_file_path:
            messagebox.showinfo("Information", "Output CSV Location Specified")

    def predict_and_save(self): # Handles input data, preprocessing and running the model
        if self.csv_file_path and self.output_file_path:
    
            csv_data = pd.read_csv(self.csv_file_path) # Load the CSV
            
            segments = segment_into_windows(csv_data)
            data_segmented = np.array([segment[['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']].values for segment in segments]) # extract the features needed, dropping time and absolute acceleration       
            data = moving_average_filter(data_segmented)
            data_features_list = []
            
            for i in range(data.shape[0]):
                segment_features = extract_features_from_segment(data[i])
                data_features_list.append(segment_features)
            
            data = pd.DataFrame(data_features_list) # Convert the list of dictionaries to a DataFrame

            data_scaled = scaler.transform(data) # Scale the features using the scaler from the training process
            
            data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns) # Convert the scaled features back to Pandas DataFrames
            
            predictions = model.predict(data_scaled_df) # Predict the resuls based on the data input 
            
            predictions_labels = ['Walking' if pred == 0 else 'Jumping' for pred in predictions] # Convert numerical predictions back to labels
            
            # Create a DataFrame with window numbers and labels
            windows = list(range(1, len(predictions_labels) + 1))  # Starting window number from 1 for readability
            output_df = pd.DataFrame({'Window #': windows, 'Predicted Activity': predictions_labels}) # Create a DataFrame with the window numbers and predictions

            output_df.to_csv(self.output_file_path, index=False) # Save the DataFrame with predictions to CSV

            self.update_plot(predictions_labels) # Update the plot with the labels

            messagebox.showinfo("Success", "Predictions Saved to CSV Successfully")
        else:
            messagebox.showwarning("Warning", "Please Load a CSV File and Specify an Output Location First")
        
if __name__ == '__main__':
    app = PredictionApp()
    app.mainloop()