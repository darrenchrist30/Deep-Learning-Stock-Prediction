import os
import numpy as np
import pandas as pd
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Fix matplotlib backend to avoid tkinter threading issues
# Must be done before importing matplotlib modules
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend

from tensorflow.keras.models import load_model
import pickle
import json
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import glob
import re
import uuid
import traceback

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "saham-prediction-app"

# Dictionary to store the last uploaded CSV path
global_data = {
    'last_csv_path': None
}

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Global variables
ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if model exists
MODEL_PATH = 'models/lstm.h5'
SCALER_PATH = 'models/scaler.pkl'
model = None
scaler = None

def get_available_models():
    """Get list of available models in models directory"""
    models_list = []
    # Get all .h5 files in models directory
    model_files = glob.glob('models/*.h5')
    for model_file in model_files:
        # Extract just the filename without path
        model_name = os.path.basename(model_file)
        models_list.append(model_name)
    return models_list

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ...existing code...

def load_saved_model(model_name=None):
    global model, scaler, MODEL_PATH, SCALER_PATH
    
    # Update model path if model_name is provided
    if model_name:
        MODEL_PATH = os.path.join('models', model_name)
        # Update scaler path based on model name (remove .h5 and add .pkl)
        base_name = os.path.splitext(model_name)[0]
        SCALER_PATH = os.path.join('models', f"{base_name}.pkl")
        print(f"Setting model path to: {MODEL_PATH}")
        print(f"Setting scaler path to: {SCALER_PATH}")
    
    try:
        print(f"Checking if model exists at {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully")
        else:
            print(f"Model file not found at {MODEL_PATH}")
            model = None
            
        # Make scaler optional
        print(f"Checking if scaler exists at {SCALER_PATH}")
        if os.path.exists(SCALER_PATH):
            print(f"Loading scaler from {SCALER_PATH}")
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully")
        else:
            print("Creating default scaler")
            # Create a simple MinMaxScaler as fallback
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            print("Using default MinMaxScaler")
            
        success = model is not None
        print(f"Model loading success: {success}")
        return success
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return False

def prepare_data(data, sequence_length=60):
    """Prepare data for prediction"""
    # Make sure we're working with a copy to avoid modifying the original
    data = data.copy()
    
    # Ensure all data types are correct
    try:
        # Make sure Close is numeric
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        # Drop any rows where Close is now NaN
        data = data.dropna(subset=['Close']).reset_index(drop=True)
        print(f"Cleaned data shape after dropping NaN Close values: {data.shape}")
    except Exception as e:
        print(f"Error converting Close to numeric: {str(e)}")
    
    # Ensure data is sorted by date
    if 'Date' in data.columns:
        data = data.sort_values('Date')
        print(f"Data sorted by Date. First date: {data['Date'].iloc[0]}, Last date: {data['Date'].iloc[-1]}")
    
    # Print data info
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data shape: {data.shape}")
    print(f"First few rows of Close column: {data['Close'].head().tolist()}")
    
    # Select only the 'Close' column for prediction
    # Make sure it's a proper 2D array
    close_data = data['Close'].values
    if len(close_data.shape) == 1:
        close_data = close_data.reshape(-1, 1)
    print(f"Close data shape: {close_data.shape}")
    
    # Scale the data
    print(f"Scaling data with scaler type: {type(scaler)}")
    scaled_data = scaler.transform(close_data)
    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"Scaled data range: min={scaled_data.min()}, max={scaled_data.max()}")
    
    # Create sequences for prediction
    x_predict = []
    for i in range(sequence_length, len(scaled_data)):
        # Handle both 1D and 2D cases for the scaled_data
        if len(scaled_data.shape) > 1 and scaled_data.shape[1] > 1:
            x_predict.append(scaled_data[i-sequence_length:i, 0])
        else:
            x_predict.append(scaled_data[i-sequence_length:i].flatten())
    
    # Convert to numpy array
    x_predict = np.array(x_predict)
    print(f"Prediction sequences created. Shape: {x_predict.shape}")
    
    # Reshape for LSTM [samples, time steps, features]
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 1))
    print(f"Reshaped for LSTM. Final shape: {x_predict.shape}")
    
    return x_predict, close_data

def make_predictions(data, future_days=30):
    """Make predictions using the model"""
    try:
        x_predict, close_data = prepare_data(data)
        
        # Make predictions with progress indication
        print(f"Making predictions with model on {x_predict.shape} input...")
        predictions = model.predict(x_predict, verbose=0)  # Set verbose=0 to reduce output noise
        print(f"Predictions shape from model: {predictions.shape}")
        
        # Ensure predictions is 2D for inverse_transform
        if len(predictions.shape) == 3:
            # If predictions is 3D, take the last dimension
            predictions = predictions.reshape(predictions.shape[0], -1)
            print(f"Reshaped predictions to 2D: {predictions.shape}")
        
        # Inverse transform to get original scale
        predictions_original = scaler.inverse_transform(predictions)
        print(f"Transformed predictions shape: {predictions_original.shape}")
        
        # Create dataframe for actual vs predicted
        actual_dates = data['Date'].values[60:]
        
        # Make sure close_data is properly flattened
        if len(close_data.shape) > 1:
            actual_prices = close_data[60:, 0]
        else:
            actual_prices = close_data[60:]
        
        predicted_prices = predictions_original.flatten()
        
        print(f"Actual dates shape: {actual_dates.shape}")
        print(f"Actual prices shape: {actual_prices.shape}")
        print(f"Predicted prices shape: {predicted_prices.shape}")
        
        # Ensure all arrays have the same length
        min_length = min(len(actual_dates), len(actual_prices), len(predicted_prices))
        actual_dates = actual_dates[:min_length]
        actual_prices = actual_prices[:min_length]
        predicted_prices = predicted_prices[:min_length]
        
        # Create a DataFrame with consistent string dates
        dates_list = [str(d) for d in actual_dates]  # Convert all dates to strings
        
        results = pd.DataFrame({
            'Date': dates_list,
            'Actual': actual_prices,
            'Predicted': predicted_prices
        })
        
        # Make future predictions
        print("Making future predictions...")
        last_sequence = x_predict[-1:]
        future_predictions = []
        
        for i in range(future_days):
            # Get prediction for next day with verbose=0 to reduce output noise
            next_pred = model.predict(last_sequence, verbose=0)
            
            # Handle various output shapes from model.predict
            if len(next_pred.shape) == 3:
                value = next_pred[0, 0, 0]
            elif len(next_pred.shape) == 2:
                value = next_pred[0, 0]
            else:
                value = next_pred[0]
                
            future_predictions.append(value)
            
            # Reshape the predicted value for appending to the sequence
            # Always make it a 3D array of shape (1, 1, 1)
            new_val = np.array([[[value]]])
            
            # Ensure correct dimensions for last_sequence
            if len(last_sequence.shape) != 3:
                print(f"Unexpected last_sequence shape: {last_sequence.shape}, reshaping...")
                last_sequence = last_sequence.reshape(1, last_sequence.shape[0], 1)
            
            # Update sequence - remove first timestep and add new value at the end
            last_sequence = np.append(last_sequence[:, 1:, :], new_val, axis=1)
        
        # Inverse transform future predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        print(f"Future predictions shape before transform: {future_predictions.shape}")
        future_predictions = scaler.inverse_transform(future_predictions)
        print(f"Future predictions shape after transform: {future_predictions.shape}")
    except Exception as e:
        print(f"Error in make_predictions: {str(e)}")
        # Create a minimal set of predictions to avoid breaking the app
        import datetime
        future_days = 30
        actual_dates = data['Date'].values[-10:]
        actual_prices = [0] * len(actual_dates)
        predicted_prices = [0] * len(actual_dates)
        
        results = pd.DataFrame({
            'Date': actual_dates,
            'Actual': actual_prices,
            'Predicted': predicted_prices
        })
        
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        date_range = pd.date_range(start=last_date, periods=future_days + 1)[1:]
        future_dates = date_range.strftime('%Y-%m-%d').tolist()
        future_predictions = np.zeros((future_days, 1))      # Generate future dates (using the same frequency as input data)
    try:
        # Make sure we have a valid last date
        try:
            last_date = pd.to_datetime(data['Date'].iloc[-1])
            print(f"Last date from data: {last_date}")
        except Exception as date_err:
            print(f"Error parsing last date from data: {str(date_err)}")
            # Use today as fallback
            import datetime
            last_date = datetime.datetime.now()
            print(f"Using current date as fallback: {last_date}")
        
        # Generate future dates
        date_range = pd.date_range(start=last_date, periods=future_days + 1)[1:]
        print(f"Generated future dates: {date_range}")
        
        # Convert to consistent string format for the DataFrame
        future_dates = date_range.strftime('%Y-%m-%d').tolist()
        
        future_results = pd.DataFrame({
            'Date': future_dates,
            'Predicted': future_predictions.flatten()
        })
    except Exception as e:
        print(f"Error generating future dates: {str(e)}")
        # More robust fallback with simple numbered days
        import datetime
        today = datetime.datetime.now()
        future_dates = []
        for i in range(1, future_days + 1):
            try:
                day = (today + datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            except:
                day = f"Day+{i}"
            future_dates.append(day)
        
        # Ensure the predictions array is the right length for the DataFrame
        pred_values = future_predictions.flatten()
        if len(pred_values) != len(future_dates):
            pred_values = np.zeros(len(future_dates))
        
        future_results = pd.DataFrame({
            'Date': future_dates,
            'Predicted': pred_values
        })
    
    return results, future_results

def create_plot(results, future_results):
    """Create plot of actual vs predicted prices and future predictions"""
    # Verify we're using non-interactive backend before creating figure
    import matplotlib
    if matplotlib.get_backend() != 'Agg':
        print("Switching to Agg backend...")
        matplotlib.use('Agg')
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    try:
        print("Creating plot with simple numeric x-axis")
        
        # Use simple numeric x-axis instead of date conversion to avoid issues
        x_historical = range(len(results))
        x_future = range(len(results), len(results) + len(future_results))
        
        # Plot actual vs predicted
        plt.plot(x_historical, results['Actual'], label='Actual Price', color='blue')
        plt.plot(x_historical, results['Predicted'], label='Predicted Price', color='red')
        
        # Plot future predictions
        plt.plot(x_future, future_results['Predicted'], label='Future Prediction', color='green', linestyle='--')
        
        # Set custom x-tick labels for dates
        # Select a reasonable number of ticks to avoid overcrowding
        n_ticks = min(10, len(results))
        
        if len(results) > 0:
            # Create indices for ticks
            tick_indices = [int(i * len(results)/n_ticks) for i in range(n_ticks)]
            
            # Add the last historical point
            if len(results) - 1 not in tick_indices:
                tick_indices.append(len(results) - 1)
                
            # Generate tick labels from dates (try both string and datetime formats)
            tick_labels = []
            for idx in tick_indices:
                if idx < len(results):
                    try:
                        date_val = results['Date'].iloc[idx]
                        # If it's already a string in a date format, use it
                        if isinstance(date_val, str) and len(date_val) >= 8:
                            tick_labels.append(date_val)
                        else:
                            # Try to parse as datetime and format
                            tick_labels.append(pd.to_datetime(date_val).strftime('%Y-%m-%d'))
                    except Exception as e:
                        # Fallback to index
                        tick_labels.append(f"#{idx}")
            
            # Add some future dates if available
            if len(future_results) > 0:
                future_ticks = [len(results), len(results) + len(future_results) - 1]  # First and last future point
                for idx in future_ticks:
                    future_idx = idx - len(results)
                    if future_idx >= 0 and future_idx < len(future_results):
                        try:
                            date_val = future_results['Date'].iloc[future_idx]
                            if isinstance(date_val, str) and len(date_val) >= 8:
                                tick_labels.append(date_val)
                            else:
                                tick_labels.append(pd.to_datetime(date_val).strftime('%Y-%m-%d'))
                        except:
                            tick_labels.append(f"F+{future_idx+1}")
                
                tick_indices.extend(future_ticks)
            
            # Set the ticks
            plt.xticks(tick_indices, tick_labels, rotation=45)
        
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
    except Exception as e:
        # Create a simple error plot if there's an issue
        print(f"Error in plotting: {str(e)}")
        plt.clf()  # Clear the figure
        plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                 ha='center', va='center', fontsize=12)
        plt.title('Error in Stock Price Prediction')
        plt.tight_layout()
    
    try:
        # Save plot to BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
        # Create a very simple error message image
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Error generating plot", ha='center', va='center')
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Ensure we close all figures to prevent memory leaks
    plt.close('all')
    
    return plot_data

def create_interactive_plot(results, future_results, stock_symbol="Stock"):
    """Create interactive plot using Plotly"""
    try:
        # Create the figure directly - cleaner than using subplots
        fig = go.Figure()
        
        # Add historical actual prices
        fig.add_trace(go.Scatter(
            x=results['Date'],
            y=results['Actual'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#38BDF8', width=2), # Sky blue for dark theme
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                         '<b>Actual Price:</b> %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add historical predicted prices
        fig.add_trace(go.Scatter(
            x=results['Date'],
            y=results['Predicted'],
            mode='lines',
            name='Predicted Price (Historical)',
            line=dict(color='#A78BFA', width=2), # Purple for dark theme
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                         '<b>Predicted Price:</b> %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add future predictions
        fig.add_trace(go.Scatter(
            x=future_results['Date'],
            y=future_results['Predicted'],
            mode='lines',
            name='Future Prediction',
            line=dict(color='#FB7185', width=2), # Rose/pink for dark theme
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                         '<b>Predicted Price:</b> %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout to match dark theme with elegant styling
        fig.update_layout(
            title={
                'text': f'{stock_symbol} - Stock Price Prediction',
                'font': {'size': 18, 'color': '#E8E9FA', 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title={'text': 'Date', 'font': {'size': 14, 'color': '#B8B9CA'}},
            yaxis_title={'text': 'Stock Price', 'font': {'size': 14, 'color': '#B8B9CA'}},
            hovermode='x unified',
            height=500,  # Reduced height for more compact display
            legend=dict(
                title={'text': "Close Price", 'font': {'size': 12, 'color': '#E8E9FA'}},
                y=0.99,
                x=1.0,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(18, 20, 28, 0.9)', 
                bordercolor='rgba(56, 189, 248, 0.3)',
                borderwidth=1,
                font={'color': '#E8E9FA', 'size': 11}
            ),
            margin=dict(l=50, r=50, t=60, b=80),  # Optimized margins
            plot_bgcolor='#0F1419',  # Dark blue-gray background
            paper_bgcolor='#12141C',  # Slightly lighter dark background
            font={'family': 'Inter, sans-serif', 'color': '#E8E9FA'}
        )
        
        # Configure X-axis to show years by default for main chart
        # Calculate date range
        try:
            first_date = pd.to_datetime(results['Date'].iloc[0]) if not results.empty else None
            last_pred_date = pd.to_datetime(future_results['Date'].iloc[-1]) if not future_results.empty else None
            
            if first_date and last_pred_date:
                date_range_days = (last_pred_date - first_date).days
                
                # For main chart, prefer yearly view for better overview
                if date_range_days <= 90:  # Three months or less
                    dtick_val = 'M1'  # Monthly ticks
                    tick_format = '%Y-%m'  # Year-Month format
                elif date_range_days <= 730:  # Up to 2 years
                    dtick_val = 'M3'  # Quarterly ticks
                    tick_format = '%Y-%m'  # Year-Month format
                else:
                    dtick_val = 'M12'  # Yearly ticks
                    tick_format = '%Y'  # Year only format
            else:
                dtick_val = 'M12'  # Default to yearly
                tick_format = '%Y'  # Year format
        except Exception:
            dtick_val = 'M12'  # Fallback to yearly on error
            tick_format = '%Y'  # Year format
            
        fig.update_xaxes(
            type='date',           # Explicitly set as date type
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(56, 189, 248, 0.2)',  # Light blue grid for dark theme
            tickangle=45,          # Angle for better readability
            dtick=dtick_val,       # Year-based tick interval
            tickformat=tick_format, # Year-based date format
            tickfont=dict(size=10, color='#B8B9CA'), # Light gray text for dark theme
            automargin=True,       # Automatically adjust margins to fit labels
            showticklabels=True,   # Make sure labels are visible
            nticks=12,             # Force a reasonable number of ticks
            title_font={'color': '#B8B9CA'}
        )
        
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(56, 189, 248, 0.2)',  # Light blue grid for dark theme
            tickfont=dict(color='#B8B9CA'),  # Light gray text for dark theme
            title_font={'color': '#B8B9CA'}
        )
        
        # Convert to HTML with improved config
        plot_html = plotly.io.to_html(
            fig, 
            include_plotlyjs='cdn', 
            full_html=False,
            config={
                'responsive': True, 
                'displayModeBar': True,
                'scrollZoom': True,         
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        )
        
        # Optimize the plot styling for immediate display without empty space
        plot_html = plot_html.replace('style="', 'style="width:100%; height:500px; margin:0; padding:0; ')
        
        # Add a container div with minimal spacing to ensure immediate visibility
        plot_html = f'<div class="plot-container" style="width:100%; max-width:100%; margin:0; padding:0;">{plot_html}</div>'
        
        return plot_html
        
    except Exception as e:
        print(f"Error creating interactive plot: {str(e)}")
        return f"<div class='alert alert-danger'>Error creating interactive plot: {str(e)}</div>"

def create_custom_interactive_plot(data, historical_days, prediction_days, stock_symbol="Stock", return_data=False):
    """Create interactive plot comparing last X days with next Y days prediction - following Google Colab approach
    
    If return_data=True, returns a tuple of (plot_html, dataframe) where dataframe contains the data used for plotting
    """
    try:
        print(f"Creating custom plot with {historical_days} historical days and {prediction_days} prediction days")
        
        # Make sure 'Date' is in datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"Data shape: {data.shape}, Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Sort data by date to ensure correct order
        data = data.sort_values('Date')
        
        # Prepare data for prediction - use all data to maintain consistency
        # Debug the available columns in case of confusion
        print(f"Available columns: {data.columns.tolist()}")
        
        # Debug the columns in data to better understand what we're working with
        print(f"DEBUG: Data columns before processing: {data.columns.tolist()}")
        
        # By this point, 'Close' column should be available from the preprocessing in custom_prediction endpoint
        # Double-check that we have the 'Close' column
        if 'Close' not in data.columns:
            print("WARNING: 'Close' column not found in data, looking for alternatives")
            
            # Check for lowercase 'close'
            if 'close' in data.columns:
                print("Found lowercase 'close' column, renaming it to 'Close'")
                data.rename(columns={'close': 'Close'}, inplace=True)
            # Check for 'Price' column
            elif 'Price' in data.columns:
                print("Using 'Price' column for prediction as 'Close' is not available")
                data.rename(columns={'Price': 'Close'}, inplace=True)
            else:
                # Last resort - try to find any price-related column
                columns_lower = [col.lower() for col in data.columns]
                potential_price_cols = [col for col, lower_col in zip(data.columns, columns_lower) 
                                       if lower_col in ['close', 'price', 'adj close', 'adjusted close']]
                if potential_price_cols:
                    col = potential_price_cols[0]
                    print(f"Using '{col}' as the price data")
                    data.rename(columns={col: 'Close'}, inplace=True)
                else:
                    raise ValueError(f"Could not find a suitable price column in the data. Available columns: {data.columns.tolist()}")
        else:
            print("'Close' column already exists in the data")
                
        print(f"DEBUG: Data columns after processing: {data.columns.tolist()}")
        print(f"DEBUG: First few values of 'Close': {data['Close'].head().tolist()}")
                
        # Use the 'Close' column for prediction
        close_data = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_data)
        print(f"Scaled data shape: {scaled_data.shape}, using standardized 'Close' column")
        
        # Generate predictions for future days using the same logic as main prediction
        sequence_length = 60
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        
        # Generate future predictions
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(prediction_days):
            next_pred = model.predict(current_sequence, verbose=0)
            
            # Extract the predicted value
            if len(next_pred.shape) == 3:
                value = next_pred[0, -1, 0]
            elif len(next_pred.shape) == 2:
                value = next_pred[0, 0]
            else:
                value = next_pred[0]
            
            future_predictions.append(value)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        np.array([[[value]]]), 
                                        axis=1)
        
        print(f"Generated {len(future_predictions)} future predictions")
        
        # Following Google Colab approach more precisely
        # Step 1: Create arrays for day indices (like the Google Colab example)
        sequence_length = 60  # This should match your model's requirements
        last_days = np.arange(1, historical_days + 1)
        day_pred = np.arange(historical_days + 1, historical_days + prediction_days + 1)
        print(f"Last days indices: {last_days}")
        print(f"Prediction days indices: {day_pred}")
        
        # Step 2: Create temp matrix with the correct dimensions
        total_length = historical_days + prediction_days + 1
        temp_mat = np.empty((total_length, 1))
        temp_mat[:] = np.nan
        temp_mat = temp_mat.reshape(1, -1).tolist()[0]
        
        # Step 3: Prepare data arrays exactly like in the Google Colab example
        last_original_days_value = temp_mat.copy()
        next_predicted_days_value = temp_mat.copy()
        
        # Step 4: Fill historical data aligned with current date
        # Convert data dates to datetime to ensure proper comparison
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Find data points to use for historical display (up to current date)
        # First, make sure we have data sorted by date
        data_sorted = data.sort_values('Date')
        
        # Find the most recent data row relative to today's date
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        mask = data_sorted['Date'] <= today
        
        if mask.any():
            # Get latest data index that is before or equal to today
            latest_idx = mask.values.argmax() if not mask.all() else len(mask) - 1
            
            # Take historical_days+1 points back from this latest date
            start_idx = max(0, latest_idx - historical_days)
            historical_data = data_sorted.iloc[start_idx:latest_idx+1]
            available_historical = len(historical_data)
            
            print(f"Using data from {historical_data['Date'].min()} to {historical_data['Date'].max()}")
            print(f"Available historical days: {available_historical} out of {historical_days + 1} requested")
            
            if available_historical > 0:
                # Get Close prices and scale if needed
                historical_prices = historical_data['Close'].values
                if len(historical_prices.shape) == 1:
                    historical_prices = historical_prices.reshape(-1, 1)
                
                historical_original = scaler.inverse_transform(
                    scaler.transform(historical_prices)
                ).flatten()
                
                # Fill the historical values aligned to the right (most recent at end)
                offset = max(0, (historical_days + 1) - available_historical)
                last_original_days_value[offset:offset+available_historical] = historical_original
                print(f"Filled {len(historical_original)} historical values starting at index {offset}")
            else:
                raise ValueError("No suitable historical data found")
        else:
            # Fallback: just use the last available data points
            available_historical = min(historical_days + 1, len(scaled_data))
            historical_scaled = scaled_data[-available_historical:]
            historical_original = scaler.inverse_transform(historical_scaled).flatten()
            last_original_days_value[0:available_historical] = historical_original
            print(f"Fallback: Filled {len(historical_original)} historical values (not date-aligned)")
        
        # Step 5: Fill future predictions (matching the Colab example)
        future_predictions_original = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()
        next_predicted_days_value[historical_days + 1:] = future_predictions_original
        print(f"Filled {len(future_predictions_original)} future prediction values")
        
        # Use current date as the reference point for the date range
        current_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        print(f"Current date: {current_date}")
        
        # Historical dates start from (current_date - historical_days) until current_date
        # Future dates start from tomorrow (current_date + 1 day) until (current_date + prediction_days)
        start_date = current_date - pd.Timedelta(days=historical_days)
        end_date = current_date + pd.Timedelta(days=prediction_days)
        
        # Create date range covering both historical and prediction days
        all_dates = pd.date_range(start=start_date, end=end_date)
        print(f"Created date range from {all_dates[0]} to {all_dates[-1]} for {len(all_dates)} periods")
        
        print(f"Created date range from {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} dates)")
        print(f"Historical values: {len([x for x in last_original_days_value if not pd.isna(x)])}")
        print(f"Future values: {len([x for x in next_predicted_days_value if not pd.isna(x)])}")
        
        # Create a DataFrame like in the Google Colab example
        from itertools import cycle
        
        new_pred_plot = pd.DataFrame({
            'last_original_days_value': last_original_days_value,
            'next_predicted_days_value': next_predicted_days_value,
            'date': all_dates
        })
        
        # Print the DataFrame head for debugging
        print("Created DataFrame with shape:", new_pred_plot.shape)
        print(new_pred_plot.head())
        
        # Use the Google Colab approach for creating the plot
        names = cycle([f'Last {historical_days} days close price', f'Predicted next {prediction_days} days close price'])
        
        # Create a descriptive title with date range
        today_str = current_date.strftime('%Y-%m-%d')
        past_date_str = (current_date - pd.Timedelta(days=historical_days)).strftime('%Y-%m-%d')
        future_date_str = (current_date + pd.Timedelta(days=prediction_days)).strftime('%Y-%m-%d')
        
        title = f"Stock Price: {past_date_str} to {today_str} (Historical) vs {today_str} to {future_date_str} (Prediction)"
        
        # Create the figure using go.Figure instead of px.line for better dark theme control
        fig = go.Figure()
        
        # Add historical data trace
        fig.add_trace(go.Scatter(
            x=new_pred_plot['date'],
            y=new_pred_plot['last_original_days_value'],
            mode='lines',
            name=f'Last {historical_days} days close price',
            line=dict(color='#38BDF8', width=2),  # Sky blue for dark theme
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add prediction data trace
        fig.add_trace(go.Scatter(
            x=new_pred_plot['date'],
            y=new_pred_plot['next_predicted_days_value'],
            mode='lines',
            name=f'Predicted next {prediction_days} days close price',
            line=dict(color='#FB7185', width=2, dash='dot'),  # Rose/pink with dotted line for dark theme
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add transition markers at the connection point between historical and predicted data
        # Calculate the indices where historical data ends and prediction begins
        historical_end_idx = historical_days
        prediction_start_idx = historical_days + 1
        
        # Add markers if we have valid values at these points
        if (not pd.isna(last_original_days_value[historical_end_idx]) and 
            not pd.isna(next_predicted_days_value[prediction_start_idx])):
            
            transition_dates = [all_dates[historical_end_idx], all_dates[prediction_start_idx]]
            transition_values = [
                last_original_days_value[historical_end_idx], 
                next_predicted_days_value[prediction_start_idx]
            ]
            
            # Add markers as additional trace
            fig.add_trace(go.Scatter(
                x=transition_dates,
                y=transition_values,
                mode='markers',
                marker=dict(size=8, color='rgba(168, 139, 250, 0.8)'),  # Purple markers for dark theme
                name='Transition Points',
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                             '<b>Price:</b> %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout with improved settings for better rendering - following Colab style more closely
        # Use current date to create a more accurate date-based title
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_str = today.strftime('%Y-%m-%d')
        past_date = today - datetime.timedelta(days=historical_days)
        past_date_str = past_date.strftime('%Y-%m-%d')
        future_date = today + datetime.timedelta(days=prediction_days)
        future_date_str = future_date.strftime('%Y-%m-%d')
        
        fig.update_layout(
            title={
                'text': f'Stock Price: {past_date_str} to {future_date_str}',
                'font': {'size': 16, 'color': '#E8E9FA', 'family': 'Inter, sans-serif'},
                'x': 0.5,  # Center the title
                'xanchor': 'center'
            },
            xaxis_title={'text': 'Date', 'font': {'size': 14, 'color': '#B8B9CA'}},
            yaxis_title={'text': 'Stock Price', 'font': {'size': 14, 'color': '#B8B9CA'}},
            hovermode='x unified',
            height=500,  # Reduced height for more compact display
            legend=dict(
                title={'text': 'Close Price', 'font': {'size': 12, 'color': '#E8E9FA'}},
                y=0.99,
                x=0.99,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(18, 20, 28, 0.9)', 
                bordercolor='rgba(56, 189, 248, 0.3)',
                borderwidth=1,
                orientation='v',
                font={'color': '#E8E9FA', 'size': 11}
            ),
            margin=dict(l=50, r=30, t=60, b=80),  # Optimized margins
            plot_bgcolor='#0F1419',  # Dark blue-gray background
            paper_bgcolor='#12141C',  # Slightly lighter dark background
            font={'family': 'Inter, sans-serif', 'color': '#E8E9FA'},
            autosize=True
        )
        
        # Ensure the X-axis is formatted as dates with more date labels
        # Determine appropriate date intervals based on the total range
        date_range_days = (future_date - past_date).days
        
        # Add more date ticks based on the range
        if date_range_days <= 14:
            # For short ranges (<= 14 days): show daily ticks
            tick_interval = 'D1'
        elif date_range_days <= 31:
            # For medium ranges (<= 1 month): show every other day
            tick_interval = 'D2' 
        elif date_range_days <= 60:
            # For two months: show every 4 days
            tick_interval = 'D4'
        else:
            # For longer ranges: show weekly ticks
            tick_interval = 'D7'
        
        # Configure x-axis with complete date labels (just one call to prevent overrides)
        fig.update_xaxes(
            type='date',
            tickformat='%Y-%m',  # Year-Month format
            tickangle=45,           # Angle the labels for better readability
            dtick='M1',             # Satu label per bulan
            rangeslider_visible=False,
            tickfont=dict(size=10, color='#B8B9CA'),  # Light gray text for dark theme
            automargin=True,        # Automatically adjust margins to fit labels
            showgrid=True,          # Show grid lines
            gridwidth=1,            # Width of grid lines
            gridcolor='rgba(56, 189, 248, 0.2)',  # Light blue grid for dark theme
            showticklabels=True,    # Make sure labels are visible
            title_font={'color': '#B8B9CA'}
        )
        
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(56, 189, 248, 0.2)',  # Light blue grid for dark theme
            tickfont=dict(color='#B8B9CA'),  # Light gray text for dark theme
            title_font={'color': '#B8B9CA'}
        )
        
        # Generate a unique ID for this plot
        import uuid
        plot_id = f"custom-plotly-{uuid.uuid4().hex[:8]}"
        
        # Convert to HTML with responsive settings and custom div ID
        plot_html = plotly.io.to_html(
            fig, 
            include_plotlyjs=True,  # Include the Plotly.js library to ensure it works consistently
            full_html=False,        # Don't generate a complete HTML document
            div_id=plot_id,         # Use the custom ID
            config={
                'responsive': True,         # Make it responsive
                'displayModeBar': True,     # Show the mode bar
                'showSendToCloud': False,   # Hide the cloud upload button
                'scrollZoom': True,         # Enable scroll to zoom
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'], # Remove unnecessary buttons
                'toImageButtonOptions': {    # Configure the download button
                    'format': 'png',
                    'filename': f'{stock_symbol}_prediction',
                    'scale': 2
                }
            }
        )
        
        # Optimize the plot styling for immediate display without empty space
        plot_html = plot_html.replace('style="', 'style="width:100%; height:500px; margin:0; padding:0; ')
        
        # Add a container div with minimal spacing to ensure immediate visibility
        plot_html = f'<div class="plot-container" style="width:100%; max-width:100%; margin:0; padding:0;">{plot_html}</div>'
        
        # Debug the HTML output size
        print(f"Generated custom plot HTML with {len(plot_html)} characters")
        print(f"First 100 chars of HTML: {plot_html[:100]}")
        print(f"Plot ID: {plot_id}")
        
        # Add debug information about data used for plotting
        print(f"DEBUG: Final plot data summary:")
        print(f"- Historical values: {len([x for x in last_original_days_value if not pd.isna(x)])}")
        print(f"- Prediction values: {len([x for x in next_predicted_days_value if not pd.isna(x)])}")
        print(f"- Date range: {all_dates[0]} to {all_dates[-1]}")
        print(f"- First few historical values: {[x for x in last_original_days_value[:5] if not pd.isna(x)]}")
        print(f"- Last few prediction values: {[x for x in next_predicted_days_value[-5:] if not pd.isna(x)]}")
        
        if return_data:
            # Return both the plot HTML and the dataframe used to create it
            return plot_html, new_pred_plot
        else:
            # Return just the plot HTML as before
            return plot_html
        
    except Exception as e:
        print(f"Error creating custom interactive plot: {str(e)}")
        import traceback
        print(traceback.format_exc())
        error_html = f"<div class='alert alert-danger'>Error creating custom interactive plot: {str(e)}</div>"
        
        if return_data:
            # Return error HTML and empty DataFrame
            return error_html, pd.DataFrame()
        else:
            # Return just error HTML
            return error_html

@app.route('/')
def landing():
    """Landing page route"""
    return render_template('landing.html')

@app.route('/app')
def index():
    """Dashboard/index page route"""
    # Get list of available models
    available_models = get_available_models()
    # Check if any model is loaded
    model_loaded = model is not None
    # Get active model name (if any)
    active_model = os.path.basename(MODEL_PATH) if model_loaded else None
    
    return render_template('index.html', 
                           model_loaded=model_loaded,
                           available_models=available_models,
                           active_model=active_model)

@app.route('/prediction')
def prediction_page():
    """Prediction results page route"""
    # This route will be used to display prediction results
    # For now, redirect to dashboard if accessed directly
    flash('Please upload a CSV file first to see predictions.')
    return redirect(url_for('index'))

@app.route('/lstm')
def lstm_info():
    """LSTM model information page route"""
    return render_template('lstm_info.html')

@app.route('/lstm-gru')
def lstm_gru_info():
    """LSTM+GRU model information page route"""
    return render_template('lstm_gru_info.html')

@app.route('/gru')
def gru_info():
    """GRU model information page route"""
    return render_template('gru_info.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store the CSV path in the global data
        global_data['last_csv_path'] = filepath
        
        try:
            # Check if model is loaded
            if model is None:
                flash("Please upload and load the LSTM model before uploading CSV data")
                return redirect('/app')
                
            # Load data
            stock_data = pd.read_csv(filepath)            # Special case: Check for non-standard CSV structures like the BBRI.csv file
            # where 'date' is in row 3 and values start from row 4
            
            print("Checking CSV structure...")
            print(f"CSV columns: {stock_data.columns.tolist()}")
            print(f"First few rows: \n{stock_data.head()}")
            
            # Check if this is the special BBRI format (where date is actually in row 3)
            date_in_row = None
            for row_idx in range(min(5, len(stock_data))):  # Check first 5 rows
                row_values = [str(val).lower() if isinstance(val, str) else "" for val in stock_data.iloc[row_idx].values]
                if 'date' in row_values:
                    date_in_row = row_idx
                    print(f"Found 'date' in row {row_idx+1}")
                    break
            
            if date_in_row is not None:
                print(f"Detected special CSV format with date in row {date_in_row+1}")
                # Look for 'date' and 'close' in the identified row
                date_index = None
                close_index = None
                
                for i, val in enumerate(stock_data.iloc[date_in_row].values):
                    val_str = str(val).lower() if not pd.isna(val) else ""
                    if val_str == 'date':
                        date_index = i
                    elif val_str == 'close':
                        close_index = i
                
                if date_index is not None:
                    # Extract actual data starting from the row after date
                    real_data = stock_data.iloc[date_in_row+1:].reset_index(drop=True)
                    
                    # Create a clean DataFrame with just Date and Close columns
                    clean_data = pd.DataFrame()
                    
                    # Extract date values
                    clean_data['Date'] = real_data.iloc[:, date_index]
                    print(f"Date column extracted: {clean_data['Date'].head().tolist()}")
                    
                    if close_index is not None:
                        # Extract close values
                        clean_data['Close'] = pd.to_numeric(real_data.iloc[:, close_index], errors='coerce')
                        print(f"Close column extracted: {clean_data['Close'].head().tolist()}")
                    else:
                        # Try to find a column that might contain price/close data
                        print("Could not find explicit 'close' column, looking for price data...")
                        # Check column names for price-related terms
                        price_col = None
                        for col_idx, col_name in enumerate(stock_data.columns):
                            if any(term in str(col_name).lower() for term in ['price', 'close', 'adj']):
                                price_col = col_idx
                                print(f"Found potential price column: {col_name}")
                                break
                        
                        if price_col is not None:
                            clean_data['Close'] = pd.to_numeric(real_data.iloc[:, price_col], errors='coerce')
                            print(f"Using column '{stock_data.columns[price_col]}' as Close. Values: {clean_data['Close'].head().tolist()}")
                        else:
                            flash("Could not find a suitable 'close' or price column in the CSV file")
                            return redirect('/app')
                    
                    # Drop rows with NaN values
                    clean_data = clean_data.dropna().reset_index(drop=True)
                    print(f"Final cleaned data shape: {clean_data.shape}")
                    
                    # Replace the stock_data with our reprocessed data
                    stock_data = clean_data
                    print("Successfully restructured the CSV data")
                else:
                    flash("Could not process the special CSV format correctly - no date column found")
                    return redirect('/app')
            else:
                # Normal CSV format processing with case-insensitive check
                required_cols = {'date': 'Date', 'close': 'Close'}
                columns_lower = [col.lower() for col in stock_data.columns]
                missing_cols = []
                
                # Check and rename columns if needed (case insensitive)
                for required_col, display_name in required_cols.items():
                    if required_col not in columns_lower:
                        missing_cols.append(display_name)
                    else:
                        # Get the actual column name from the CSV (with original case)
                        actual_col = stock_data.columns[columns_lower.index(required_col)]
                        # Rename to standard name if different
                        if actual_col != display_name:
                            stock_data.rename(columns={actual_col: display_name}, inplace=True)
                            print(f"Renamed column '{actual_col}' to '{display_name}'")
            
            # Check if we have the required columns after processing
            if 'Date' not in stock_data.columns or 'Close' not in stock_data.columns:
                flash(f"Required column(s) not found in CSV file: {', '.join(missing_cols)}")
                return redirect('/app')
              # Convert Date column to datetime
            try:
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                print("Successfully converted Date column to datetime")
            except Exception as e:
                print(f"Error converting Date column to datetime: {e}")
                # Try to fix common date formatting issues
                try:
                    # First make sure the date column contains strings
                    stock_data['Date'] = stock_data['Date'].astype(str)
                    # Remove any whitespace
                    stock_data['Date'] = stock_data['Date'].str.strip()
                    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
                    # Drop rows with invalid dates
                    stock_data = stock_data.dropna(subset=['Date']).reset_index(drop=True)
                    print("Date column fixed after handling formatting issues")
                except Exception as e2:
                    flash(f"Could not parse Date column: {str(e2)}")
                    return redirect('/app')
            
            # If we're using the default scaler, fit it on the data
            if not os.path.exists(SCALER_PATH):
                close_data = stock_data['Close'].values.reshape(-1, 1)
                scaler.fit(close_data)
                print("Default scaler fitted on data")
            
            # If we're using the default scaler, fit it on the data
            if not os.path.exists(SCALER_PATH):
                close_data = stock_data['Close'].values.reshape(-1, 1)
                scaler.fit(close_data)
            
            # Get stock symbol from filename (remove extension)
            stock_symbol = os.path.splitext(filename)[0]
            
            # Make predictions
            results, future_results = make_predictions(stock_data)
            
            # Create interactive plot with Plotly
            plot_html = create_interactive_plot(results, future_results, stock_symbol)
            
            # Prepare data for display with safer JSON handling
            import json
            
            # Convert to Python lists/dicts first, then JSON 
            # This helps avoid issues with pandas' to_json and Flask template escaping
            results_dict = results.to_dict(orient='records')
            future_dict = future_results.to_dict(orient='records')
            
            # Use json module for more control over serialization
            results_json = json.dumps(results_dict)
            future_json = json.dumps(future_dict)
            
            print(f"JSON Data prepared: {len(results_dict)} historical points, {len(future_dict)} future points")
            
            return render_template('prediction.html', 
                                  stock_symbol=stock_symbol,
                                  plot_html=plot_html,  # Use the interactive plot HTML
                                  results=results_json,
                                  future_results=future_json)
        
        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            return redirect('/')
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect('/')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    print("Upload model route called")
    
    # Get the selected model from the form
    selected_model = request.form.get('selected_model')
    print(f"Selected model: {selected_model}")
    
    # Load the selected model
    if selected_model:
        try:
            success = load_saved_model(selected_model)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'Model {selected_model} loaded successfully'
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': f'Error loading model {selected_model}'
                })
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return jsonify({
                'success': False, 
                'message': f'Error loading model: {str(e)}'
            })
    else:
        return jsonify({
            'success': False, 
            'message': 'No model selected'
        })
    
    if 'model_file' not in request.files:
        print("No model_file in request.files")
        flash('Missing model file')
        return redirect('/')
    
    model_file = request.files['model_file']
    print(f"Model file: {model_file.filename}")
    
    if model_file.filename == '':
        print("Empty model filename")
        flash('No model file selected')
        return redirect('/')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        # Save model file
        model_filepath = os.path.join('models', 'lstm.h5')
        print(f"Saving model to {model_filepath}")
        model_file.save(model_filepath)
        print(f"Model saved to {model_filepath}")
        
        # Check if scaler file is provided (optional)
        if 'scaler_file' in request.files and request.files['scaler_file'].filename != '':
            scaler_file = request.files['scaler_file']
            scaler_filepath = os.path.join('models', 'scaler.pkl')
            print(f"Saving scaler to {scaler_filepath}")
            scaler_file.save(scaler_filepath)
            print("Scaler file saved")
        else:
            # If scaler file doesn't exist, remove it to trigger using default scaler
            scaler_filepath = os.path.join('models', 'scaler.pkl')
            if os.path.exists(scaler_filepath):
                os.remove(scaler_filepath)
                print("Removed existing scaler file")
            print("No scaler file provided, will use default")
            
        # Try to load the model and scaler
        print("Attempting to load saved model...")
        success = load_saved_model()
        print(f"Model load success: {success}")
        
        if success:
            if 'scaler_file' in request.files and request.files['scaler_file'].filename != '':
                flash('Model and scaler uploaded successfully')
            else:
                flash('Model uploaded successfully (using default scaler)')
        else:
            flash('Error loading uploaded model')
        
    except Exception as e:
        print(f"Error in upload_model: {str(e)}")
        flash(f"Error uploading model: {str(e)}")
    
    return redirect('/')

@app.route('/custom_prediction', methods=['POST'])
def custom_prediction():
    """Handle custom prediction requests with dynamic historical and future days"""
    try:
        data = request.get_json()
        
        # Get parameters from request
        historical_days = int(data.get('historical_days', 15))
        prediction_days = int(data.get('prediction_days', 30))
        stock_symbol = data.get('stock_symbol', 'Stock')
        
        print(f"\n--- Custom Prediction Request ---")
        print(f"Historical days: {historical_days}, Prediction days: {prediction_days}")
        print(f"Stock symbol: {stock_symbol}")
        
        # Validate parameters
        if historical_days < 1:
            historical_days = 15  # Default
        
        if prediction_days < 1 or prediction_days > 90:
            prediction_days = 30  # Default, max 90 days (3 months)
        
        # Get the CSV file path from global data or use the most recent file
        if global_data.get('last_csv_path'):
            csv_path = global_data['last_csv_path']
            print(f"Using stored CSV path: {csv_path}")
        else:
            # Find the most recent CSV file in the uploads folder
            csv_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.csv'))
            if not csv_files:
                print("No CSV files found in uploads directory")
                return jsonify({
                    'success': False,
                    'message': 'No CSV file found. Please upload a CSV file first.'
                })
            # Sort by modification time (most recent first)
            csv_path = max(csv_files, key=os.path.getmtime)
            print(f"Using most recent CSV file: {csv_path}")
        
        # Load the CSV file
        stock_data = pd.read_csv(csv_path)
        print(f"Loaded CSV data with shape: {stock_data.shape}")
        print(f"CSV columns: {stock_data.columns.tolist()}")
        
        # Process the file similar to upload route
        try:
            # Check for special CSV formats (like the BBRI format)
            # Check if we need to skip some header rows
            if stock_data.shape[1] < 2 or 'date' not in ' '.join(stock_data.columns).lower():
                # Try to find the header row by checking the first few rows
                print("CSV format doesn't match expectations, checking for special format...")
                header_row = None
                for row_idx in range(min(10, len(stock_data))):
                    row_values = [str(val).lower() if isinstance(val, str) else "" for val in stock_data.iloc[row_idx].values]
                    if 'date' in row_values or 'time' in row_values:
                        header_row = row_idx
                        print(f"Found header row at index {header_row}")
                        break
                
                if header_row is not None:
                    # Reload the CSV skipping rows until the header
                    stock_data = pd.read_csv(csv_path, skiprows=header_row)
                    print(f"Reloaded CSV with header row {header_row}. New shape: {stock_data.shape}")
                    print(f"New columns: {stock_data.columns.tolist()}")
            
            # Look for Date and Close columns (case insensitive)
            columns_lower = [col.lower() for col in stock_data.columns]
            
            # Print all columns for debugging
            print(f"Original columns: {stock_data.columns.tolist()}")
            
            # Handle date column
            if 'date' in columns_lower:
                date_col = stock_data.columns[columns_lower.index('date')]
                stock_data.rename(columns={date_col: 'Date'}, inplace=True)
                print(f"Renamed '{date_col}' to 'Date'")
                
            # Handle different CSV formats - AAPL_sample.csv vs bbri.csv
            print("==== COLUMN HANDLING FOR CSV FILE ====")
            print(f"Original columns: {stock_data.columns.tolist()}")
            print(f"Lowercase columns: {columns_lower}")
            
            # Handle the date column first
            if 'Date' not in stock_data.columns and 'date' in columns_lower:
                date_col = stock_data.columns[columns_lower.index('date')]
                stock_data.rename(columns={date_col: 'Date'}, inplace=True)
                print(f"Renamed '{date_col}' to 'Date'")
            
            # Special handling for price columns
            # AAPL_sample.csv has 'Close' column
            # bbri.csv has both 'Price' and 'close' columns
            
            # First try for 'Close' with capital C (AAPL_sample.csv)
            if 'Close' in stock_data.columns:
                print(f"Found 'Close' column, using it directly")
            
            # For bbri.csv format:
            else:
                # CRITICAL FIX: In bbri.csv, 'close' is more accurate than 'Price'
                if 'close' in columns_lower:
                    close_col = stock_data.columns[columns_lower.index('close')]
                    stock_data.rename(columns={close_col: 'Close'}, inplace=True)
                    print(f"Renamed '{close_col}' to 'Close'")
                
                # If we still don't have 'Close', try 'Price'
                elif 'Price' in stock_data.columns or 'price' in columns_lower:
                    price_col = 'Price' if 'Price' in stock_data.columns else stock_data.columns[columns_lower.index('price')]
                    stock_data.rename(columns={price_col: 'Close'}, inplace=True)
                    print(f"Used '{price_col}' as Close column")
                
                # Last resort - try any column with price-related name
                else:
                    potential_cols = [col for col in stock_data.columns 
                                    if any(price_term in col.lower() 
                                          for price_term in ['price', 'close', 'value', 'adj'])]
                    
                    if potential_cols:
                        stock_data.rename(columns={potential_cols[0]: 'Close'}, inplace=True)
                        print(f"Last resort: Used '{potential_cols[0]}' as Close column")
                    else:
                        print("ERROR: Could not find a suitable price column")
                
            # Print columns after renaming
            print(f"Columns after renaming: {stock_data.columns.tolist()}")
            
            # Convert Date to datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
            before_dropna = len(stock_data)
            stock_data = stock_data.dropna(subset=['Date']).reset_index(drop=True)
            after_dropna = len(stock_data)
            if before_dropna != after_dropna:
                print(f"Dropped {before_dropna - after_dropna} rows with invalid dates")
            
            # Ensure Close is numeric
            stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
            before_dropna = len(stock_data)
            stock_data = stock_data.dropna(subset=['Close']).reset_index(drop=True)
            after_dropna = len(stock_data)
            if before_dropna != after_dropna:
                print(f"Dropped {before_dropna - after_dropna} rows with invalid prices")
            
            # Check we have enough data
            if len(stock_data) < 61:  # Need at least sequence_length + 1
                print(f"Not enough data for prediction: {len(stock_data)} rows")
                return jsonify({
                    'success': False,
                    'message': f'Not enough data for prediction. Need at least 61 rows, but got {len(stock_data)}.'
                })
            
            # Sort data by date to ensure correct order
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            print(f"Processed data shape: {stock_data.shape}")
            print(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
            print(f"Close price range: {stock_data['Close'].min()} to {stock_data['Close'].max()}")
            
            # Create custom interactive plot
            plot_html, plot_data = create_custom_interactive_plot(stock_data, historical_days, prediction_days, stock_symbol, return_data=True)
            
            # Check if the plot was generated successfully
            if plot_html.startswith("<div class='alert alert-danger'>"):
                print("Error in plot generation")
                return jsonify({
                    'success': False,
                    'message': plot_html
                })
            
            # Process the data from the plot to create consistent tables
            import datetime
            today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Extract historical and future data from the plot_data
            # plot_data contains the DataFrame used to create the plot with columns:
            # 'date', 'last_original_days_value', 'next_predicted_days_value'
            
            # Create historical prediction data using actual data from plot
            historical_results = []
            
            # Get historical data points (non-NaN values in last_original_days_value)
            historical_data_points = plot_data[~plot_data['last_original_days_value'].isna()]
            
            if len(historical_data_points) > 0:
                print(f"Using {len(historical_data_points)} historical data points from plot")
                
                # Filter actual data for the date range in historical_data_points
                min_date = historical_data_points['date'].min()
                max_date = historical_data_points['date'].max()
                
                # Get actual data from stock_data for the same date range
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                actual_data = stock_data[(stock_data['Date'] >= min_date) & 
                                        (stock_data['Date'] <= max_date)]
                
                # Create a date-indexed dictionary of actual prices for quick lookup
                actual_prices_dict = {}
                for _, row in actual_data.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d')
                    actual_prices_dict[date_str] = float(row['Close'])
                
                # Create historical results with actual and predicted values
                for _, row in historical_data_points.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    predicted_price = float(row['last_original_days_value'])
                    
                    # Try to find actual price for this date
                    actual_price = actual_prices_dict.get(date_str)
                    
                    # If we have actual price, calculate difference and accuracy
                    if actual_price is not None:
                        diff = predicted_price - actual_price
                        accuracy = 100 - abs(diff / actual_price * 100)
                    else:
                        # If no actual price is available, use prediction as actual (will show 100% accuracy)
                        actual_price = predicted_price
                        diff = 0.0
                        accuracy = 100.0
                    
                    historical_results.append({
                        'Date': date_str,
                        'Actual': float(actual_price),
                        'Predicted': predicted_price,
                        'Difference': float(diff),
                        'Accuracy': float(accuracy)
                    })
            else:
                print("No historical data points found in plot data")
            
            # Create future prediction data from plot next_predicted_days_value
            future_results = []
            
            # Get future data points (non-NaN values in next_predicted_days_value)
            future_data_points = plot_data[~plot_data['next_predicted_days_value'].isna()]
            
            if len(future_data_points) > 0:
                print(f"Using {len(future_data_points)} future data points from plot")
                
                for _, row in future_data_points.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    predicted_price = float(row['next_predicted_days_value'])
                    
                    future_results.append({
                        'Date': date_str,
                        'Predicted': predicted_price
                    })
            else:
                print("No future data points found in plot data")
            
            print("Custom plot and prediction tables created successfully")
            
            # Make sure we're not sending empty HTML
            if not plot_html or len(plot_html.strip()) == 0:
                print("Warning: Empty plot HTML generated")
                return jsonify({
                    'success': False,
                    'message': 'Error: Failed to generate plot (empty HTML returned)'
                })
            
            # Add date information to the plot HTML content
            import datetime
            today = datetime.datetime.now()
            today_str = today.strftime('%Y-%m-%d')
            past_date = today - datetime.timedelta(days=historical_days)
            past_date_str = past_date.strftime('%Y-%m-%d')
            future_date = today + datetime.timedelta(days=prediction_days)
            future_date_str = future_date.strftime('%Y-%m-%d')
            
            # Verify data format for the tables - more detailed debug info
            print(f"Historical data sample: {historical_results[0] if historical_results else 'None'}")
            print(f"Future data sample: {future_results[0] if future_results else 'None'}")
            
            # Show some sample rows from the plot data
            print("\nPlot data sample (first 3 rows):")
            print(plot_data.head(3).to_string())
            
            # Verify numbers in historical and future tables match the plot
            if historical_results and future_results:
                print("\nVerifying data consistency:")
                print(f"Plot's first historical date: {plot_data[~plot_data['last_original_days_value'].isna()].iloc[0]['date'].strftime('%Y-%m-%d')}")
                print(f"Plot's first historical value: {plot_data[~plot_data['last_original_days_value'].isna()].iloc[0]['last_original_days_value']:.2f}")
                print(f"Table's first historical date: {historical_results[0]['Date']}")
                print(f"Table's first historical value (predicted): {historical_results[0]['Predicted']:.2f}")
                
                print(f"Plot's first future date: {plot_data[~plot_data['next_predicted_days_value'].isna()].iloc[0]['date'].strftime('%Y-%m-%d')}")
                print(f"Plot's first future value: {plot_data[~plot_data['next_predicted_days_value'].isna()].iloc[0]['next_predicted_days_value']:.2f}")
                print(f"Table's first future date: {future_results[0]['Date']}")
                print(f"Table's first future value: {future_results[0]['Predicted']:.2f}")
            
            # Add date range header before the plot
            date_header = f'''
            <div class="date-range-info mb-3">
                <h4 class="text-center">Stock Price Prediction: {past_date_str} to {future_date_str}</h4>
                <p class="text-center text-muted">Historical data from {past_date_str} to {today_str}, predictions from {today_str} to {future_date_str}</p>
            </div>
            '''
            
            # Insert the date header before the plot
            plot_html = date_header + plot_html
            
            # Include extra debugging information in the response
            print(f"SUCCESS: Plot HTML generated with {len(plot_html)} characters")
            print(f"HTML begins with: {plot_html[:100] if len(plot_html) > 0 else 'EMPTY'}")
            print(f"Historical data: {len(historical_results)} rows, Future data: {len(future_results)} rows")
            
            return jsonify({
                'success': True,
                'plot_html': plot_html,
                'html_length': len(plot_html),
                'message': f'Custom plot created from {past_date_str} to {future_date_str}',
                'date_info': {
                    'today': today_str,
                    'past_date': past_date_str,
                    'future_date': future_date_str
                },
                # Add table data
                'historical_data': historical_results,
                'future_data': future_results,
                'historical_days': historical_days,
                'prediction_days': prediction_days
            })
            
        except Exception as e:
            print(f"ERROR processing data: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            print(trace)
            return jsonify({
                'success': False,
                'message': f'Error processing data: {str(e)}',
                'traceback': trace
            })
    
    except Exception as e:
        print(f"Server error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        })

if __name__ == '__main__':
    # Try to load model on startup
    try:
        load_saved_model()
        print("Model loaded successfully on startup.")
    except Exception as e:
        print(f"Warning: Could not load model on startup. Error: {str(e)}")
    
    # Clear port information before starting
    print("\n" + "="*50)
    print("Starting Flask server...")
    print("="*50)
    
    # Run the Flask app with specific host and debug information
    try:
        print("Server will be accessible at: http://127.0.0.1:5000")
        print("Press CTRL+C to quit the server")
        print("="*50 + "\n")
        app.run(debug=True, port=5000, host='127.0.0.1')
    except Exception as e:
        print(f"Error starting Flask server: {str(e)}")
        
        # Try fallback port if 5000 is already in use
        try:
            print("\nTrying alternative port 5001...")
            print("Server will be accessible at: http://127.0.0.1:5001")
            app.run(debug=True, port=5001, host='127.0.0.1')
        except Exception as e2:
            print(f"Error starting Flask server on fallback port: {str(e2)}")
            print("Please check if another application is using these ports.")
