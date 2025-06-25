import os
import numpy as np
import pandas as pd
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
from io import BytesIO
import base64
import glob
import re

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "saham-prediction-app"

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

@app.route('/')
def index():
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
        
        try:
            # Check if model is loaded
            if model is None:
                flash("Please upload and load the LSTM model before uploading CSV data")
                return redirect('/')
                
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
                            return redirect('/')
                    
                    # Drop rows with NaN values
                    clean_data = clean_data.dropna().reset_index(drop=True)
                    print(f"Final cleaned data shape: {clean_data.shape}")
                    
                    # Replace the stock_data with our reprocessed data
                    stock_data = clean_data
                    print("Successfully restructured the CSV data")
                else:
                    flash("Could not process the special CSV format correctly - no date column found")
                    return redirect('/')
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
                return redirect('/')
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
                    return redirect('/')
            
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
            
            # Create plot
            plot_data = create_plot(results, future_results)
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
                                  plot_data=plot_data,
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

if __name__ == '__main__':
    # Try to load model on startup
    load_saved_model()
    app.run(debug=True, port=5000)
