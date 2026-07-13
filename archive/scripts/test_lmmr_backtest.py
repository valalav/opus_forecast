"""
Script for backtesting the LMMR Claude model
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sirena.models.lmmr_claude import LMMRForecasterClaude
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_lmmr_backtest():
    """
    Run a backtest for the LMMR Claude model
    """
    print("Loading data...")
    
    # Load inflation data
    try:
        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        print(f"Raw data loaded: {len(df_raw)} rows")

        # Clean the MoM column - convert to numeric, replacing commas with dots if needed
        df_raw['MoM'] = pd.to_numeric(df_raw['MoM'].astype(str).str.replace(',', '.'), errors='coerce')

        # Pivot the data to have products as columns
        df = df_raw.pivot(index='Date', columns='Товар', values='MoM')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        print(f"Data after pivoting: {len(df)} rows")

        # Get the main target column
        target_col = 'Все товары и услуги'

        print(f"Target column: {target_col}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Available columns: {list(df.columns)}")

        # Load additional data that might be needed for exogenous variables
        try:
            # Load inflation_data.csv to get USD and other factors
            additional_data = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
            additional_data['Date'] = pd.to_datetime(additional_data['Date'], format='%d.%m.%Y', errors='coerce')

            # Convert columns to numeric where needed
            for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real']:
                if col in additional_data.columns:
                    additional_data[col] = pd.to_numeric(additional_data[col], errors='coerce')

            # Align the date formats for joining: convert both to month start
            inflation_index = df.index.to_period('M').to_timestamp()
            df.index = inflation_index
            # Create a new column with period-converted dates, then set as index
            additional_data['period_date'] = additional_data['Date'].dt.to_period('M').dt.to_timestamp()
            additional_data.set_index('period_date', inplace=True)

            # Add relevant columns if they exist, joining on aligned dates
            exog_cols = []
            for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real']:
                if col in additional_data.columns:
                    exog_cols.append(col)
                    # Join the column
                    df = df.join(pd.DataFrame(additional_data[col]), how='left')

            print(f"Successfully joined exogenous columns: {exog_cols}")

        except FileNotFoundError:
            print("Additional data file not found, proceeding with basic data")

    except FileNotFoundError:
        print("Data file not found. Creating synthetic data for testing...")
        # Create synthetic data if real data is not available
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)

        # Simulate realistic inflation data
        trend = np.linspace(100, 105, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.5, n)
        y_values = trend + seasonal + noise

        # Convert to MoM format
        mom_values = [y_values[0]]
        for i in range(1, n):
            mom_values.append((y_values[i]/y_values[i-1])*100)

        # Add USD and Brent as external factors
        usd_values = 60 + np.cumsum(np.random.normal(0, 0.5, n))
        brent_values = 50 + np.cumsum(np.random.normal(0, 1, n))

        df = pd.DataFrame({
            'Все товары и услуги': mom_values,
            'usd_nom_i': usd_values,
            'brent': brent_values
        }, index=dates)

        target_col = 'Все товары и услуги'

    print("Starting backtest...")
    
    # Check which dates have all required data (target + at least some exogenous variables)
    required_cols = [target_col]
    
    # Add exogenous columns that exist in the df
    exog_cols = []
    for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real']:
        if col in df.columns:
            exog_cols.append(col)
            required_cols.append(col)
    
    print(f"Required exogenous columns: {exog_cols}")
    
    # Keep only dates that have all required columns (non-NaN)
    complete_data = df.dropna(subset=required_cols)
    print(f"Complete data points (with all required columns): {len(complete_data)}")
    
    # Define backtest parameters - start after we have sufficient data
    start_date = '2015-01-01'  # Start from 2015 to have more complete data
    test_dates = complete_data.index[complete_data.index >= pd.Timestamp(start_date)]
    
    if len(test_dates) < 12:
        print("Not enough complete data for backtesting. Using available data.")
        if len(complete_data) < 12:
            print("Not enough data points overall. Using last 12 points where available.")
            test_dates = complete_data.index[-12:] if len(complete_data) >= 12 else complete_data.index
        else:
            test_dates = complete_data.index[-12:]  # Use last 12 months if not enough data from start_date
    
    print(f"Backtest period: {test_dates[0]} to {test_dates[-1]}")
    print(f"Number of potential test points: {len(test_dates)}")
    
    results = []
    
    for target_date in test_dates:
        # Prepare training data (all data before target date)
        train_df = complete_data[complete_data.index < target_date].copy()
        
        # Skip if not enough training data
        if len(train_df) < 48:  # Minimum required for STL
            print(f"Skipping {target_date}, insufficient training data: {len(train_df)}")
            continue
        
        try:
            # Create and train model
            model = LMMRForecasterClaude(alpha=0.5)
            model.fit(train_df, target_col)
            
            # Prepare test data
            test_df = df[df.index <= target_date].copy()
            
            # Make prediction
            pred = model.predict(test_df, target_date)
            prediction = pred['prediction']
            
            # Get actual value
            actual = df.loc[target_date, target_col]
            
            # Calculate error
            error = actual - prediction
            
            results.append({
                'date': target_date,
                'actual': actual,
                'prediction': prediction,
                'error': error,
                'abs_error': abs(error)
            })
            
            print(f"Date: {target_date.strftime('%Y-%m')}, Actual: {actual:.3f}, Predicted: {prediction:.3f}, Error: {error:.3f}")
            
        except Exception as e:
            print(f"Error predicting for {target_date}: {e}")
            continue
    
    if not results:
        print("No results generated from backtest.")
        print("This could be due to:")
        print("1. Not enough complete data points (dates with all required columns)")
        print("2. Insufficient training data for the model (need at least 48 months)")
        print("3. Issues with seasonal decomposition (STL needs sufficient data)")
        return
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    mae = results_df['abs_error'].mean()
    rmse = np.sqrt((results_df['error'] ** 2).mean())
    mape = (results_df['abs_error'] / results_df['actual'].abs() * 100).mean()
    
    # Direction accuracy
    actual_direction = np.sign(results_df['actual'] - 100)
    pred_direction = np.sign(results_df['prediction'] - 100)
    direction_accuracy = (actual_direction == pred_direction).mean()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Number of predictions: {len(results_df)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print(f"Mean Actual Value: {results_df['actual'].mean():.3f}")
    print(f"Mean Predicted Value: {results_df['prediction'].mean():.3f}")
    print(f"Mean Absolute Error: {results_df['abs_error'].mean():.3f}")
    print(f"Max Error: {results_df['abs_error'].max():.3f}")
    print(f"Min Error: {results_df['abs_error'].min():.3f}")
    
    # Show first few results
    print(f"\nFirst 5 results:")
    print(results_df[['date', 'actual', 'prediction', 'error', 'abs_error']].head())
    
    # Show correlation
    correlation = results_df['actual'].corr(results_df['prediction'])
    print(f"\nCorrelation between actual and predicted: {correlation:.4f}")
    
    return results_df

def run_simple_test():
    """
    Run a simple test with synthetic data to ensure the model works
    """
    print("\nRunning simple test with synthetic data...")
    
    # Create synthetic data
    dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
    n = len(dates)

    # Simulate realistic inflation data
    trend = np.linspace(100, 105, n)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 0.5, n)
    y_values = trend + seasonal + noise

    # Convert to MoM format
    mom_values = [y_values[0]]
    for i in range(1, n):
        mom_values.append((y_values[i]/y_values[i-1])*100)

    # Add USD and other factors as external factors
    usd_values = 60 + np.cumsum(np.random.normal(0, 0.5, n))
    brent_values = 50 + np.cumsum(np.random.normal(0, 1, n))
    real_income_values = 100 + np.cumsum(np.random.normal(0, 0.2, n))

    df = pd.DataFrame({
        'Все товары и услуги': mom_values,
        'usd_nom_i': usd_values,
        'brent': brent_values,
        'all_real': real_income_values
    }, index=dates)

    target_col = 'Все товары и услуги'
    
    print(f"Created synthetic dataset with {len(df)} rows")
    
    # Test the model with a single prediction
    try:
        train_df = df[df.index < '2024-01-01'].copy()
        test_date = pd.Timestamp('2024-01-01')
        test_df = df[df.index <= test_date].copy()
        
        print(f"Training data points: {len(train_df)}")
        print(f"Testing on date: {test_date}")
        
        model = LMMRForecasterClaude(alpha=0.5)
        model.fit(train_df, target_col)
        
        prediction = model.predict(test_df, test_date)
        actual = df.loc[test_date, target_col]
        
        print(f"Single prediction test:")
        print(f"  Actual: {actual:.3f}")
        print(f"  Predicted: {prediction['prediction']:.3f}")
        print(f"  Error: {actual - prediction['prediction']:.3f}")
        print(f"  Model successfully made a prediction!")
        
    except Exception as e:
        print(f"Error in simple test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    results = run_lmmr_backtest()
    run_simple_test()