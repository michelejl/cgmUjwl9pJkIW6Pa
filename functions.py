import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def preprocess_dataframe(df):
    """
    Preprocess a DataFrame by converting 'Date' to datetime, 'Price' to numeric,
    dropping rows with NaNs in these columns, sorting by 'Date', and resetting the index.
    
    Parameters:
    df (pd.DataFrame): DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Convert 'Price' column to numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    # Drop rows where the 'Date' or 'Price' conversion failed
    df = df.dropna(subset=['Date', 'Price'])
    # Drop rows where 'Vol.' is '-'
    df = df[df['Vol.'] != '-']
    # Replace 'M' and 'K', convert to numeric, and multiply accordingly
    df['Volume (M)'] = df['Vol.'].str.replace('M', '').str.replace('K', 'e-3').str.replace('[^\d.-]', '').astype(float)
    # Multiply values by 0.001 where the letter is 'K'
    df.loc[df['Vol.'].str.endswith('K'), 'Volume (M)'] *= 0.001
    # Drop the original 'Vol.' column
    df.drop(columns=['Vol.'], inplace=True)
    # Adding numeric Year & Month columns
    df['Year'] = df['Date'].dt.year.astype(int)
    df['Month'] = df['Date'].dt.month.astype(int)
    # Sort the DataFrame by the 'Date' column
    df = df.sort_values('Date')
    # Reset the index
    df = df.reset_index(drop=True)
    
    return df

# Data Visualization
def plot_stock_data(df, country_name):
    """
    Plot 'Price', 'High', and 'Low' data for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Date', 'Price', 'High', and 'Low' columns.
    country_name (str): Name of the country for labeling the plot.

    Returns:
    None
    """
    # Create the figure and plot the data
    plt.figure(figsize=(18, 9))
    plt.plot(df['Date'], df['Price'], label='Price')
    plt.plot(df['Date'], df['High'], label='High')
    plt.plot(df['Date'], df['Low'], label='Low')

    # Label the axes and the plot
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.title(f'Stock Price - {country_name}')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()

#Candlestick chart
def candlestick_volume(df, dataset_name):
    """
    Create and display a candlestick chart with volume from a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Date', 'Open', 'High', 'Low', 'Price', and 'Volume (M)' columns.
    title (str): The title of the chart.

    Returns:
    None
    """
    # Convert 'Date' column to datetime if it isn't already
    if df['Date'].dtype != '<M8[ns]':
        df['Date'] = pd.to_datetime(df['Date'])

    # Calculate Bollinger Bands
    df['20_SMA'] = df['Price'].rolling(window=20).mean()
    df['20_STD'] = df['Price'].rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Create subplots for candlestick and volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Price'],
        name='Candlestick',
        showlegend=False  # Hide the candlestick from the legend
    ), row=1, col=1)

    # Add Bollinger Bands to the chart
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Upper_Band'],
        line=dict(color='blue', width=1),
        name='Upper Bollinger Band',
        mode='lines'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Lower_Band'],
        line=dict(color='blue', width=1),
        name='Lower Bollinger Band',
        mode='lines'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['20_SMA'],
        line=dict(color='orange', width=1),
        name='20 SMA',
        mode='lines'
    ), row=1, col=1)

    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume (M)'],
        name='Volume',
        marker=dict(color='gray')
    ), row=2, col=1)

    # Add titles and labels
    fig.update_layout(
        title=f'Candlestick Chart with Bollinger Bands and Volume - {dataset_name}',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='black',   # Set plot background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font=dict(color='white'),  # Set font color to white
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            tickfont=dict(color='white')
        )
    )

    fig.update_xaxes(showgrid=True, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridcolor='gray')

    # Show the chart
    fig.show()

#Draw Box Plots
def draw_boxplots(df, country_name):
    """
    Draw year-wise and month-wise box plots for the 'Price' column.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'Date', 'Price', 'Year', and 'Month' columns.
    
    Returns:
    None
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=80)
    
    # Year-wise box plot
    sns.boxplot(x='Year', y='Price', data=df, ax=axes[0])
    axes[0].set_title(f'Yearly Variation/Trend - {country_name}', fontsize=18)
    
    # Month-wise box plot
    sns.boxplot(x='Month', y='Price', data=df, ax=axes[1])
    axes[1].set_title(f'Monthly Variation - {country_name}', fontsize=18)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Linear Regression
# Initialize a DataFrame to store the results
def linear_regression_forecast(data, dataset_name):
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Ensure 'Price' is integer
    df['Price'] = df['Price'].astype(int)

    # Create lagged features
    n_lags = 5
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Price'].shift(lag)

    # Create lead features
    leads = [1, 7, 30]
    for lead in leads:
        df[f'lead_{lead}'] = df['Price'].shift(-lead)

    df.rename(columns={'Price': 'lag_0'}, inplace=True)
    df.dropna(inplace=True)

    # Split the data into training and testing sets
    train = df[df.index.year == 2020].copy()
    test = df[df.index.year == 2021].copy()

    y_cols = [f'lead_{j}' for j in leads]
    x_cols = [f'lag_{j}' for j in range(1, n_lags + 1)]

    # Train the Linear Regression model
    model = LinearRegression().fit(train[x_cols].values, train[y_cols])

    # Initialize a DataFrame to store the test predictions
    test_predictions_df = pd.DataFrame(index=test.index, columns=y_cols)

    # Iteratively predict one day at a time
    for date in test.index:
        if test.loc[date, x_cols].isna().any():
            # Use last n_lags rows from the train set to initialize the first prediction
            lagged_features = train[x_cols].iloc[-1].values.reshape(1, -1)
        else:
            lagged_features = test.loc[date, x_cols].values.reshape(1, -1)

        # Predict the next day's prices
        predicted_values = model.predict(lagged_features)[0]

        # Store the predictions
        test_predictions_df.loc[date] = predicted_values

        # Update the lagged features in the test set
        for lag in range(n_lags, 0, -1):
            if lag == 1:
                if (date + pd.Timedelta(days=1)) in test.index:
                    test.loc[date + pd.Timedelta(days=1), f'lag_{lag}'] = predicted_values[0]
            else:
                test[f'lag_{lag}'] = test[f'lag_{lag-1}'].shift(-1)

        # Update the test set's lag_0 value with the predicted value
        if (date + pd.Timedelta(days=1)) in test.index:
            test.loc[date + pd.Timedelta(days=1), 'lag_0'] = predicted_values[0]

    # Calculate Bollinger Bands for train and test sets
    df['20_SMA'] = df['lag_0'].rolling(window=20).mean()
    df['20_STD'] = df['lag_0'].rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Plot the results for each lead time
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['lag_0'], label='Train Price')
    plt.plot(test.index, test['lag_0'], label='Test Price')
    for lead in leads:
        plt.plot(test_predictions_df.index, test_predictions_df[f'lead_{lead}'], label=f'Predicted Price + {lead} day(s)', linestyle='--')

    # Plot Bollinger Bands for train and test sets
    plt.plot(df.index, df['20_SMA'], label='20 SMA', color='orange')
    plt.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='blue', alpha=0.1)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Linear Regression Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Evaluate the model
    for lead in leads:
        rmse = np.sqrt(mean_squared_error(test[f'lead_{lead}'], test_predictions_df[f'lead_{lead}']))
        mape = mean_absolute_percentage_error(test[f'lead_{lead}'], test_predictions_df[f'lead_{lead}'])
        print(f'Lead {lead} day(s):')
        print(f'Root Mean Squared Error: {rmse}')
        print(f'Mean Absolute Percentage Error: {mape * 100}%')
        print('')

    return test_predictions_df


#ARIMA
# Convert 'Date' column to datetime if it isn't already
def arima(df, dataset_name):
    if df['Date'].dtype != '<M8[ns]':
        df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Normalize the data
    columns_to_normalize = ['Price']
    scaler = MinMaxScaler()

    # Fit the scaler on the data and transform
    df_scaled = df.copy()
    df_scaled[columns_to_normalize] = scaler.fit_transform(df_scaled[columns_to_normalize])

    # Split the data into training and testing sets
    train_df = df_scaled[df_scaled.index.year == 2020].copy()
    test_df = df_scaled[df_scaled.index.year == 2021].copy()

    # Check for stationarity
    result = adfuller(train_df['Price'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    # If p-value > 0.05, data is non-stationary, apply differencing
    if result[1] > 0.05:
        train_df['Price_diff'] = train_df['Price'].diff().dropna()
    else:
        train_df['Price_diff'] = train_df['Price']

    # Find the optimal parameters using auto_arima
    stepwise_model = auto_arima(train_df['Price_diff'].dropna(), start_p=1, start_q=1,
                                max_p=5, max_q=5, seasonal=False,
                                d=1, trace=True, error_action='ignore', suppress_warnings=True)
    print(stepwise_model.summary())

    # Fit the ARIMA model with the optimal parameters
    p, d, q = stepwise_model.order
    arima_model = ARIMA(train_df['Price'], order=(p, d, q))
    arima_model = arima_model.fit()

    # Forecasting
    forecast = arima_model.get_forecast(steps=len(test_df))

    # Forecast the future values (for the length of the test set)
    forecast_mean = forecast.predicted_mean
    
    # Create a combined dataframe for plotting
    train_df['Forecast'] = None
    test_df['Forecast'] = forecast_mean.values

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Price'], label='Train Price')
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, test_df['Forecast'], label='Forecast', color='red')

    # Calculate Bollinger Bands for train and test sets
    df_scaled['20_SMA'] = df_scaled['Price'].rolling(window=20).mean()
    df_scaled['20_STD'] = df_scaled['Price'].rolling(window=20).std()
    df_scaled['Upper_Band'] = df_scaled['20_SMA'] + (df_scaled['20_STD'] * 2)
    df_scaled['Lower_Band'] = df_scaled['20_SMA'] - (df_scaled['20_STD'] * 2)

    # Plot Bollinger Bands for train and test sets
    plt.plot(df_scaled.index, df_scaled['20_SMA'], label='20 SMA', color='orange')
    plt.fill_between(df_scaled.index, df_scaled['Upper_Band'], df_scaled['Lower_Band'], color='blue', alpha=0.1)
        
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'ARIMA Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_df['Price'], test_df['Forecast']))

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_df['Price'], test_df['Forecast'])

    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    return rmse, mape

##SARIMAX
def sarimax_forecast(df, dataset_name):
    """
    Perform SARIMAX modeling, forecasting, and evaluation on the given training and testing DataFrames.
    
    Parameters:
    train_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for training.
    test_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for testing.
    
    Returns:
    float: RMSE (Root Mean Squared Error) of the forecast.
    float: MAPE (Mean Absolute Percentage Error) of the forecast.
    """
    train_df = df[df['Year'] == 2020]
    test_df = df[df['Year'] == 2021]

    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)

    # Ensure all numeric columns are correctly cast to numeric types
    numeric_columns = train_df.columns.difference(['Date'])
    train_df[numeric_columns] = train_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    test_df[numeric_columns] = test_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill missing values with forward fill method
    train_df = train_df.fillna(method='ffill')
    test_df = test_df.fillna(method='ffill')

    # Drop remaining rows with missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Check if the train and test DataFrames are not empty
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test DataFrame is empty after preprocessing.")

    # Select exogenous features, excluding 'Price' and 'Date'
    exogenous_features = train_df.columns.difference(['Price', 'Date', 'Year', 'Month']).tolist()

    # Use auto_arima to find the optimal parameters
    auto_model = auto_arima(train_df['Price'], 
                            seasonal=True, exogenous=train_df[exogenous_features],
                            stepwise=True, trace=True,
                            error_action='ignore', suppress_warnings=True)

    print(auto_model.summary())

    # Extract the optimal parameters
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    # Fit the SARIMAX model with the optimal parameters and exogenous variables
    sarimax_model = SARIMAX(train_df['Price'], 
                            order=order, seasonal_order=seasonal_order, 
                            exog=train_df[exogenous_features])
    sarimax_model_fit = sarimax_model.fit()

    # Dynamic prediction
    predictions = []
    history = train_df['Price'].tolist()
    exog_history = train_df[exogenous_features].values.tolist()

    for t in range(len(test_df)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, exog=exog_history)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(history), end=len(history), exog=[test_df[exogenous_features].iloc[t]], dynamic=True)
        predictions.append(yhat[0])
        history.append(test_df['Price'].iloc[t])
        exog_history.append(test_df[exogenous_features].iloc[t].tolist())

    # Convert the predictions to a DataFrame
    forecast_mean_sarimax = pd.DataFrame(predictions, index=test_df.index, columns=['Forecast'])
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Price'], label='Train Price')
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, forecast_mean_sarimax, label='SARIMAX Forecasted Price', linestyle='--', color='red')
        
    # Calculate Bollinger Bands for train and test sets
    df['20_SMA'] = df['Price'].rolling(window=20).mean()
    df['20_STD'] = df['Price'].rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Plot Bollinger Bands for train and test sets
    plt.plot(df.index, df['20_SMA'], label='20 SMA', color='orange')
    plt.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='blue', alpha=0.1)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'SARIMAX Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_df['Price'], forecast_mean_sarimax))

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_df['Price'], forecast_mean_sarimax)

    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    return rmse, mape

#Prophet
from prophet import Prophet

def prophet_forecast(df, dataset_name):
    """
    Trains a Prophet model with exogenous variables and forecasts future prices.

    Parameters:
    train_df (pd.DataFrame): The training dataframe containing 'Date' and 'Price' columns and exogenous variables.
    test_df (pd.DataFrame): The testing dataframe containing 'Date' and 'Price' columns and exogenous variables.
    exogenous_vars (list): List of exogenous variables to include in the Prophet model.

    Returns:
    tuple: A tuple containing the RMSE and MAPE of the forecast.
    """
    train_df = df[df['Year'] == 2020]
    test_df = df[df['Year'] == 2021]

    # Prepare the data for Prophet
    prophet_train = train_df.rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_test = test_df.rename(columns={'Date': 'ds', 'Price': 'y'})

    # Initialize the Prophet model
    model = Prophet()

    exogenous_vars = ['Open', 'High', 'Low', 'Volume (M)', 'Change %']

    # Add additional regressors
    for var in exogenous_vars:
        model.add_regressor(var)

    # Fit the model
    model.fit(prophet_train[['ds', 'y'] + exogenous_vars])

    # Prepare the test data for prediction
    future = prophet_test[['ds'] + exogenous_vars]

    # Make predictions
    forecast = model.predict(future)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(prophet_train['ds'], prophet_train['y'], label='Train Price')
    plt.plot(prophet_test['ds'], prophet_test['y'], label='Test Price')
    plt.plot(prophet_test['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='red')

    # Calculate Bollinger Bands for the combined dataset
    df['ds'] = df['Date']
    df['y'] = df['Price']
    df['20_SMA'] = df['y'].rolling(window=20).mean()
    df['20_STD'] = df['y'].rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Plot Bollinger Bands for train and test sets
    plt.plot(df['ds'], df['20_SMA'], label='20 SMA', color='orange')
    plt.fill_between(df['ds'], df['Upper_Band'], df['Lower_Band'], color='blue', alpha=0.1)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Prophet Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = mean_squared_error(prophet_test['y'], forecast['yhat'], squared=False)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(prophet_test['y'], forecast['yhat'])

    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    return rmse, mape

## Weekly data
import pandas as pd

# Sample data
data = {
    'Date': ['2021-03-31 00:00:00', '2021-03-30 00:00:00', '2021-03-29 00:00:00', '2021-03-26 00:00:00', '2021-03-25 00:00:00'],
    'Price': [291.02, 293.25, 293.3, 291.22, 286.66],
    'Open': [294.0, 294.86, 289.72, 288.63, 286.5],
    'High': [294.42, 295.72, 294.09, 292.75, 287.03],
    'Low': [290.26, 291.5, 289.26, 288.32, 283.85],
    'Vol.': ['47.00M', '38.81M', '43.68M', '56.07M', '35.22M'],
    'Change %': [-0.0076, -0.0002, 0.0071, 0.0159, 0.003],
}

# Create a DataFrame
df = pd.DataFrame(data)

def preprocess_and_resample(data, freq):
    """
    Preprocess a DataFrame by converting 'Date' to datetime, 'Price' to numeric,
    dropping rows with NaNs in these columns, sorting by 'Date', and resetting the index.
    Then, resample the data to weekly frequency.
    
    Parameters:
    data (pd.DataFrame): DataFrame to preprocess and resample.
    
    Returns:
    pd.DataFrame: Resampled weekly DataFrame.
    """
  # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    # Convert 'Price' column to numeric
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    # Drop rows where the 'Date' or 'Price' conversion failed
    data = data.dropna(subset=['Date', 'Price'])
    # Drop rows where 'Vol.' is '-'
    data = data[data['Vol.'] != '-']
    # Replace 'M' and 'K', convert to numeric, and multiply accordingly
    data['Volume (M)'] = data['Vol.'].str.replace('M', '').str.replace('K', 'e-3').str.replace('[^\d.-]', '').astype(float)
    # Multiply values by 0.001 where the letter is 'K'
    data.loc[data['Vol.'].str.endswith('K'), 'Volume (M)'] *= 0.001
    # Drop the original 'Vol.' column
    data.drop(columns=['Vol.'], inplace=True)
    # Adding numeric Year & Month columns
    data['Year'] = data['Date'].dt.year.astype(int)
    data['Month'] = data['Date'].dt.month.astype(int)
    # Sort the DataFrame by the 'Date' column
    data = data.sort_values('Date')
    # Reset the index
    data = data.reset_index(drop=True)
    
    # Set 'Date' as the index for resampling
    data.set_index('Date', inplace=True)
    
    # Resample data to weekly frequency
    weekly_df = data.resample(freq).agg({
        'Price': 'last',           # Use the last closing price of the week
        'Open': 'first',           # Use the first opening price of the week
        'High': 'max',             # Use the highest price of the week
        'Low': 'min',              # Use the lowest price of the week
        'Volume (M)': 'sum',       # Sum the volumes of the week
        'Change %': 'sum'          # Sum the percentage changes of the week
    })
    
    # Reset the index to make 'Date' a column again
    weekly_df.reset_index(inplace=True)
    
    return weekly_df
 	
#ARIMA
# Convert 'Date' column to datetime if it isn't already
def arima_monthly(df, dataset_name):
    if df['Date'].dtype != '<M8[ns]':
        df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Normalize the data
    columns_to_normalize = ['Price']
    scaler = MinMaxScaler()

    # Fit the scaler on the data and transform
    df_scaled = df.copy()
    df_scaled[columns_to_normalize] = scaler.fit_transform(df_scaled[columns_to_normalize])

    # Split the data into training and testing sets
    train_df = df_scaled[df_scaled.index.year == 2020].copy()
    test_df = df_scaled[df_scaled.index.year == 2021].copy()

    # Check for stationarity
    result = adfuller(train_df['Price'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    # If p-value > 0.05, data is non-stationary, apply differencing
    if result[1] > 0.05:
        train_df['Price_diff'] = train_df['Price'].diff().dropna()
    else:
        train_df['Price_diff'] = train_df['Price']

    # Find the optimal parameters using auto_arima
    stepwise_model = auto_arima(train_df['Price_diff'].dropna(), start_p=1, start_q=1,
                                max_p=5, max_q=5, seasonal=False,
                                d=1, trace=True, error_action='ignore', suppress_warnings=True)
    print(stepwise_model.summary())

    # Fit the ARIMA model with the optimal parameters
    p, d, q = stepwise_model.order
    arima_model = ARIMA(train_df['Price'], order=(p, d, q))
    arima_model = arima_model.fit()

    # Forecasting
    forecast = arima_model.get_forecast(steps=len(test_df))

    # Forecast the future values (for the length of the test set)
    forecast_mean = forecast.predicted_mean
    
    # Create a combined dataframe for plotting
    train_df['Forecast'] = None
    test_df['Forecast'] = forecast_mean.values

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Price'], label='Train Price')
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, test_df['Forecast'], label='Forecast', color='red')

    # Calculate Bollinger Bands for train and test sets
    df_scaled['20_SMA'] = df_scaled['Price'].rolling(window=3).mean()
    df_scaled['20_STD'] = df_scaled['Price'].rolling(window=3).std()
    df_scaled['Upper_Band'] = df_scaled['20_SMA'] + (df_scaled['20_STD'] * 2)
    df_scaled['Lower_Band'] = df_scaled['20_SMA'] - (df_scaled['20_STD'] * 2)

    # Plot Bollinger Bands for train and test sets
    plt.plot(df_scaled.index, df_scaled['20_SMA'], label='3 Month SMA', color='orange')
    plt.fill_between(df_scaled.index, df_scaled['Upper_Band'], df_scaled['Lower_Band'], color='blue', alpha=0.1)
        
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'ARIMA Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_df['Price'], test_df['Forecast']))

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_df['Price'], test_df['Forecast'])

    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    return rmse, mape

##SARIMAX
def sarimax_forecast_monthly(df, dataset_name):
    """
    Perform SARIMAX modeling, forecasting, and evaluation on the given training and testing DataFrames.
    
    Parameters:
    train_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for training.
    test_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for testing.
    
    Returns:
    float: RMSE (Root Mean Squared Error) of the forecast.
    float: MAPE (Mean Absolute Percentage Error) of the forecast.
    """
    train_df = df[df['Year'] == 2020]
    test_df = df[df['Year'] == 2021]

    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)

    # Ensure all numeric columns are correctly cast to numeric types
    numeric_columns = train_df.columns.difference(['Date'])
    train_df[numeric_columns] = train_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    test_df[numeric_columns] = test_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill missing values with forward fill method
    train_df = train_df.fillna(method='ffill')
    test_df = test_df.fillna(method='ffill')

    # Drop remaining rows with missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Check if the train and test DataFrames are not empty
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test DataFrame is empty after preprocessing.")

    # Select exogenous features, excluding 'Price' and 'Date'
    exogenous_features = train_df.columns.difference(['Price', 'Date']).tolist()

    # Use auto_arima to find the optimal parameters
    auto_model = auto_arima(train_df['Price'], 
                            seasonal=True, exogenous=train_df[exogenous_features],
                            stepwise=True, trace=True,
                            error_action='ignore', suppress_warnings=True)

    print(auto_model.summary())

    # Extract the optimal parameters
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    # Fit the SARIMAX model with the optimal parameters and exogenous variables
    sarimax_model = SARIMAX(train_df['Price'], 
                            order=order, seasonal_order=seasonal_order, 
                            exog=train_df[exogenous_features])
    sarimax_model_fit = sarimax_model.fit()

    # Dynamic prediction
    predictions = []
    history = train_df['Price'].tolist()
    exog_history = train_df[exogenous_features].values.tolist()

    for t in range(len(test_df)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, exog=exog_history)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(history), end=len(history), exog=[test_df[exogenous_features].iloc[t]], dynamic=True)
        predictions.append(yhat[0])
        history.append(test_df['Price'].iloc[t])
        exog_history.append(test_df[exogenous_features].iloc[t].tolist())
    
    # Convert the predictions to a DataFrame
    forecast_mean_sarimax = pd.DataFrame(predictions, index=test_df.index, columns=['Forecast'])

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Price'], label='Train Price')
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, forecast_mean_sarimax, label='SARIMAX Forecasted Price', linestyle='--', color='red')
        
    # Calculate Bollinger Bands for train and test sets
    df['20_SMA'] = df['Price'].rolling(window=3).mean()
    df['20_STD'] = df['Price'].rolling(window=3).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Plot Bollinger Bands for train and test sets
    plt.plot(df.index, df['20_SMA'], label='3 Month SMA', color='orange')
    plt.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='blue', alpha=0.1)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'SARIMAX Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_df['Price'], forecast_mean_sarimax))

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_df['Price'], forecast_mean_sarimax)

    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    return rmse, mape

#Prophet
from prophet import Prophet

def prophet_forecast_monthly(df, dataset_name):
    """
    Trains a Prophet model with exogenous variables and forecasts future prices.

    Parameters:
    train_df (pd.DataFrame): The training dataframe containing 'Date' and 'Price' columns and exogenous variables.
    test_df (pd.DataFrame): The testing dataframe containing 'Date' and 'Price' columns and exogenous variables.
    exogenous_vars (list): List of exogenous variables to include in the Prophet model.

    Returns:
    tuple: A tuple containing the RMSE and MAPE of the forecast.
    """
    train_df = df[df['Year'] == 2020]
    test_df = df[df['Year'] == 2021]

    # Prepare the data for Prophet
    prophet_train = train_df.rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_test = test_df.rename(columns={'Date': 'ds', 'Price': 'y'})

    # Initialize the Prophet model
    model = Prophet()

    exogenous_vars = ['Open', 'High', 'Low', 'Volume (M)', 'Change %']

    # Add additional regressors
    for var in exogenous_vars:
        model.add_regressor(var)

    # Fit the model
    model.fit(prophet_train[['ds', 'y'] + exogenous_vars])

    # Prepare the test data for prediction
    future = prophet_test[['ds'] + exogenous_vars]

    # Make predictions
    forecast = model.predict(future)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(prophet_train['ds'], prophet_train['y'], label='Train Price')
    plt.plot(prophet_test['ds'], prophet_test['y'], label='Test Price')
    plt.plot(prophet_test['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='red')

    # Calculate Bollinger Bands for the combined dataset
    df['ds'] = df['Date']
    df['y'] = df['Price']
    df['20_SMA'] = df['y'].rolling(window=3).mean()
    df['20_STD'] = df['y'].rolling(window=3).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Plot Bollinger Bands for train and test sets
    plt.plot(df['ds'], df['20_SMA'], label='3 Month SMA', color='orange')
    plt.fill_between(df['ds'], df['Upper_Band'], df['Lower_Band'], color='blue', alpha=0.1)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Prophet Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = mean_squared_error(prophet_test['y'], forecast['yhat'], squared=False)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(prophet_test['y'], forecast['yhat'])

    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    return rmse, mape

#RECOMMENDATION SYSTEM FOR SARIMAX
def sarimax_recommend(df, dataset_name):
    """
    Perform SARIMAX modeling, forecasting, and evaluation on the given training and testing DataFrames.
    
    Parameters:
    train_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for training.
    test_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for testing.
    
    Returns:
    float: RMSE (Root Mean Squared Error) of the forecast.
    float: MAPE (Mean Absolute Percentage Error) of the forecast.
    """
    # Ensure 'Date' is in datetime format and set it as the index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create 'Year' column
    df['Year'] = df.index.year
    
    train_df = df[df['Year'] == 2020]
    test_df = df[df['Year'] == 2021]

    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)

    # Ensure all numeric columns are correctly cast to numeric types
    numeric_columns = train_df.columns.difference(['Date'])
    train_df[numeric_columns] = train_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    test_df[numeric_columns] = test_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill missing values with forward fill method
    train_df = train_df.fillna(method='ffill')
    test_df = test_df.fillna(method='ffill')

    # Drop remaining rows with missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Select exogenous features, excluding 'Price' and 'Date'
    exogenous_features = train_df.columns.difference(['Price', 'Date']).tolist()

    # Use auto_arima to find the optimal parameters
    auto_model = auto_arima(train_df['Price'], 
                            seasonal=True, exogenous=train_df[exogenous_features],
                            stepwise=True, trace=True,
                            error_action='ignore', suppress_warnings=True)

    # Extract the optimal parameters
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    # Fit the SARIMAX model with the optimal parameters and exogenous variables
    sarimax_model = SARIMAX(train_df['Price'], 
                            order=order, seasonal_order=seasonal_order, 
                            exog=train_df[exogenous_features])
    sarimax_model_fit = sarimax_model.fit()

    # Dynamic prediction
    predictions = []
    history = train_df['Price'].tolist()
    exog_history = train_df[exogenous_features].values.tolist()

    for t in range(len(test_df)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, exog=exog_history)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(history), end=len(history), exog=[test_df[exogenous_features].iloc[t]], dynamic=True)
        predictions.append(yhat[0])
        history.append(test_df['Price'].iloc[t])
        exog_history.append(test_df[exogenous_features].iloc[t].tolist())

    # Convert the predictions to a DataFrame
    forecast_mean_sarimax = pd.DataFrame(predictions, index=test_df.index, columns=['Forecast'])

    # Calculate Bollinger Bands for the combined dataset
    df['20_SMA'] = df['Price'].rolling(window=20).mean()
    df['20_STD'] = df['Price'].rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + df['20_STD']
    df['Lower_Band'] = df['20_SMA'] - df['20_STD']

    # Generate recommendations based on alternation between "Buy" and "Sell"
    in_position = False
    recommendations = []
    capital = 10000
    shares = 0
    trade_dates = []
    buy_prices = []
    sell_prices = []
    account_balance = []
    num_trades = 0

    for i in range(len(test_df)):
        if not in_position and forecast_mean_sarimax.iloc[i]['Forecast'] <= df['Lower_Band'].iloc[i + len(train_df)]:
            recommendations.append('Buy')
            shares = capital / test_df['Price'].iloc[i]
            trade_dates.append(test_df.index[i])
            buy_prices.append(test_df['Price'].iloc[i])
            account_balance.append(capital)
            in_position = True
        elif in_position and forecast_mean_sarimax.iloc[i]['Forecast'] >= df['20_SMA'].iloc[i + len(train_df)]:
            recommendations.append('Sell')
            capital = shares * test_df['Price'].iloc[i]
            trade_dates.append(test_df.index[i])
            sell_prices.append(test_df['Price'].iloc[i])
            account_balance.append(capital)
            shares = 0
            num_trades += 1
            in_position = False
        else:
            recommendations.append('Hold')
            account_balance.append(capital)

    test_df['Forecast'] = forecast_mean_sarimax
    test_df['Recommendation'] = recommendations
    test_df['Upper_Band'] = df['Upper_Band'].loc[test_df.index]
    test_df['Lower_Band'] = df['Lower_Band'].loc[test_df.index]
    test_df['20_SMA'] = df['20_SMA'].loc[test_df.index]

    # Calculate hold times
    # zip pairs each buy date with the corresponding sell date
    hold_times = [(sell_date - buy_date).days for buy_date, sell_date in zip(trade_dates[::2], trade_dates[1::2])]
    avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

    # Calculate average return
    returns = [(sell_price - buy_price) / buy_price for buy_price, sell_price in zip(buy_prices, sell_prices)]
    avg_return = (sum(returns) / num_trades * 100) if num_trades > 0 else 0

    # Plot the results for the test set (year 2021) only
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, test_df['Forecast'], label='SARIMAX Forecasted Price', linestyle='--', color='purple')

    # Plot Bollinger Bands for the test set
    plt.fill_between(test_df.index, test_df['Upper_Band'], test_df['Lower_Band'], color='blue', alpha=0.1)
    
    # Plot the 20-day SMA for the test set
    plt.plot(test_df.index, test_df['20_SMA'], label='20 SMA', color='orange')

    # Plot buy and sell signals
    buy_signals = test_df[test_df['Recommendation'] == 'Buy']
    sell_signals = test_df[test_df['Recommendation'] == 'Sell']
    
    plt.scatter(buy_signals.index, buy_signals['Forecast'], marker='^', color='green', label='Buy Signal', s=70)
    plt.scatter(sell_signals.index, sell_signals['Forecast'], marker='v', color='red', label='Sell Signal', s=70)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'SARIMAX - Buy & Sell Signals - {dataset_name}')
    plt.legend()
    plt.show()

    # Plot account balance over time
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, account_balance, label='Account Balance', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Account Balance ($)')
    plt.title(f'Account Balance Over Time - {dataset_name}')
    plt.legend()
    plt.show()

    # Print summary statistics
    print(f"Number of trades: {num_trades}")
    print(f"Average hold time: {avg_hold_time:.2f} days")
    print(f"Average return per trade: {avg_return:.2f}%")
    print(f"Final account balance: ${capital:.2f}")

    return test_df

#Linear Regression Recommendation
def lr_recommend(data, dataset_name, leads=[1]):
    """
    Perform linear regression modeling, forecasting, and evaluation on the given dataset.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Date' and 'Price' columns.
    dataset_name (str): Name of the dataset for plotting and reporting.

    Returns:
    None
    """
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Ensure 'Price' is integer
    df['Price'] = df['Price'].astype(int)

    # Create lagged features
    n_lags = 5
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Price'].shift(lag)

    # Create lead features
    for lead in leads:
        df[f'lead_{lead}'] = df['Price'].shift(-lead)

    df.rename(columns={'Price': 'lag_0'}, inplace=True)
    df.dropna(inplace=True)

    # Split the data into training and testing sets
    train = df[df.index.year == 2020].copy()
    test = df[df.index.year == 2021].copy()

    y_cols = [f'lead_{j}' for j in leads]
    x_cols = [f'lag_{j}' for j in range(1, n_lags + 1)]

    # Train the Linear Regression model
    model = LinearRegression().fit(train[x_cols].values, train[y_cols])

    # Initialize a DataFrame to store the test predictions
    test_predictions_df = pd.DataFrame(index=test.index, columns=y_cols)

    # Iteratively predict one day at a time
    for date in test.index:
        if test.loc[date, x_cols].isna().any():
            # Use last n_lags rows from the train set to initialize the first prediction
            lagged_features = train[x_cols].iloc[-1].values.reshape(1, -1)
        else:
            lagged_features = test.loc[date, x_cols].values.reshape(1, -1)

        # Predict the next day's prices
        predicted_values = model.predict(lagged_features)[0]

        # Store the predictions
        test_predictions_df.loc[date] = predicted_values

        # Update the lagged features in the test set
        for lag in range(n_lags, 0, -1):
            if lag == 1:
                if (date + pd.Timedelta(days=1)) in test.index:
                    test.loc[date + pd.Timedelta(days=1), f'lag_{lag}'] = predicted_values[0]
            else:
                test[f'lag_{lag}'] = test[f'lag_{lag-1}'].shift(-1)

        # Update the test set's lag_0 value with the predicted value
        if (date + pd.Timedelta(days=1)) in test.index:
            test.loc[date + pd.Timedelta(days=1), 'lag_0'] = predicted_values[0]

    # Calculate Bollinger Bands for train and test sets
    df['20_SMA'] = df['lag_0'].rolling(window=20).mean()
    df['20_STD'] = df['lag_0'].rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

    # Generate recommendations for lead = 1
    recommendations = []
    for i in range(len(test)):
        if test_predictions_df[f'lead_{leads[0]}'].iloc[i] >= df['Upper_Band'].iloc[i + len(train)]:
            recommendations.append('Sell')
        elif test_predictions_df[f'lead_{leads[0]}'].iloc[i] <= df['Lower_Band'].iloc[i + len(train)]:
            recommendations.append('Buy')
        else:
            recommendations.append('Hold')

    test['Forecast'] = test_predictions_df[f'lead_{leads[0]}']
    test['Recommendation'] = recommendations
    test['Upper_Band'] = df['Upper_Band'].loc[test.index]
    test['Lower_Band'] = df['Lower_Band'].loc[test.index]

    # Plot the results for the test set (year 2021) only
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['lag_0'], label='Test Price')
    plt.plot(test.index, test['Forecast'], label='Linear Regression Forecasted Price', linestyle='--', color='purple')

    # Plot Bollinger Bands for the test set
    plt.fill_between(test.index, test['Upper_Band'], test['Lower_Band'], color='blue', alpha=0.1)

    # Plot buy and sell signals
    buy_signals = test[test['Recommendation'] == 'Buy']
    sell_signals = test[test['Recommendation'] == 'Sell']
    
    plt.scatter(buy_signals.index, buy_signals['Forecast'], marker='^', color='green', label='Buy Signal', s=50)
    plt.scatter(sell_signals.index, sell_signals['Forecast'], marker='v', color='red', label='Sell Signal', s=50)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Linear Regression - Buy & Sell Signals - {dataset_name}')
    plt.legend()
    plt.show()