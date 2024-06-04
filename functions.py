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
results_df = pd.DataFrame(columns=['Dataset', 'Mean Squared Error', 'R^2 Score'])

def linear_regression_forecast(data, dataset_name):
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Ensure 'Price' is integer
    df['Price'] = df['Price'].astype(int)

    # Create a numeric representation of the date for regression
    df['Date_ordinal'] = df.index.map(pd.Timestamp.toordinal)

    # Split the data into training and testing sets
    train = df[df.index.year == 2020]
    test = df[df.index.year == 2021]

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(train[['Date_ordinal']], train['Price'])

    # Make predictions on the test data
    test['Predicted_Price'] = model.predict(test[['Date_ordinal']])

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Price'], label='Train Price')
    plt.plot(test.index, test['Price'], label='Test Price')
    plt.plot(test.index, test['Predicted_Price'], label='Predicted Price', linestyle='--', color='red')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Linear Regression Forecast vs Actual - {dataset_name}')
    plt.legend()
    plt.show()

    # Evaluate the model
    mse = mean_squared_error(test['Price'], test['Predicted_Price'])
    r2 = r2_score(test['Price'], test['Predicted_Price'])
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Append the results to the results DataFrame
    results_df.loc[len(results_df)] = [dataset_name, mse, r2]

#ARIMA
def arima(df_train, df_test, dataset_name):
    """
    Perform preprocessing, ARIMA modeling, and forecasting on the given training and testing DataFrames.
    
    Parameters:
    df_train (pd.DataFrame): DataFrame containing 'Date' and 'Price' columns for training.
    df_test (pd.DataFrame): DataFrame containing 'Date' and 'Price' columns for testing.
    
    Returns:
    float: RMSE (Root Mean Squared Error) of the forecast.
    float: MAPE (Mean Absolute Percentage Error) of the forecast.
    """
    # Normalize the data
    columns_to_normalize = ['Price']
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both train and test data
    train_df = df_train.copy()
    train_df[columns_to_normalize] = scaler.fit_transform(train_df[columns_to_normalize])

    test_df = df_test.copy()
    test_df[columns_to_normalize] = scaler.transform(test_df[columns_to_normalize])

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
    forecast_ci = forecast.conf_int()

    # Create a combined dataframe for plotting
    train_df['Forecast'] = None
    test_df['Forecast'] = forecast_mean.values

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Price'], label='Train Price')
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, test_df['Forecast'], label='Forecast', color='red')

    # Plot the confidence intervals
    plt.fill_between(test_df.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1], color='pink', alpha=0.3)

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
def sarimax_forecast(train_df, test_df, dataset_name):
    """
    Perform SARIMAX modeling, forecasting, and evaluation on the given training and testing DataFrames.
    
    Parameters:
    train_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for training.
    test_df (pd.DataFrame): DataFrame containing 'Date', 'Price', and other exogenous features for testing.
    
    Returns:
    float: RMSE (Root Mean Squared Error) of the forecast.
    float: MAPE (Mean Absolute Percentage Error) of the forecast.
    """
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

    # Forecast the future values for the length of the test set
    forecast_sarimax = sarimax_model_fit.get_forecast(steps=len(test_df), 
                                                      exog=test_df[exogenous_features])
    forecast_mean_sarimax = forecast_sarimax.predicted_mean
    forecast_ci_sarimax = forecast_sarimax.conf_int()

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Price'], label='Train Price')
    plt.plot(test_df.index, test_df['Price'], label='Test Price')
    plt.plot(test_df.index, forecast_mean_sarimax, label='SARIMAX Forecasted Price', linestyle='--', color='blue')
    plt.fill_between(test_df.index, forecast_ci_sarimax.iloc[:, 0], forecast_ci_sarimax.iloc[:, 1], color='blue', alpha=0.3)
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
