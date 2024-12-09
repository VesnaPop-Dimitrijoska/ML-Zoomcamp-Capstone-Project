import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib  # To save the model

def train_best_model(file_path):
    # Load data
    df = pd.read_csv(file_path,index_col=0, parse_dates=['Date'])

    # Train-test split 
    train_total = df.loc[df.index <  pd.Timestamp('2018-04-01')]
    test_total  = df.loc[df.index >= pd.Timestamp('2018-04-01')]

    # Time Series Forecasting with ARIMA - using best parameters for (p, d, q) -> (7, 2, 2)
    model_ARIMA = ARIMA(train_total['Sessions'],
                exog=None,
                order=(7, 2, 2),
                seasonal_order=(0, 0, 0, 0), 
                trend=None, 
                enforce_stationarity=True, 
                freq='1D')

    results = model_ARIMA.fit()

    return results


if __name__ == "__main__":
    data_path = 'df_total_final.csv'
    output_model_path = "best_model.pkl"

    # Train the model
    best_model = train_best_model(data_path)

    # Save the model to a file
    joblib.dump(best_model, output_model_path)
    print(f"Model saved to {output_model_path}")