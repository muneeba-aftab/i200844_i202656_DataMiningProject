{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "06c1218b-5a41-4b7e-ac2d-39c0d3c24433",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from statsmodels.tsa.stattools import adfuller\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Load datasets\n",
    "sp500 = pd.read_csv('sp500_index.csv', parse_dates=['Date'], index_col='Date')\n",
    "aep_hourly = pd.read_csv('AEP_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')\n",
    "co2 = pd.read_csv('Daily_atmospheric_CO2_concentration.csv', index_col='Unnamed: 0')\n",
    "\n",
    "# Clean data\n",
    "def clean_data(df):\n",
    "    df = df.dropna()\n",
    "    return df\n",
    "\n",
    "sp500 = clean_data(sp500)\n",
    "aep_hourly = clean_data(aep_hourly)\n",
    "co2 = clean_data(co2)\n",
    "\n",
    "# Normalize/Standardize data\n",
    "scaler = StandardScaler()\n",
    "\n",
    "sp500['S&P500'] = scaler.fit_transform(sp500[['S&P500']])\n",
    "aep_hourly['AEP_MW'] = scaler.fit_transform(aep_hourly[['AEP_MW']])\n",
    "co2[['cycle', 'trend']] = scaler.fit_transform(co2[['cycle', 'trend']])\n",
    "\n",
    "# Stationarization function\n",
    "def make_stationary(df, column):\n",
    "    result = adfuller(df[column])\n",
    "    if result[1] > 0.05:  # p-value > 0.05 suggests non-stationarity\n",
    "        df[column] = df[column].diff().dropna()  # Differencing\n",
    "    return df\n",
    "\n",
    "sp500 = make_stationary(sp500, 'S&P500')\n",
    "aep_hourly = make_stationary(aep_hourly, 'AEP_MW')\n",
    "co2 = make_stationary(co2, 'trend')\n",
    "\n",
    "# Save preprocessed data\n",
    "sp500.to_csv('processed_sp500.csv')\n",
    "aep_hourly.to_csv('processed_aep_hourly.csv')\n",
    "co2.to_csv('processed_co2.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64da880f-aa84-409f-bf4d-74b7e3ccb9e2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
