{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f0c724b8-b718-488d-8c99-41def86ae076",
   "metadata": {},
   "source": [
    "Combine ARIMA and ANN predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "913097ce-35f1-4491-91e6-af61ffbe91cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.\n",
      "  return get_prediction_index(\n",
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.\n",
      "  return get_prediction_index(\n",
      "C:\\Users\\Muneeba\\AppData\\Roaming\\Python\\Python311\\site-packages\\keras\\src\\layers\\rnn\\rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(**kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m96998/96998\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m828s\u001b[0m 9ms/step - loss: 0.0015\n",
      "\u001b[1m3032/3032\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 6ms/step\n",
      "\u001b[1m758/758\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 7ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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H9JPuOeee9S1a1dlZWWpfv362rt3r7766ivNmjXLp93gwYN14403SpLGjBmjGTNmaOzYsRo2bJi3zbPPPqsrr7yy3HpGjRql4cOHe/v597//rW+++cav76Vhw4Y+93/99Vft379fU6dO1aJFi9S1a1dJ0scff6y0tDRNmTJF/fr1kyQVFhbqzTffVNu2bSVJW7Zs0SeffKIdO3aofv36ko6fPzRjxgyNHz9e//znP/XCCy+oY8eOevPNN71jtmzZ0vv1ie/hhLFjxyo5OVnr1q1Tq1atlJmZqbPPPlsXX3yxXC6XzjzzTG/bM844Q5KUmJiolJQUv77/qmAlBwAAAI5x+eWXKyMjw+f27rvv+rTp1KmTWrZsqQ8++ECS9OGHH6pRo0a69NJLfdp16dLF+3VkZKQ6duyo9evX+7Tp2LFjubXk5uYqOzu7zH78sWDBAp/vo3bt2lq/fr0iIyPVuXNnb7u6deuqefPmPrVFR0f7rCytXLlShmHonHPO8Z43VKtWLc2fP997yNmJlZzybNmyRbfccovOOussJSQkqEmTJpKkzMxMScdXmDIyMtS8eXM99NBDmjlzpl/fpxlYyQki1nEAAIBTRcVGafih4ZaMG4i4uDg1a9bM57EdO3aUanfPPfdo9OjReuKJJzR+/Hjdeeedcrlcp+3/1DZxcXEB1ReIJk2aKDEx0eex8g7fMwzDp7aaNWv63C8pKVGNGjW0YsUK1ahRw+e1J1a+atasWWE911xzjdLS0vTOO++ofv36KikpUatWrVRQUCBJOv/887V161ZNnz5d3377rW666Sb16NGj1PlPocBKTiUZhqGJyzL102+5VpcCAABgOpfLpei46JDf/AkelXHbbbcpMzNTr7/+utauXas77rijVJsffvjB+3VRUZFWrFihc8891+8xPB6PUlNTy+ynss477zwVFRVpyZIl3sf27dunjRs3qkWLFuW+rn379iouLtbu3bvVrFkzn9uJw8fatGlT7vbX+/bt0/r16/XUU0+pe/fuatGihXdDgpMlJCTo5ptv1jvvvKOJEyfq888/1/79+yVJUVFRpQ4dNAsrOZX07frdevzzNT6PcUoOAACAPdSuXVs33HCDHnvsMfXs2bPU+S+S9MYbb+jss89WixYt9OqrryonJ0d33XVXQOM8/PDDGjlypLefV155pUrX6zn77LPV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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from statsmodels.tsa.arima.model import ARIMA\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, LSTM\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "# Load preprocessed data\n",
    "aep_hourly = pd.read_csv('processed_aep_hourly.csv', index_col='Datetime', parse_dates=True)\n",
    "\n",
    "# ARIMA Model\n",
    "def train_arima(data, order):\n",
    "    model = ARIMA(data, order=order)\n",
    "    model_fit = model.fit()\n",
    "    return model_fit\n",
    "\n",
    "def forecast_arima(model_fit, steps):\n",
    "    forecast = model_fit.forecast(steps=steps)\n",
    "    return forecast\n",
    "\n",
    "arima_order = (5, 1, 0)\n",
    "arima_model_fit = train_arima(aep_hourly['AEP_MW'].dropna(), arima_order)\n",
    "arima_forecast = forecast_arima(arima_model_fit, len(aep_hourly))\n",
    "\n",
    "# ANN Model\n",
    "scaler = MinMaxScaler()\n",
    "scaled_data = scaler.fit_transform(aep_hourly[['AEP_MW']])\n",
    "\n",
    "def create_dataset(data, time_step=1):\n",
    "    X, Y = [], []\n",
    "    for i in range(len(data) - time_step - 1):\n",
    "        a = data[i:(i + time_step), 0]\n",
    "        X.append(a)\n",
    "        Y.append(data[i + time_step, 0])\n",
    "    return np.array(X), np.array(Y)\n",
    "\n",
    "time_step = 24\n",
    "X, Y = create_dataset(scaled_data, time_step)\n",
    "\n",
    "train_size = int(len(X) * 0.8)\n",
    "test_size = len(X) - train_size\n",
    "X_train, X_test = X[0:train_size], X[train_size:len(X)]\n",
    "Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]\n",
    "\n",
    "X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)\n",
    "X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)\n",
    "\n",
    "model = Sequential()\n",
    "model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))\n",
    "model.add(LSTM(50, return_sequences=False))\n",
    "model.add(Dense(25))\n",
    "model.add(Dense(1))\n",
    "\n",
    "model.compile(optimizer='adam', loss='mean_squared_error')\n",
    "model.fit(X_train, Y_train, batch_size=1, epochs=1)\n",
    "\n",
    "train_predict = model.predict(X_train)\n",
    "test_predict = model.predict(X_test)\n",
    "\n",
    "train_predict = scaler.inverse_transform(train_predict)\n",
    "test_predict = scaler.inverse_transform(test_predict)\n",
    "\n",
    "# Hybrid Model\n",
    "def hybrid_model(arima_pred, ann_pred, alpha=0.5):\n",
    "    arima_pred_reshaped = arima_pred.to_numpy().reshape(-1, 1)  # Convert to numpy array and reshape\n",
    "    return alpha * arima_pred_reshaped + (1 - alpha) * ann_pred\n",
    "\n",
    "hybrid_forecast = hybrid_model(arima_forecast[-test_size:], test_predict)\n",
    "\n",
    "# Plot results\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(aep_hourly['AEP_MW'], label='Actual')\n",
    "plt.plot(range(len(aep_hourly) - test_size, len(aep_hourly)), hybrid_forecast, label='Hybrid Forecast', color='purple')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2d65d6c-77c9-46ed-aa5c-c50120f8248e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90917e06-cb71-4915-8a19-830fc9104c03",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75efac0a-6013-4318-afed-619a79cf2dc0",
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
