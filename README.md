# stocks_direction

The project was an attempt to forecast stock direction movement based on past observations.
More details inside .py file.

# Unfortunately the model does not give good results even after trying many different variants:
# original data, first and second differences, diffrent numbers of LSTM units, layers, dropout layers, no. of epochs,
# LSTM, GRU, SimpleRNN units

# it might be impossible to get better results at predicting direction of the stock movement
# things to further try:
# - ARIMA / GARCH models analysis --- there is conditional heteroscedacity in first diffrences, so GARCH might be
# working better, or at least give some insight if we analyze statistically significant lag periods

# another thing to try is to use more metrics as exgoenic variables -- eg. EPS, P/B, FCF
