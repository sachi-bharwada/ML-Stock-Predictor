import pandas as pd
import yfinance as yf  # calls yahoo finance api to get daily stock and index prices

# Works by training individual decision trees with randomized parameters and then
# averaging those results from the decision trees. Thus, random forests are resistant to
# overfitting, it harder for them to overfit. Also, they can pick up non-linear tendencies in data
# in stock price predictions, most relationships are not linear
from sklearn.ensemble import RandomForestClassifier

# when we say market will go up, did it actually go up/down depending on what the prediction is
from sklearn.metrics import precision_score

sp500 = yf.Ticker("^GSPC")  # Ticker class to help enable download of price history for a single symbol
sp500 = sp500.history(period="max")  # query historical prices, query data from the beginning from when index was
# created
# we want open, high low, and close columns in order to predict of stock price will go up
# or down tomorrow

del sp500["Dividends"]
del sp500["Stock Splits"]

# some people like abs. price, the big problem is your model can be really accurate,
# but you can still lose a lot of money, you don't need accurate price, you want accurate directionality
# accurate on price but bad on directionality

sp500["Tomorrow"] = sp500["Close"].shift(-1)  # shifts all prices back one day of Close column

# return boolean  if tomorrow's price is greater than today's price
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()

# n_estimators = num of individual trees we want to train (the higher, the better --> generally)
# min_samples_split = helps to protect from overfitting if the tree is built to deeply,
# the higher this is the less accurate the model is but the less it will overfit
# random_state = random forests have some randomization, setting a random state means if
# we run the same model twice the randoms numbers that are generated will be in a predictable sequence each time
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# time series data, so we can't use cross validation --> b/c you use future data to predict the past
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]


# BUILD BACKTESTING SYSTEM
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])  # column since model is not going to know the future
    preds = model.predict_proba(test[predictors])[:, 1]  # This will return a probability that the run will be a 0 or 1,
    # so the probability that the stock price will go up or down tomorrow
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    # Combine actual values with predicted values
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# start val and step val = when you back test you want a certain amount data to train your first model.
# So every trading year has 250 days, so we are saying take 10 years of data and train your model with
# 10 years of data, we are going to train model for a year and then going to the next year and so on
# take the first 10 years and predict values for 11th year, so on and so forth
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []  # list of dataframes, each dataframe is the predictions for a single year

    # loop across data, year by year
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()  # train set is for all year prior to current year
        test = data.iloc[i:(i + step)].copy()  # test set is for current year
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# ADD ADDITIONAL PREDICTORS TO MODEL
# Create various rolling averages
horizons = [2, 5, 60, 250, 1000]  # get the mean close price in the last 2, 5, 60, 250, and 1000 days
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    # number of days in the past x days that the stock price actually went up
    trend_column = f"Trend_{horizon}"

    # This is going to, on any day, it will look at the past few days and
    # see the sum of the target
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

predictions = backtest(sp500, model, new_predictors)
predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

sp500 = sp500.dropna()
