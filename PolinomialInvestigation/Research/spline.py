import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import sqrt as sqrt
from sklearn.metrics import mean_squared_error as mean_squared_error

# Adapted from https://www.kaggle.com/renanhuanca/regression-splines
# Original idea in 2018 from GURCHETAN SINGH, https://www.analyticsvidhya.com/blog/2018/03/introduction-regression-splines-python-codes/

def spline(real, noise):

    data_d = []
    le = len(noise) + 1
    data_d.extend(range(1, le))

    df= pd.DataFrame({'n':data_d ,'onda':noise})
    df.to_csv("data.csv", encoding='utf-8', index=False)

    data = pd.read_csv("data.csv")
    data_x = data['n']
    data_y = data['onda']
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.33, random_state=1)

    plt.scatter(train_x, train_y, facecolor='None', edgecolor='k', alpha=0.3)

    # Fit linear regression model
    x = train_x.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(x, train_y)

    # prediction on validation dataset
    valid_x = valid_x.values.reshape(-1, 1)
    pred = model.predict(valid_x)

    #visualization
    xp = np.linspace(valid_x.min(), valid_x.max(), 100)
    xp = xp.reshape(-1,1)
    pred_plot = model.predict(xp)

    plt.scatter(valid_x, valid_y, facecolor='None', edgecolor='k', alpha=0.3)

    #BASICO
    plt.plot(xp, pred_plot, label = "Regresion linear basica")

    #REAL
    plt.plot(xp, real, label = "Grafica real")


    rms = sqrt(mean_squared_error(valid_y, pred))

    weights = np.polyfit(train_x, train_y, 25)
    
    # Generating model with the given weights
    model = np.poly1d(weights)

    # Prediction on validation set
    pred = model(valid_x)

    # Plot the graph for 70 observations only
    xp = np.linspace(valid_x.min(), valid_x.max(), 100)
    pred_plot = model(xp)
    plt.scatter(valid_x, valid_y, facecolor='None', edgecolor='k', alpha=0.3)

    #POLINOMIAL.
    plt.plot(xp, pred_plot, label = "Regresion linear polinomial")
    plt.show()


    # dividing the data into 4 bins 
    df_cut, bins = pd.cut(train_x, 4, retbins=True, right=True)
    df_cut.value_counts(sort=False)

    df_steps = pd.concat([train_x, df_cut, train_y], keys=['n','n_cuts','onda'], axis=1)

    # create dummy variables for the age groups
    df_steps_dummies = pd.get_dummies(df_cut)

    df_steps_dummies.columns = ['17.938-33.5','33.5-49.0','49.0-64.5','64.5-80.0']

    # fitting generalized linear models
    fit3 = sm.GLM(df_steps.onda, df_steps_dummies).fit()

    # binning validation set into same 4 bins
    bin_mapping = np.digitize(valid_x, bins).flatten()

    X_valid = pd.get_dummies(bin_mapping)

    # removing any outliers
    X_valid = pd.get_dummies(bin_mapping).drop([5], axis=1)

    # prediction
    pred2 = fit3.predict(X_valid)

    # calculating RMSE
    rms = sqrt(mean_squared_error(valid_y, pred2))
    # print(rms)
    
    # we sill plot the graph for the 70 observations only
    xp = np.linspace(valid_x.min(), valid_x.max()-1, 100)
    bin_mapping= np.digitize(xp, bins)
    X_valid_2 = pd.get_dummies(bin_mapping)
    X_valid_2.drop(X_valid_2.columns[[4]], axis=1, inplace=True)


    pred2 = fit3.predict(X_valid_2)

    # visualization
    fig, (ax1) =  plt.subplots(1,1, figsize=(12,5))
    fig.suptitle("Piecewise constant", fontsize=14)

    # scatter plot with polynomial regression line
    ax1.scatter(train_x, train_y, facecolor='None', edgecolor='k', alpha=0.3)
    ax1.plot(xp, pred2, c='b')
    plt.show()

    # generating cubic spline with 3 knots at 25, 40 and 60
    transformed_x = dmatrix("bs(train, knots=(25,40,60), degree=3, include_intercept=False)", {"train": train_x}, return_type='dataframe')

    # fitting generalized linear model on transformed dataset
    fit1 = sm.GLM(train_y, transformed_x).fit()

    # generating cubic spline with 4 knots
    transformed_x2 = dmatrix("bs(train, knots=(25,40,50,65), degree=3, include_intercept=False)", {"train": train_x}, return_type='dataframe')

    # fitting generalized linear model on transformed dataset
    fit2 = sm.GLM(train_y, transformed_x2).fit()

    # predictions on both splines
    pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,40,60), include_intercept=False)", {"valid":valid_x}, return_type='dataframe'))
    pred2 = fit2.predict(dmatrix("bs(valid, knots=(25, 40,50,65), degree=3, include_intercept=False)", {"valid":valid_x}, return_type='dataframe'))

    # calculating rmse
    rms1 = sqrt(mean_squared_error(valid_y, pred1))
    print(rms1)

    rms2 = sqrt(mean_squared_error(valid_y, pred2))
    print(rms2)

    # we wil plot the graph for 70 observations only
    xp = np.linspace(valid_x.min(), valid_x.max(), 100)

    # make some predictions
    pred1 = fit1.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp":xp}, return_type='dataframe'))
    pred2 = fit2.predict(dmatrix("bs(xp, knots=(25,40,50,60), include_intercept=False)", {"xp":xp}, return_type='dataframe'))

    # plot the splines and error bands
    plt.scatter(data.n, data.onda, facecolor='None', edgecolor='k', alpha=0.1)
    plt.plot(xp, pred1, label='Specifying degree=3 with 3 knots')
    plt.plot(xp, pred2, label='Specifying degree=3 with 4 knots')
    plt.plot(xp, real, label = 'Real')
    plt.legend()
    plt.show()


