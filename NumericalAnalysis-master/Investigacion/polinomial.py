from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator
import damage, recognize, utils, divider
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

def polinomial(samples, newsamples):

damage.noiseadd(newsamples, 0.7, 0.3)
matches = recognize.cheat(samples, newsamples, false_positives=0.04, false_negatives=0.1)
matchesSD = recognize.cheat(samples, samples, false_positives=0.04, false_negatives=0.1)
xSD, ySD = utils.tovalidxy(samples, matchesSD)
x, y = utils.tovalidxy(newsamples, matches)
x = np.array(x).reshape((-1, 1))
y = np.array(y)

number = 100 #Number of segments
counter = 0
x_poly = []
y_poly_pred = []
sorted_zip = []
xAux = []
yAux = []
x_polyAux = []
y_poly_predAux = []
i = 0 
countAux = 0

divX = divider.divider(x,number)
divY = divider.divider(y,number)
ySD = divider.divider(ySD,number)

while counter < number:
    tmp = counter/number * 100

    if 0 <= tmp < 20:
        polynomial_features= PolynomialFeatures(degree=1)
    elif 20 <= tmp < 40:
        polynomial_features= PolynomialFeatures(degree=2)
    elif 40 <= tmp < 60:
        polynomial_features= PolynomialFeatures(degree=3)
    elif 60 <= tmp < 80:
        polynomial_features= PolynomialFeatures(degree=4)
    elif 80 <= tmp < 100:
        polynomial_features= PolynomialFeatures(degree=5)

    model = LinearRegression()
    x_poly.append(polynomial_features.fit_transform(divX[counter]))
    model.fit(x_poly[counter], divY[counter])
    y_poly_pred.append(model.predict(x_poly[counter]))
    sort_axis = operator.itemgetter(0)
    sorted_zip.append(sorted(zip(divX[counter],y_poly_pred[counter]), key=sort_axis))
    divX[counter], y_poly_pred[counter] = zip(*sorted_zip[counter])
    if counter != 0:
        auxX = divX[counter]
        lastAuxX = auxX[0:3]
        auxY = y_poly_pred[counter]
        lastAuxY = auxY[0:3]
        auxX2 = divX[counter - 1]
        initAuxX = auxX2[-3:]
        auxY2 = y_poly_pred[counter-1]
        initAuxY = auxY2[-3:]
        xAux.append(initAuxX)
        yAux.append(initAuxY)
        xAux.append(lastAuxX)
        yAux.append(lastAuxY)
    counter = counter + 1
    
xAux = list(itertools.chain.from_iterable(xAux))
yAux = list(itertools.chain.from_iterable(yAux))

while countAux < (len(xAux)-5):
    tmp = countAux/(len(xAux)-5) * 100

    elif 0 <= tmp < 20:
        polynomial_features= PolynomialFeatures(degree=1)
    elif 20 <= tmp < 40:
        polynomial_features= PolynomialFeatures(degree=2)
    elif 40 <= tmp < 60:
        polynomial_features= PolynomialFeatures(degree=3)
    elif 60 <= tmp < 80:
        polynomial_features= PolynomialFeatures(degree=4)
    elif 80 <= tmp < 100:
        polynomial_features= PolynomialFeatures(degree=5)

    model1 = LinearRegression()
    dxx = xAux[countAux:countAux+6]
    dyy = yAux[countAux:countAux+6]
    x_polyTest = polynomial_features.fit_transform(dxx)
    model1.fit(x_polyTest, dyy)
    y_poly_predTest = model1.predict(x_polyTest)
    sort_axis1 = operator.itemgetter(0)
    sorted_zip1 = sorted(zip(dxx,y_poly_predTest), key=sort_axis1)
    dxx, dyy = zip(*sorted_zip1)
    plt.plot(dxx, dyy, color='r')
    countAux = countAux + 6

counter = 0
plt.plot(samples, label='real')
ax = plt.axes()
ax.xaxis.grid()


while  counter < number :
    plt.plot(divX[counter], y_poly_pred[counter], color='y')
    counter = counter + 1

print("Division en ", number, " segmentos")

for s in range(len(x_poly)):
    rmse = np.sqrt(mean_squared_error(ySD[s],y_poly_pred[s]))
    r2 = r2_score(ySD[s],y_poly_pred[s])
    print("RMSE: ",rmse,"R2: ",r2)

plt.show()