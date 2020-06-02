#! /usr/bin/env python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator
import damage, recognize, utils, divider, polinomial, spline
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

# Selecciona cancion y tomo el arreglo "wav"
samplerate, samples = wavfile.read('canciones/hakuna_matata.wav')
samples = samples[5000000:5000100]

# Crea una copia de la real y le agrega ruido
newsamples = samples.copy()
damage.noiseadd(newsamples, 0.7, 0.3)

spline.spline(samples, newsamples)

# Debe haber una mejor manera de hacer las cosas mas sencillas, pero solo me funciono asi.
tmp = samples.tolist()
samples = np.array(tmp*5)
tmp = newsamples.tolist()
newsamples = np.array(tmp*5)

# Crea los arreglos de "aciertos", basados en % que tan parecidos son.
matches = recognize.cheat(samples, newsamples, false_positives=0.04, false_negatives=0.1)
matchesSD = recognize.cheat(samples, samples, false_positives=0.04, false_negatives=0.1)
xSD, ySD = utils.tovalidxy(samples, matchesSD)
x, y = utils.tovalidxy(newsamples, matches)
x = np.array(x).reshape((-1, 1))

y = np.array(y)

# Inicializa variables para el proceso.
number = 10 #Number of segments
x_poly = []
y_poly_pred = []
sorted_zip = []
k = 0
xAux = []
yAux = []
x_polyAux = []
y_poly_predAux = []
i = 0 
countAux = 0

# Divisiones para x y para "y"
divX = divider.divider(x,number * 5)
divY = divider.divider(y,number * 5)
ySD = divider.divider(ySD,number * 5)

while k < 5:
    total = number * k
    counter = 0
    while counter < number:

        if k == 4: 
            polynomial_features = PolynomialFeatures(degree=5)
        elif k == 3:
            polynomial_features = PolynomialFeatures(degree=4)
        elif k == 2:
            polynomial_features = PolynomialFeatures(degree=3)
        elif k == 1:
            polynomial_features = PolynomialFeatures(degree=2)
        elif k == 0:
            polynomial_features = PolynomialFeatures(degree=1)

        model = LinearRegression()
        x_poly.append(polynomial_features.fit_transform(divX[counter + total]))
        model.fit(x_poly[counter + total], divY[counter + total])
        y_poly_pred.append(model.predict(x_poly[counter + total]))
        sort_axis = operator.itemgetter(0)
        sorted_zip.append(sorted(zip(divX[counter + total],y_poly_pred[counter + total]), key=sort_axis))
        divX[counter + total], y_poly_pred[counter + total] = zip(*sorted_zip[counter + total])
        if counter != 0:
            auxX = divX[counter + total]
            lastAuxX = auxX[0:3]
            auxY = y_poly_pred[counter]
            lastAuxY = auxY[0:3]
            auxX2 = divX[counter - 1 + total]
            initAuxX = auxX2[-3:]
            auxY2 = y_poly_pred[counter-1 + total]
            initAuxY = auxY2[-3:]
            xAux.append(initAuxX)
            yAux.append(initAuxY)
            xAux.append(lastAuxX)
            yAux.append(lastAuxY)
        counter = counter + 1
    
    k = k + 1
   
xAux = list(itertools.chain.from_iterable(xAux))
yAux = list(itertools.chain.from_iterable(yAux))

while countAux < (len(xAux)-5):

    tmp = countAux/(len(xAux)-5) * 100

    if 0 <= tmp < 20:
        polynomial_features= PolynomialFeatures(degree=1)
    if 20 <= tmp < 40:
        polynomial_features= PolynomialFeatures(degree=2)
    if 40 <= tmp < 60:
        polynomial_features= PolynomialFeatures(degree=3)
    if 60 <= tmp < 80:
        polynomial_features= PolynomialFeatures(degree=4)
    if 80 <= tmp < 100:
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

# Real
plt.plot(samples, label='real', color = "black")
plt.legend(loc='best')
ax = plt.axes()
ax.xaxis.grid()

# Resultado
while  counter < number * k :
    plt.plot(divX[counter], y_poly_pred[counter], color='y')
    counter = counter + 1

# print("Division en ", number, " segmentos")

for s in range(len(x_poly)):
    rmse = np.sqrt(mean_squared_error(ySD[s],y_poly_pred[s]))
    r2 = r2_score(ySD[s],y_poly_pred[s])
    # print("RMSE: ",rmse,"  R2: ",r2)
plt.show()

# Polinomial segmentado
polinomial.method(1000)

# Fourier
## TESTING
somearray = np.fft.rfftn(newsamples)
counter = 0
# print(somearray)

plt.plot(somearray, color='black')

plt.show()
## TESTING
#exit()