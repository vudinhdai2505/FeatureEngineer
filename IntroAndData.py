import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

x = np.array([1,2,3,4,5,6],dtype=np.float64)
y = np.array([2,3.5,5,7,9,13], dtype=np.float64)
def bet_fit_slope_and_intercept(x,y):
    m = (mean(x)*mean(y) - mean(x*y))/ (mean(x)*mean(x) - mean(x*x))
    b = mean(y) - m*mean(x)
    return m,b

m,b = bet_fit_slope_and_intercept(x,y)

regession_line = []

for X in x:
    regession_line.append(m*X+b)
print(regession_line)
plt.scatter(x,y, c="g")
plt.plot(x,regession_line)
plt.show()

