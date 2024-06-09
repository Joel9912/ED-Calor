import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def posxy(x,y):
    if (x < 0.15):
        return 1
    if (x > 1.9):
        return 0
    if (y < 0.15):
        return 2
    if (y > 1.9):
        return 3
    else:
        return 4
def condini(x,y):
    z = np.ones_like(x)
    borde = (x < 0.15) | (x > 1.9) | (y < 0.15) | (y > 1.9)
    anillo = (x > 0.35) & (x < 1.7) & (y > 0.35) & (y < 1.7)
    centro = (x > 0.5) & (x < 1.55) & (y > 0.5) & (y < 1.55)
    z[borde] = 0
    z[anillo] = 2
    z[centro] = 3
    return z

x = np.linspace(0, 2, 41)
y = np.linspace(0, 2, 41)
X,Y = np.meshgrid(x,y)
Z = condini(X,Y)

fig2 = plt.figure()
ax2 = fig2.subplots()

c = ax2.pcolor(X, Y, Z,edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto')

ax2.set_xlabel("x", fontsize=15)
ax2.set_ylabel("y", fontsize=15)
fig2.colorbar(c)
plt.ylim(-0.025,2.025)
plt.xlim(-0.025,2.025)
plt.show()
