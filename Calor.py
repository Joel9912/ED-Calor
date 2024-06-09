import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

pm = 1000 #calor medio
pn = 50000 #calor máximo

def condini(x, y):
    z = np.zeros_like(x) #iniciamos con calor 0
    borde = (x < 0.05) | (x > 1.95) | (y < 0.05) | (y > 1.95)
    anillo = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.9 #generamos un anillo de calor
    centro1 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.16 #la zona central sea caliente
    centro2 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.13 #varios anillos para hacer una transición
    centro3 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.1
    centro4 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.07
    z[borde] = 0
    z[anillo] = pm / 2 #calor del anillo

    puntos_calor = 25  # Número de puntos de calor
    indices_calor = np.random.choice(np.flatnonzero(z == pm / 2), puntos_calor, replace=False)
    for index in indices_calor:
        i, j = np.unravel_index(index, z.shape)
        # Generarmos una máscara circular de 4x4 alrededor del punto de calor
        mask_i, mask_j = np.ogrid[-2:3, -2:3]  # Modificado el rango de la máscara
        mask = mask_i ** 2 + mask_j ** 2 <= 4
        slice_i_ini = max(0, i - 2)
        slice_i_fin = min(z.shape[0], i + 3)
        slice_j_ini = max(0, j - 2)
        slice_j_fin = min(z.shape[1], j + 3)
        z[slice_i_ini:slice_i_fin, slice_j_ini:slice_j_fin][
            mask[:slice_i_fin - slice_i_ini, :slice_j_fin - slice_j_ini]] = 40 * pm #que el calor de cada punto sea 40 veces pm

    z[centro1] = pn/4 #hacemos la transición de calor en el punto central
    z[centro2] = pn/2
    z[centro3] = pn/1.5
    z[centro4] = pn

    return z



def act_estado(Z):
    nx, ny = Z.shape
    Z_nuevo = np.copy(Z) #se realiza una copia para poder modificar el estado

    for i in range(nx):
        for j in range(ny):
            vecinos = [Z[i, j]]  # Siempre se incluye la casilla central

            if i > 0:
                vecinos.append(Z[i-1, j])  # Arriba
            if i < nx-1:
                vecinos.append(Z[i+1, j])  # Abajo
            if j > 0:
                vecinos.append(Z[i, j-1])  # Izquierda
            if j < ny-1:
                vecinos.append(Z[i, j+1])  # Derecha

            # Calcular el nuevo estado basado en los vecinos
            Z_nuevo[i, j] = np.mean(vecinos)  # Actualizamos usando el promedio de los vecinos

    return Z_nuevo

#Generación de la malla y sus condiciones iniciales

x = np.linspace(0, 2, 81)
y = np.linspace(0, 2, 81)
X,Y = np.meshgrid(x,y)
Z = condini(X,Y)

#Guardamos las actualizaciones en una cartpeta personalizada
output_dir = 'D:\\Usuario\\Documentos\\Cursos UCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

#primera imagen con las condiciones iniciales

fig = plt.figure()
ax = fig.subplots()

c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
fig.colorbar(c)
plt.ylim(-0.0125, 2.0125)
plt.xlim(-0.0125, 2.0125)

plt.savefig(os.path.join(output_dir, f'iteracion_{0}.png'))
plt.close(fig)

#iteramos para calcular las nuevas temperaturas

n = 100
for k in range(n):
    Z = act_estado(Z)

    fig = plt.figure()
    ax = fig.subplots()

    c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    fig.colorbar(c)
    plt.ylim(-0.0125, 2.0125)
    plt.xlim(-0.0125, 2.0125)

    plt.savefig(os.path.join(output_dir, f'iteracion_{k+1}.png'))
    plt.close(fig)
