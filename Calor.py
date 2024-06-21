import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

pm = 500  # calor bajo
pn = 10000  # calor máximo
diffusion_const = 1.2e-5  # difusividad térmica para el acero en m^2/s
delta_t = 0.5  # paso de tiempo, puede necesitar ajuste dependiendo de la estabilidad numérica
delta_x = delta_y = 2 / 160  # tamaño de la celda espacial

def condini(x, y):
    z = np.zeros_like(x)  # iniciamos con calor 0
    anillo = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.9  # generamos un anillo de calor
    centro1 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.15  # la zona central sea caliente
    centro2 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.12  # varios anillos para hacer una transición
    centro3 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.09
    centro4 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.06

    z[anillo] = pm  # calor del anillo
    z[centro1] = pn / 4  # hacemos la transición de calor en el punto central
    z[centro2] = pn / 2
    z[centro3] = pn / 1.5
    z[centro4] = pn

    return z

def act_estado(Z):
    nx, ny = Z.shape
    Z_nuevo = np.copy(Z)  # se realiza una copia para poder modificar el estado

    for i in range(nx):
        for j in range(ny):
            sum_diff = 0
            num_neighbors = 0

            if i > 0:
                sum_diff += Z[i-1, j]  # Arriba
                num_neighbors += 1
            if i < nx-1:
                sum_diff += Z[i+1, j]  # Abajo
                num_neighbors += 1
            if j > 0:
                sum_diff += Z[i, j-1]  # Izquierda
                num_neighbors += 1
            if j < ny-1:
                sum_diff += Z[i, j+1]  # Derecha
                num_neighbors += 1

            # Actualizar el estado utilizando la suma de diferencias y el número de vecinos
            if num_neighbors > 0:
                Z_nuevo[i, j] = Z[i, j] + delta_t * diffusion_const * (
                    (sum_diff - num_neighbors * Z[i, j]) / (delta_x**2)
                )

    return Z_nuevo

# Generación de la malla y sus condiciones iniciales
a = b = 2

x = np.linspace(0, a, 81)
y = np.linspace(0, b, 81)
X, Y = np.meshgrid(x, y)
Z = condini(X, Y)

# Guardamos las actualizaciones en una carpeta personalizada
output_dir = 'D:\\Usuario\\Documentos\\CursosUCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

# Primera imagen con las condiciones iniciales
fig = plt.figure()
ax = fig.subplots()

color_plot = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
fig.colorbar(color_plot)
plt.ylim(-0.0125, 2.0125)
plt.xlim(-0.0125, 2.0125)

plt.savefig(os.path.join(output_dir, f'iteracion_{0}.png'))
plt.close(fig)

# Iteramos para calcular las nuevas temperaturas
n = 200
for k in range(n):
    Z = act_estado(Z)

    fig = plt.figure()
    ax = fig.subplots()

    color_plot = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    fig.colorbar(color_plot)
    plt.ylim(-0.0125, 2.0125)
    plt.xlim(-0.0125, 2.0125)

    plt.savefig(os.path.join(output_dir, f'iteracion_{k+1}.png'))
    plt.close(fig)
