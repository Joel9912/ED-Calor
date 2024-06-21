import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

pm = 50  # calor bajo
pn = 300  # calor máximo
diffusion_const = 1.0e-4  # difusividad térmica para el acero en m^2/s
delta_t = 0.2  # paso de tiempo
delta_x = delta_y = 1 / 100  # tamaño de la celda espacial

def condini(x, y):
    z = np.zeros_like(x)  # iniciamos con calor 0

    # Líneas contiguas de calor bajo
    z[:, -1] = 0  # borde derecho

    # Líneas de calor en los bordes
    z[:, 0] = pn  # borde izquierdo
    z[0, :] = pn  # borde inferior
    z[-1, :] = pn  # borde superior

    return z

def act_estado(Z):
    nx, ny = Z.shape
    Z_nuevo = np.copy(Z)  # se realiza una copia para poder modificar el estado

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            sum_diff = (
                Z[i-1, j] + Z[i+1, j] + Z[i, j-1] + Z[i, j+1]
            )
            Z_nuevo[i, j] = Z[i, j] + delta_t * diffusion_const * (
                (sum_diff - 4 * Z[i, j]) / (delta_x**2)
            )

    # Líneas contiguas de calor bajo
    Z_nuevo[:, -1] = 0  # borde derecho

    # Líneas de calor en los bordes
    Z_nuevo[:, 0] = pn  # borde izquierdo
    Z_nuevo[0, :] = pn  # borde inferior
    Z_nuevo[-1, :] = pn  # borde superior

    return Z_nuevo

# Generación de la malla y sus condiciones iniciales
a = b = 2

x = np.linspace(0, a, 101)
y = np.linspace(0, b, 101)
X, Y = np.meshgrid(x, y)
Z = condini(X, Y)

# Crear directorio de salida si no existe
output_dir = 'D:\\Usuario\\Documentos\\CursosUCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

# Primera imagen con las condiciones iniciales
fig = plt.figure(figsize=(8, 6), dpi=400)
ax = fig.add_subplot(111)

color_plot = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.1, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
fig.colorbar(color_plot)
plt.ylim(-0.01, 2.01)
plt.xlim(-0.01, 2.01)

plt.savefig(os.path.join(output_dir, f'iteracion_{0}.png'), bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

# Iterar para calcular las nuevas temperaturas
n = 200
for k in range(n):
    Z[1:-1, 1:-1] = act_estado(Z[1:-1, 1:-1])  # Actualizar el estado sin modificar los bordes

    # Guardar la figura de cada iteración
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = fig.add_subplot(111)

    color_plot = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.1, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    fig.colorbar(color_plot)
    plt.ylim(-0.01, 2.01)
    plt.xlim(-0.01, 2.01)

    plt.savefig(os.path.join(output_dir, f'iteracion_{k+1}.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
