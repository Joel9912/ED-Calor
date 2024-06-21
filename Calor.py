import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

cc = 50  # calor contiguo
cmax = 300  # calor máximo
const_diff = 1.0e-4  # constante de difusividad térmica
dt = 0.2  # paso de tiempo
dx = dy = 1 / 100  # tamaño de la celda

def condini(x, y):
    z = np.zeros_like(x)  # se incia con un calor, por ejemplo con calor 0

    # Línea de calor 0
    z[:, -1] = 0  # borde derecho

    # Líneas contiguas de calor bajo
    z[:, 1] = cc  # linea contigua al borde izquierdo
    z[1, :] = cc  # linea contigua al borde inferior
    z[-2, :] = cc  # línea contigua al borde superior

    # Líneas de calor en los bordes
    z[:, 0] = cmax  # borde izquierdo
    z[0, :] = cmax  # borde inferior
    z[-1, :] = cmax  # borde superior

    return z

def act_calor(Z):
    nx, ny = Z.shape
    Z_nuevo = np.copy(Z)  # se realiza una copia para poder modificar el estado

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            sum_diff = (
                Z[i-1, j] + Z[i+1, j] + Z[i, j-1] + Z[i, j+1]
            )
            Z_nuevo[i, j] = Z[i, j] + dt * const_diff * (
                (sum_diff - 4 * Z[i, j]) / (dx**2) # también serviría dy
            )
    # Mantenemos las condiciones iniciales de los bordes

    # Línea de calor 0
    Z_nuevo[:, -1] = 0  # borde derecho

    # Líneas de calor en los bordes
    Z_nuevo[:, 0] = cmax  # borde izquierdo
    Z_nuevo[0, :] = cmax  # borde inferior
    Z_nuevo[-1, :] = cmax  # borde superior

    return Z_nuevo

# Generación de la malla y sus condiciones iniciales

a = b = 1
x = np.linspace(0, a, 101)
y = np.linspace(0, b, 101)
X, Y = np.meshgrid(x, y)
Z = condini(X, Y)

# Directorio de salida de las imágenes

output_dir = 'D:\\Usuario\\Documentos\\CursosUCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

# Imagen inicial con las condiciones iniciales

fig = plt.figure(figsize=(8, 6), dpi=400)
ax = fig.add_subplot(111)

c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.05, cmap=cm.jet, shading='auto', vmin=0, vmax=cmax)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
fig.colorbar(c) # en c definimos la barra para que siempre tenga los valores máximos (cmax=300) y mínimos (0)
#Los límites son algo mayores para que la cuadrícula quede bien
plt.ylim(-0.005, 1.005)
plt.xlim(-0.005, 1.005)
# Creamos el archivo con la imagen
plt.savefig(os.path.join(output_dir, f'iteracion_{0}.png'), bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

# Ahora iteramos para obtener las nuevas imágenes con las nuevas temperaturas

n = 250
for k in range(n):
    Z = act_calor(Z)  # Actualizar el calor de cada celda

    # Guardamos la figura de cada iteración
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = fig.add_subplot(111)

    c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.05, cmap=cm.jet, shading='auto', vmin=0, vmax=cmax)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    fig.colorbar(c) # en c definimos la barra para que siempre tenga los valores máximos (cmax=300) y mínimos (0)
    # Los límites son algo mayores para que la cuadrícula quede bien
    plt.ylim(-0.005, 1.005)
    plt.xlim(-0.005, 1.005)
    # Creamos el archivo con la imagen
    plt.savefig(os.path.join(output_dir, f'iteracion_{k+1}.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
