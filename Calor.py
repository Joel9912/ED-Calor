import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio.v2 as imageio

cc = 20  # calor contiguo
cmax = 500  # calor máximo
const_diff = 1.0e-4  # constante de difusividad térmica
dt = 0.032  # paso de tiempo
m = 250
dx = dy = 1 / m  # tamaño de la celda

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
            Z_nuevo[i,j] = Z[i,j] + (const_diff * dt / dx**2) * (
                    Z[i+1,j]+Z[i-1,j]+Z[i,j+1]+Z[i,j-1]-4*Z[i,j])
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
x = np.linspace(0, a, int(m+m/100))
y = np.linspace(0, b, int(m+m/100))
X, Y = np.meshgrid(x, y)
Z = condini(X, Y)

# Directorio de salida de las imágenes

output_dir = 'D:\\Usuario\\Documentos\\CursosUCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

# Imagen inicial con las condiciones iniciales

fig = plt.figure(figsize=(8, 6), dpi=1200)
ax = fig.add_subplot(111)

c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.05, cmap=cm.jet, shading='auto', vmin=0, vmax=cmax)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
fig.colorbar(c) # en c definimos la barra para que siempre tenga los valores máximos (cmax=300) y mínimos (0)
#Los límites son algo mayores para que la cuadrícula quede bien
plt.ylim(0 - 0.8*dy, b + 0.8*dy)
plt.xlim(0 - 0.8*dx, a + 0.8*dx)
# Creamos el archivo con la imagen
plt.savefig(os.path.join(output_dir, f'iteracion_{0}.png'), bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

# Ahora iteramos para obtener las nuevas imágenes con las nuevas temperaturas

n = 3000
# Guardamos las figuras intermedias
for k in range(n):
    Z = act_calor(Z)
    if (k + 1) % 10 == 0:  # Guardar cada 10 iteraciones
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = fig.add_subplot(111)
        c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.05, cmap=cm.jet, shading='auto', vmin=0, vmax=cmax)
        ax.set_xlabel("x", fontsize=15)
        ax.set_ylabel("y", fontsize=15)
        fig.colorbar(c)
        plt.ylim(0 - 0.8*dy, b + 0.8*dy)
        plt.xlim(0 - 0.8*dx, a + 0.8*dx)
        plt.savefig(os.path.join(output_dir, f'iteracion_{k+1}.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

# Creamos una lista de imágenes en orden
imagenes = []
for i in range(n + 1):  # +1 para incluir desde iteracion_0 hasta iteracion_n
    imagen_path = os.path.join(output_dir, f'iteracion_{i}.png')
    if os.path.exists(image_path):
        imagenes.append(imageio.imread(imagen_path))

# Ruta donde se guardará el GIF
gif_path = os.path.join(output_dir, 'simulacion_termica.gif')

# Creamos el GIF animado
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for frame in imagenes:
        writer.append_data(frame)
