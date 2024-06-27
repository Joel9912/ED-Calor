import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import imageio.v2 as imageio
from numba import njit, prange
import time

inicio_tiempo_global = time.time() #para calcular el tiempo de ejecución del código
# Constantes
cmax = 500 #condicion inicial borde a 500k
cc = 50 #condición inicial contiguo al borde a 20k
const_diff = 1.0e-4 #constante de difusión
m = 500 #tamaño de la malla
dx = dy = 1 / m #diferencial de espacio
dt = (0.2 * dx**2 ) / const_diff  #diferencial de tiempo en función del tamaño de la malla
# y de la constante de difusión para evitar errores en el cálculo es decir que const_diif*dt/dx**2 <= 0.2
n = 40000 #número de iteraciones
l = n / 150 #número de imágenes en función del número de iteraciones
# se ajusta para que siempre sean 100 imágenes más la imagen de la condición inicial

# Función para condiciones iniciales
def condini(shape):
    z = np.zeros(shape)
    z[:, 1] = z[1, :] = z[-2, :] = cc
    z[:, -1] = 0
    z[:, 0] = z[0, :] = z[-1, :] = cmax
    return z

# Función para actualizar la distribución de calor utilizando numba
@njit(parallel=True)
def act_calor(Z):
    Z_nuevo = np.copy(Z)
    for i in prange(1, Z.shape[0] - 1):
        for j in prange(1, Z.shape[1] - 1):
            Z_nuevo[i, j] = Z[i, j] + (const_diff * dt / dx ** 2) * (
                    Z[(i + 1) % m, j] + Z[(i - 1) % m, j] + Z[i, (j + 1) % m] + Z[i, (j - 1) % m] - 4 * Z[i, j]
            )
    Z_nuevo[:, -1] = 0
    Z_nuevo[:, 0] = Z_nuevo[0, :] = Z_nuevo[-1, :] = cmax
    return Z_nuevo

# Generación de la malla y sus condiciones iniciales
a = b = 1
x = np.linspace(0, a, m)
y = np.linspace(0, b, m)
X, Y = np.meshgrid(x, y)
Z = condini(X.shape)

# Directorio de salida de las imágenes
output_dir = 'D:\\Usuario\\Documentos\\CursosUCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

def save_imagen(Z, iteration):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    c = ax.pcolor(X, Y, Z, cmap=cm.jet, shading='auto', vmin=0, vmax=cmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c)
    plt.ylim(0 - 0.9 * dy, b + dy)
    plt.xlim(0 - dx, a + dx)  
    plt.savefig(os.path.join(output_dir, f'iteracion_{iteration}.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# Guardar la imagen inicial
save_imagen(Z, 0)

# Iterar para actualizar y guardar imágenes
for k in range(1, n + 1):
    Z = act_calor(Z)
    if k % l == 0:
        save_imagen(Z, k)

# Creación del GIF

# Crear lista de imágenes en el orden deseado
imagenes = []
for i in range(0, n + 1, int(l)):  # Solo cargar imágenes guardadas cada l iteraciones
    imagen = os.path.join(output_dir, f'iteracion_{i}.png')
    if os.path.exists(imagen):
        imagenes.append(Image.open(imagen))

# Guardamos el GIF incrementalmente para ahorrar memoria
gif_path = os.path.join(output_dir, 'difusion_termica.gif')
imagenes[0].save(gif_path, save_all=True, append_images=imagenes[1:], duration=100, loop=0, quality=95)

# Terminamos con el cálculo de ejecución del código
fin_tiempo_global = time.time()
print(f"Tiempo de ejecución del programa: {fin_tiempo_global-inicio_tiempo_global} segundos")
