import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

pm = 1000
pn = 50000

def condini(x, y):
    z = np.zeros_like(x)
    borde = (x < 0.05) | (x > 1.95) | (y < 0.05) | (y > 1.95)
    anillo = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.9
    centro1 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.16
    centro2 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.13
    centro3 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.1
    centro4 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2) < 0.07
    z[borde] = 0
    z[anillo] = pm / 2

    num_heat_points = 25  # Número de puntos de calor
    heat_indices = np.random.choice(np.flatnonzero(z == pm / 2), num_heat_points, replace=False)
    for index in heat_indices:
        i, j = np.unravel_index(index, z.shape)
        # Generar una máscara circular de 4x4 alrededor del punto de calor
        mask_i, mask_j = np.ogrid[-2:3, -2:3]  # Modificado el rango de la máscara
        mask = mask_i ** 2 + mask_j ** 2 <= 4
        slice_i_start = max(0, i - 2)
        slice_i_end = min(z.shape[0], i + 3)
        slice_j_start = max(0, j - 2)
        slice_j_end = min(z.shape[1], j + 3)
        z[slice_i_start:slice_i_end, slice_j_start:slice_j_end][
            mask[:slice_i_end - slice_i_start, :slice_j_end - slice_j_start]] = 40 * pm

    z[centro1] = pn/4
    z[centro2] = pn/2
    z[centro3] = pn/1.5
    z[centro4] = pn

    return z



def update_state(Z):
    nx, ny = Z.shape
    Z_new = np.copy(Z)

    for i in range(nx):
        for j in range(ny):
            neighbors = [Z[i, j]]  # Always include the central cell

            if i > 0:
                neighbors.append(Z[i-1, j])  # Arriba
            if i < nx-1:
                neighbors.append(Z[i+1, j])  # Abajo
            if j > 0:
                neighbors.append(Z[i, j-1])  # Izquierda
            if j < ny-1:
                neighbors.append(Z[i, j+1])  # Derecha

            # Calcular el nuevo estado basado en los vecinos
            Z_new[i, j] = np.mean(neighbors)  # Ejemplo de actualización usando el promedio de los vecinos

    return Z_new


output_dir = 'D:\\Usuario\\Documentos\\Cursos UCR\\Física\\Física-Computacional\\Proyecto'
os.makedirs(output_dir, exist_ok=True)

x = np.linspace(0, 2, 81)
y = np.linspace(0, 2, 81)
X,Y = np.meshgrid(x,y)
Z = condini(X,Y)

fig = plt.figure()
ax = fig.subplots()

c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
fig.colorbar(c)
plt.ylim(-0.0125, 2.0125)
plt.xlim(-0.0125, 2.0125)

plt.savefig(os.path.join(output_dir, f'iteration_{0}.png'))
plt.close(fig)

num_iterations = 100
for iteration in range(num_iterations):
    Z = update_state(Z)

    fig = plt.figure()
    ax = fig.subplots()

    c = ax.pcolor(X, Y, Z, edgecolors='k', linewidths=0.2, cmap=cm.jet, shading='auto', vmin=0, vmax=pn)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    fig.colorbar(c)
    plt.ylim(-0.0125, 2.0125)
    plt.xlim(-0.0125, 2.0125)

    plt.savefig(os.path.join(output_dir, f'iteration_{iteration+1}.png'))
    plt.close(fig)
