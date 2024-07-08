import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
import imageio
import tkinter as tk
from tkinter import ttk, messagebox

# Función para generar la función f de forma que se pueda usar
def generar_funcion(cond_ini):
    # Genera una función f(x, y) a partir de las condiciones iniciales suministradas en la ventana
    codigo = f"""def f(x, y):
    return {cond_ini}"""
    local_vars = {}
    exec(codigo, globals(), local_vars)  # Ejecuta el código para definir la función f
    return jit(nopython=True)(local_vars['f'])  # Se usa Numba para mejorar la eficiencia

# Función para calcular B0n
@jit(nopython=True)
def calcB0n(a, b, g, n, f):
    # Calculamos el coeficiente B0n de la serie de Fourier para una función f(x, y)
    integral = 0.0
    dx = a / g
    dy = b / g
    for i in range(g):
        x = dx * (i + 0.5)
        for j in range(g):
            y = dy * (j + 0.5)
            val_f = f(x, y)
            integral += val_f * np.sin(n * np.pi * y / b) * dx * dy
    return 4 * integral / (a * b)

# Función para calcular Bmn
@jit(nopython=True)
def calcBmn(a, b, g, m, n, f):
    integral = 0.0
    dx = a / g
    dy = b / g
    for i in range(g):
        x = dx * (i + 0.5)
        for j in range(g):
            y = dy * (j + 0.5)
            val_f = f(x, y)
            integral += val_f * np.sin(n * np.pi * y / b) * np.cos(m * np.pi * x / a) * dx * dy
    return 4 * integral / (a * b)

# Función para precalcular los coeficientes B0n y Bmn
def precalc_inte(a, b, g, iteraciones, f):
    B0n_n = np.zeros(iteraciones)
    Bmn_mn = np.zeros((iteraciones, iteraciones))
    for n in range(1, iteraciones + 1):
        B0n_n[n - 1] = calcB0n(a, b, g, n, f)
    for m in range(1, iteraciones + 1):
        for n in range(1, iteraciones + 1):
            Bmn_mn[m - 1][n - 1] = calcBmn(a, b, g, m, n, f)
    return B0n_n, Bmn_mn

# Función para calcular la onda de calor en 2D
def onda2D(a, b, g, iteraciones, k, t, B0n_n, Bmn_mn):
    x = np.linspace(0, a, g)
    y = np.linspace(0, b, g)
    u = np.zeros((g, g))
    for n in range(1, iteraciones + 1):
        B0n = B0n_n[n - 1]
        u += 0.5 * B0n * np.exp(-(n ** 2 / b ** 2) * np.pi ** 2 * k * t) * np.sin(n * np.pi * y[:, None] / b)
        # convierte y a una matriz bidimensional lo que mejora el cálculo al usar Broadcasting de Numpy
    for m in range(1, iteraciones + 1):
        for n in range(1, iteraciones + 1):
            Bmn = Bmn_mn[m - 1][n - 1]
            u += Bmn * np.exp(-(m ** 2 / a ** 2 + n ** 2 / b ** 2) * np.pi ** 2 * k * t) \
                 * np.sin(n * np.pi * y[:, None] / b) * np.cos(m * np.pi * x[None, :] / a) #igual para x e y
    return x, y, u

# Función para generar las gráficas de la onda de calor a través del tiempo
def generar_imagenes(a, b, g, k, t_inicial, t_final, num_pasos, iteraciones, progreso_barra, vmax, f):
    B0n_n, Bmn_mn = precalc_inte(a, b, g, iteraciones, f)
    imagenes = []
    for step in range(num_pasos + 1):
        t = t_inicial + step * (t_final - t_inicial) / num_pasos
        x, y, Z = onda2D(a, b, g, iteraciones, k, t, B0n_n, Bmn_mn)
        fig, ax = plt.subplots()
        c = ax.contourf(x, y, Z, cmap='viridis', levels=20, vmin=0, vmax=vmax)
        fig.colorbar(c)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Onda de calor en t={t:.2f}s')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        imagenes.append(image)
        plt.close(fig)
        progreso = (step / num_pasos) * 100
        progreso_barra.set(progreso)
        vent.update_idletasks()
        print(f"Progreso: {progreso:.1f}%") # Para tener el progreso en la terminal al generar las imágnes
    return imagenes

# Función para generar el GIF
def generar_gif():
    try:
        # Para obtener los parámetros de la interfaz
        a, b, k = float(entradas[0].get()), float(entradas[1].get()), float(entradas[2].get())
        t_inicial, t_final = float(entradas[3].get()), float(entradas[4].get())
        g, iteraciones, num_pasos, vmax = 100, 100, 50, 0  # Tamaño de la malla, iteraciones y número de pasos
        cond_ini = entradas[5].get()  # Condiciones iniciales para la función f(x, y)
        # Generamos la función f(x, y), calculamos los valores iniciales y el valor máximo para las gráficas
        f = generar_funcion(cond_ini)
        x = np.linspace(0, a, g)
        y = np.linspace(0, b, g)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        vmax = np.max(Z)
        # Generamos las imágenes
        imagenes = generar_imagenes(a, b, g, k, t_inicial, t_final, num_pasos, iteraciones, progreso_barra, vmax, f)
        # Guardamos el GIF en el directorio 'gifs', si no existe se crea
        output_dir = os.path.join(os.getcwd(), 'gifs')
        os.makedirs(output_dir, exist_ok=True)
        gif_name = f'Difusion Termica para una placa de dimensiones {a, b},' \
                   f' constante de difusión k={k} desde t={t_inicial:.2f}s a t={t_final:.2f}s.gif'
        imageio.mimsave(os.path.join(output_dir, gif_name), imagenes, fps=4, loop=0)
        messagebox.showinfo("Éxito", f"GIF generado y guardado en: {os.path.join(output_dir, gif_name)}")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

# Ahora usamos Tkinter para toda la interfaz gráfica

# Creamos la interfaz gráfica con las etiquetas y entradas para los parámetros
vent = tk.Tk()
vent.title("Interfaz Generadora Onda Calor")
mainframe = ttk.Frame(vent, padding="10 10 10 10")
mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
etiquetas = ["a (Largo de la placa):", "b (Ancho de la placa):", "k (Constante de difusión):",
             "Tiempo inicial:", "Tiempo final:", "Condiciones iniciales:"]
entradas = []
for i, etiqueta in enumerate(etiquetas):
    ttk.Label(mainframe, text=etiqueta).grid(row=i, column=0, sticky=tk.W)
    entrada = ttk.Entry(mainframe)
    entrada.grid(row=i, column=1)
    entradas.append(entrada)

# Barra de progreso para mostrar el avance de la generación de imágenes
progreso_barra = tk.DoubleVar()
barra_progreso = ttk.Progressbar(mainframe, variable=progreso_barra, maximum=100)
barra_progreso.grid(row=len(etiquetas), column=0, columnspan=2, sticky=(tk.W, tk.E))

# Botón para generar el GIF
ttk.Button(mainframe, text="Generar GIF", command=generar_gif).grid(row=len(etiquetas) + 1, column=0, columnspan=2)

# Ejecutar la interfaz gráfica
vent.mainloop()
