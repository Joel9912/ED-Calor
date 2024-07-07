import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
import imageio
import tkinter as tk
from tkinter import ttk, messagebox


# Definición de la condición inicial f(x, y)
@jit(nopython=True)
def f(x, y):
    return x ** 2 + y ** 2

# Función para calcular B0n y Bmn
@jit(nopython=True)
def calcB(a, b, g, m=0, n=0, isB0n=False):
    integral = 0.0
    for i in range(g):
        for j in range(g):
            x, y = a * (i + 0.5) / g, b * (j + 0.5) / g
            val_f = f(x, y)
            if isB0n:
                integral += val_f * np.sin(n * np.pi * y / b) * a * b / g / g
            else:
                integral += val_f * np.sin(n * np.pi * y / b) * np.cos(m * np.pi * x / a) * a * b / g / g
    return 4 * integral / (a * b)

# Función para precalcular los coeficientes
def precalc_inte(a, b, g, iteraciones):
    B0n_n = np.array([calcB(a, b, g, n=n, isB0n=True) for n in range(1, iteraciones + 1)])
    Bmn_mn = np.array([[calcB(a, b, g, m=m, n=n) for n in range(1, iteraciones + 1)] for m in range(1, iteraciones + 1)])
    return B0n_n, Bmn_mn

# Función para calcular la onda de calor en 2D
def onda2D(a, b, g, iteraciones, k, t, B0n_n, Bmn_mn):
    x, y = np.linspace(0, a, g), np.linspace(0, b, g)
    X, Y = np.meshgrid(x, y)
    u = sum(0.5 * B0n_n[n - 1] * np.exp(-(n ** 2 / b ** 2) * np.pi ** 2 * k * t) * np.sin(n * np.pi * Y / b) for n in range(1, iteraciones + 1))
    u += sum(Bmn_mn[m - 1][n - 1] * np.exp(-(m ** 2 / a ** 2 + n ** 2 / b ** 2) * np.pi ** 2 * k * t) *
             np.sin(n * np.pi * Y / b) * np.cos(m * np.pi * X / a) for m in range(1, iteraciones + 1) for n in range(1, iteraciones + 1))
    return X, Y, u

# Función para generar imágenes
def generar_imagenes(a, b, g, k, t_inicial, t_final, num_pasos, iteraciones, progress_var):
    B0n_n, Bmn_mn = precalc_inte(a, b, g, iteraciones)
    imagenes = []
    for step in range(num_pasos + 1):
        t = t_inicial + step * (t_final - t_inicial) / num_pasos
        X, Y, Z = onda2D(a, b, g, iteraciones, k, t, B0n_n, Bmn_mn)
        fig, ax = plt.subplots()
        c = ax.contourf(X, Y, Z, cmap='viridis', levels=20, vmin=0, vmax=f(a, b))
        fig.colorbar(c)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Onda de calor en t={t:.2f}s')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imagenes.append(image)
        plt.close(fig)
        progreso = (step / num_pasos) * 100
        progress_var.set(progreso)
        root.update_idletasks()
        print("progreso:",progreso)
    return imagenes

# Función para generar el GIF
def generar_gif():
    try:
        a, b, k = float(entry_a.get()), float(entry_b.get()), float(entry_k.get())
        t_inicial, t_final = float(entry_t_inicial.get()), float(entry_t_final.get())
        g, iteraciones, num_pasos = 100, 100, 50
        imagenes = generar_imagenes(a, b, g, k, t_inicial, t_final, num_pasos, iteraciones, progress_var)
        output_dir = os.path.join(os.getcwd(), 'gifs')
        os.makedirs(output_dir, exist_ok=True)
        gif_name = f'Difusion Termica para una placa de dimensiones {a,b}, constante de difusión k={k} desde t={t_inicial:.2f}s a t={t_final:.2f}s.gif'
        imageio.mimsave(os.path.join(output_dir, gif_name), imagenes, fps=4, loop=0)
        messagebox.showinfo("Éxito", f"GIF generado y guardado en: {os.path.join(output_dir, gif_name)}")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Simulación de Difusión Térmica")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(mainframe, text="a:").grid(row=0, column=0, sticky=tk.W)
entry_a = ttk.Entry(mainframe)
entry_a.grid(row=0, column=1)

ttk.Label(mainframe, text="b:").grid(row=1, column=0, sticky=tk.W)
entry_b = ttk.Entry(mainframe)
entry_b.grid(row=1, column=1)

ttk.Label(mainframe, text="k:").grid(row=2, column=0, sticky=tk.W)
entry_k = ttk.Entry(mainframe)
entry_k.grid(row=2, column=1)

ttk.Label(mainframe, text="Tiempo inicial:").grid(row=3, column=0, sticky=tk.W)
entry_t_inicial = ttk.Entry(mainframe)
entry_t_inicial.grid(row=3, column=1)

ttk.Label(mainframe, text="Tiempo final:").grid(row=4, column=0, sticky=tk.W)
entry_t_final = ttk.Entry(mainframe)
entry_t_final.grid(row=4, column=1)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(mainframe, variable=progress_var, maximum=100)
progress_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))

ttk.Button(mainframe, text="Generar GIF", command=generar_gif).grid(row=6, columnspan=2)

root.mainloop()
