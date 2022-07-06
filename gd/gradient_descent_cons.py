# gradient descent with constant lambda

import time
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x


def df(x):
    return 2*x


# параметры алгоритма градиентного спуска
n = 35       # число итераций
r = 1024     # начальное значение
lmd = 1 / 4  # шаг сходимости (learning rate)

# формуруем дипазон изменения аргумента функции f(x)
x_plt = np.arange(0, 2048, 1)

# формируем функцию f(x) для визуализации процесса
f_plt = [f(x) for x in x_plt]

# настраиваем графики
plt.ion()  # включение интерактивного режима отображения графиков
fig, ax = plt.subplots()  # создание окна и осей для графика
ax.grid(True)  # отображение сетки на графике
ax.plot(x_plt, f_plt)  # отобразим график функции и нашу точку на графике
point = ax.scatter(r, f(r), c="red")

# запуск алгоритма градиентного спуска
for i in range(n):
    r = r - lmd * df(r)  # пересчет позиции точки
    point.set_offsets([r, f(r)])  # отобразить следующее значение точки на графике

    # перерисовка графика с задержкой в 50 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.3)

plt.ioff()
print(r)
ax.scatter(r, f(r), c="blue")
plt.show()
