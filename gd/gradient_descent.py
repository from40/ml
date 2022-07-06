# gradient descent with variable lambda

import time
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x) + 0.5*x


def df(x):
    return np.cos(x) + 0.5


# формуруем дипазон изменения аргумента функции f(x)
x_plt = np.arange(-5, 8.0, 0.01)

# формируем функцию f(x) для визуализации процесса
f_plt = [f(x) for x in x_plt]

# параметры алгоритма градиентного спуска
n = 0     # число итераций
r = 6     # начальное значение
lmd = 0.1  # шаг сходимости (learning rate)
min_limit = 100

# настраиваем графики
plt.ion()  # включение интерактивного режима отображения графиков
fig, ax = plt.subplots()  # создание окна и осей для графика
ax.grid(True)  # отображение сетки на графике
ax.plot(x_plt, f_plt)  # отобразим график функции и нашу точку на графике
point = ax.scatter(r, f(r), c="red")

# запуск алгоритма градиентного спуска
while df(r) > 0.0001 or df(r) < -0.0001:
    lmd = 1 / min(n + 1, min_limit)
    r = r - lmd * np.sign(df(r))  # пересчет позиции точки
    point.set_offsets([r, f(r)])  # отобразить следующее значение точки на графике
    n += 1

    # перерисовка графика с задержкой в 50 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.2)

plt.ioff()
print(r)
print(n)
ax.scatter(r, f(r), c="blue")
plt.show()
