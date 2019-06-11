#Импорт библиотек и функций
import os
import sys

current_dir_path = os.path.dirname(os.path.realpath(__file__))
project_root_path = os.path.join(current_dir_path, os.pardir, os. pardir)
sys.path.insert(0, project_root_path)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import constants as const

#Определение функции анимации графика
def animate(i):
    #Чтение файла с относительными координатами
    with open(const.COORDINATES_SOURCE_PATH, 'r') as data:
        lines = data.read().split('\n')

        #Последовательное формирование точек графика по данным из файла
        X = [0]
        Y = [0]
        for line in lines:
            if(len(line) > 1):
                delta_x, delta_y = line.split(' ')
                X.append(X[-1] + int(delta_x))
                Y.append(Y[-1] + int(delta_y))

        #Обновление графика
        main_plot.clear()
        main_plot.plot(X, Y)
        plt.xlabel('x')
        plt.ylabel('y')


if __name__ == '__main__':
    #Задание графика
    fig = plt.figure()
    main_plot = fig.add_subplot(1, 1, 1)
    plt.xlabel('x')
    plt.ylabel('y')

    #Задание функции анимации для графика с периодом обновления 1 секунда
    ani = animation.FuncAnimation(fig, animate, interval=1000)

    #Отображение графика
    plt.show()