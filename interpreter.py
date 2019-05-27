import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
main_plot = fig.add_subplot(1,1,1)

def animate(i):
    with open('commands.txt', 'r') as data:
        lines = data.read().split('\n')
        X = []
        Y = []
        X.append(0)
        Y.append(1)
        for line in lines:
            if(len(line) > 1):
                delta_x, delta_y = line.split(',')
                X.append(X[-1] + delta_x)
                Y.append(Y[-1] + delta_y)
        main_plot.clear()
        main_plot.plot(X, Y)


animation.FuncAnimation(fig, animate, interval=1000)
plt.show()