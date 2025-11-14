import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time

def animate_optimization(history, function, x_range=(-5,5), y_range=(-5,5)):
    history = np.array(history)
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = function([X, Y])

    fig = plt.figure(figsize=(12, 5))

    # 3d
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    point3d, = ax1.plot([], [], [], 'ro', markersize=6)
    ax1.set_title('3D Surface View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X,Y)')

    # Contour map
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
    point2d, = ax2.plot([], [], 'ro', markersize=5)
    ax2.set_title('Contour Map View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    def update(frame):
        x, y, z = history[frame]
        point3d.set_data([x], [y])
        point3d.set_3d_properties([z])
        point2d.set_data([x], [y])

        ax1.set_title(f'3D View: {frame * 10}')
        ax2.set_title(f'Contour View: {frame * 10}')

        return point3d, point2d
    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=500, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()


def animate_genetic(history, function, x_range=(-5,5), y_range=(-5,5)):
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function([X[i,j], Y[i,j]])
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3d
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, antialiased=True)
    scat3d = ax1.scatter([], [], [], c='red', s=30, depthshade=False)
    ax1.set_title('3D Surface - Population Evolution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X,Y)')
    
    # Contour map
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='f(X,Y)')
    scat2d = ax2.scatter([], [], c='red', s=30, edgecolors='black', linewidth=0.5)
    ax2.set_title('Contour Map - Population Evolution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    def update(frame):
        pop = history[frame]
        x, y = pop[:, 0], pop[:, 1]
        z = np.array([function(ind) for ind in pop])
        
        scat3d._offsets3d = (x, y, z)

        scat2d.set_offsets(np.c_[x, y])
        
        ax1.set_title(f'3D View - Generation')
        ax2.set_title(f'Contour View - Generation')
        
        return scat3d, scat2d

    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=800, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()


def animate_abc(history, function, x_range=(-5, 5), y_range=(-5, 5)):
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = function([X, Y])

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    points3d, = ax1.plot([], [], [], 'ro', markersize=4)
    ax1.set_title('3D Surface View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X,Y)')

    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
    points2d, = ax2.plot([], [], 'ro', markersize=4)
    ax2.set_title('Contour Map View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    def update(frame):
        swarm = np.array(history[frame])
        x_vals = swarm[:, 0]
        y_vals = swarm[:, 1]
        z_vals = swarm[:, 2]

        points3d.set_data(x_vals, y_vals)
        points3d.set_3d_properties(z_vals)

        points2d.set_data(x_vals, y_vals)
        return points3d, points2d

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()