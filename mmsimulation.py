import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import odeint

# Set up the figure and layout
plt.figure(figsize=(12, 8))
plt.subplots_adjust(bottom=0.3)


# Kuramoto model equations
def kuramoto(theta, t, N, K, omega):
    dtheta = np.zeros(N)
    for i in range(N):
        dtheta[i] = omega[i] + (K / N) * np.sum(np.sin(theta - theta[i]))
    return dtheta


# Parameters
N = 50  # Number of oscillators
omega = np.random.normal(0, 1, N)  # Natural frequencies

# Initial conditions
theta0 = np.random.uniform(0, 2 * np.pi, N)

# Time points
t = np.linspace(0, 20, 1000)

# Create main plot axes
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 1, 2)

# Initial simulation with K=0
K = 0
theta = odeint(kuramoto, theta0, t, args=(N, K, omega))
theta = np.mod(theta, 2 * np.pi)  # Keep angles between 0 and 2pi

# Plot initial phase distribution
scat = ax1.scatter(np.cos(theta[-1]), np.sin(theta[-1]), c=theta[-1], cmap="hsv")
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_title("Phase Distribution on Unit Circle")
ax1.set_aspect("equal")

# Plot initial time series
lines = []
for i in range(min(5, N)):  # Only plot first 5 for clarity
    (line,) = ax2.plot(t, theta[:, i], alpha=0.7)
    lines.append(line)
ax2.set_title("Phase Time Series")
ax2.set_xlabel("Time")
ax2.set_ylabel("Phase")

# Plot order parameter over time
order_param = np.array([np.abs(np.mean(np.exp(1j * theta[i]))) for i in range(len(t))])
(order_line,) = ax3.plot(t, order_param)
ax3.set_title("Order Parameter (Synchrony Measure)")
ax3.set_xlabel("Time")
ax3.set_ylabel("|r| (0=async, 1=sync)")
ax3.set_ylim(0, 1.1)

# Add slider for coupling strength
axcolor = "lightgoldenrodyellow"
ax_K = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
K_slider = Slider(ax_K, "Coupling (K)", 0, 10, valinit=K, valstep=0.1)


# Update function for slider
def update(val):
    K = K_slider.val
    theta = odeint(kuramoto, theta0, t, args=(N, K, omega))
    theta = np.mod(theta, 2 * np.pi)

    # Update phase distribution
    scat.set_offsets(np.column_stack([np.cos(theta[-1]), np.sin(theta[-1])]))
    scat.set_array(theta[-1])

    # Update time series
    for i, line in enumerate(lines):
        line.set_ydata(theta[:, i])

    # Update order parameter
    order_param = np.array(
        [np.abs(np.mean(np.exp(1j * theta[i]))) for i in range(len(t))]
    )
    order_line.set_ydata(order_param)

    plt.draw()


K_slider.on_changed(update)

plt.suptitle("Model: Synchrony vs Coupling Strength", y=0.95)
plt.show()
