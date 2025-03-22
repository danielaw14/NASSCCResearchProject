import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Generating a spiral for testing
def generate_spiral(n, noise=0.1):
    """
    Generate a spiral dataset with n points.
    """
    fix = int(n / 2)
    t = np.linspace(0, 4 * np.pi, fix)
    x = (2.2**2) * np.exp(0.25 * t) * np.cos(t) + (np.random.randn(fix) * noise if noise > 0 else 0)
    y = (2.2**2) * np.exp(0.25 * t) * np.sin(t) + (np.random.randn(fix) * noise if noise > 0 else 0)
    return t, x, y

# Create the spiral data
t, x, y = generate_spiral(100, 0)

plt.plot(x, y, '-')
plt.show()

# Define velocity functions
def x_velocity(t):
    return -4.84 * np.exp(0.25 * t) * np.sin(t) + 1.21 * np.exp(0.25 * t) * np.cos(t)

def y_velocity(t):
    return 4.84 * np.exp(0.25 * t) * np.cos(t) + 1.21 * np.exp(0.25 * t) * np.sin(t)

# Numerical derivative function
def diff(f, x_array, dx=1e-6):
    return np.array([(f(x + dx) - f(x - dx)) / (2.0 * dx) for x in x_array])

# Compute acceleration data
acc_x = diff(x_velocity, t)
acc_y = diff(y_velocity, t)

plt.figure()
plt.grid()
plt.xlabel('Time (sec)')
plt.ylabel('Acceleration')
plt.plot(t, acc_x, label='Ax')
plt.plot(t, acc_y, label='Ay')
plt.legend()
plt.show()

# Compute velocity and position estimates using trapezoidal rule
vel_x = integrate.cumulative_trapezoid(acc_x, t, initial=None)
vel_x = np.insert(vel_x, 0, x_velocity(0))  # Insert initial velocity at the start
est_x = integrate.cumulative_trapezoid(vel_x, t, initial=None)
est_x = np.insert(est_x, 0, x[0])  # Insert initial position at the start

vel_y = integrate.cumulative_trapezoid(acc_y, t, initial=None)
vel_y = np.insert(vel_y, 0, y_velocity(0))  # Insert initial velocity at the start
est_y = integrate.cumulative_trapezoid(vel_y, t, initial=None)
est_y = np.insert(est_y, 0, y[0])  # Insert initial position at the start

plt.plot(x, y, label='Actual Spiral Path')
plt.plot(est_x, est_y, label='Estimated Path')
plt.legend()
plt.show()

# Correct non-standard time steps using your custom method
vx = np.zeros_like(acc_x)
vy = np.zeros_like(acc_y)

vx[0] = x_velocity(0)
vy[0] = y_velocity(0)

for i in range(len(acc_x) - 1):
    elapsedTime = t[i + 1] - t[i]
    vx[i + 1] = vx[i] + acc_x[i] * elapsedTime
    vy[i + 1] = vy[i] + acc_y[i] * elapsedTime

px = np.zeros_like(acc_x)
py = np.zeros_like(acc_y)

px[0] = x[0]

for i in range(len(acc_x) - 1):
    elapsedTime = t[i + 1] - t[i]
    px[i + 1] = px[i] + vx[i] * elapsedTime
    py[i + 1] = py[i] + vy[i] * elapsedTime

plt.figure()
plt.grid()
plt.xlabel('Time (sec)')
plt.ylabel('Position')
plt.plot(t, px, label='X Position Estimate')
plt.plot(t, py, label='Y Position Estimate')
plt.legend()
plt.show()

plt.plot(x, y, label='Actual Spiral Path')
plt.plot(px, py, label='Improved Estimated Path')
plt.legend()
plt.show()
