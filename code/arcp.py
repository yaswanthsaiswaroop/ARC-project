import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the system dynamics
def system_dynamics(y, t, K):
    q, K = y
    s = q
    if K >= 0.1:
        Kdot = s * np.sign(abs(s) - 1)
    else:
        Kdot = 0.1
    u = -K * np.sign(s)
    d = 0.05 * np.sin(0.05 * t)
    qdot = -q + u + d
    return [qdot, Kdot]

# Set up the initial conditions and time vector
q0 = 0.5
K0 = 1.0
t = np.linspace(0, 100, 1000)

# Solve the differential equation using odeint
sol = odeint(system_dynamics, [q0, K0], t, args=(K0,))

# Extract the solution for q(t) and K(t)
q = sol[:, 0]
K = sol[:, 1]

# Plot the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, q)
plt.xlabel('Time')
plt.ylabel('q(t)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, K)
plt.xlabel('Time')
plt.ylabel('K(t)')
plt.grid(True)

plt.show()
