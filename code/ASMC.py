import numpy as np
from scipy.integrate import solve_ivp

# System parameters (placeholders, define according to your system)
m = 1.0  # Mass
c = 0.1  # Damping coefficient
g = 9.81 # Gravity

# Control parameters
Lambda = np.array([[1]])  # Positive definite matrix for control law
alpha = [0.1, 0.1, 0.1]  # Design scalars for adaptive gains

# Initial conditions
q0 = [0.1, 0]  # Initial position and velocity
K0 = [0.5, 0.5, 0.5]  # Initial gains

# Dynamics of the system
def dynamics(t, y):
    q, dq, K = y[0], y[1], y[2:]
    s = dq + Lambda[0][0] * q  # Sliding variable, simple case
    d = 0.05 * np.sin(0.05 * t)  # External disturbance
    
    # Control law
    rho = K[0] + K[1] * abs(s) + K[2] * abs(s)**2
    tau = -Lambda[0][0] * s - rho * np.sign(s)
    
    # System dynamics
    ddq = (tau - c*dq - m*g) / m
    
    # Gain adaptation
    dK = [alpha[0] * (abs(s) - K[0]),
          alpha[1] * (abs(s) - K[1]),
          alpha[2] * (abs(s) - K[2])]
    
    return [dq, ddq, *dK]

# Time span for the simulation
t_span = [0, 10]  # 10 seconds
t_eval = np.linspace(*t_span, 300)  # Time points to evaluate the solution

# Solve the system
sol = solve_ivp(dynamics, t_span, [q0[0], q0[1], *K0], t_eval=t_eval, method='RK45')

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(sol.t, sol.y[0], label='Position q')
plt.legend()

plt.subplot(312)
plt.plot(sol.t, sol.y[1], label='Velocity dq')
plt.legend()

plt.subplot(313)
plt.plot(sol.t, sol.y[2], label='Gain K0')
plt.plot(sol.t, sol.y[3], label='Gain K1')
plt.plot(sol.t, sol.y[4], label='Gain K2')
plt.legend()

plt.show()