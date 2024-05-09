import numpy as np
import matplotlib.pyplot as plt

# Define the system dynamics
def system_dynamics(q, u, t):
    
    # d = 0.05 * np.sin(0.05 * t)  # disturbance function
    # d = 2.5*q+0.05 * np.sin(0.05 * t)  # disturbance function
    d = 2.5*q+0.05 * np.sin(0.05 * t)  # disturbance function
    
    return -q + u + d

# Define the sliding mode control law
def Proposed_sliding_mode_control(q, K0,K1):
    global A
    s = q
    rho=K0+K1*abs(q)
    u=  (-A*s)-(rho * np.sign(s))
    return u

# Define the adaptive gain law
def adaptive_gain_law(K0,K1, s,alpha):
    #declare an array of 2 elements
    K_dot=np.zeros(2)
    K_dot[0]=abs(s)- alpha[0] * K0
    K_dot[1]=abs(s)*abs(s)- alpha[1] * K1

    #first element of the array= abs(s)-1.1
    return K_dot

# Set the initial conditions and simulation parameters

d = lambda t: 0.05 * np.sin(0.05 * t)  # disturbance function
t0, tf = 0, 50  # simulation time interval
dt = 0.001  # simulation time step

# Initialize the simulation
t = np.arange(t0, tf, dt)
q = np.zeros_like(t)
K0 = np.zeros_like(t)
K1 = np.zeros_like(t)
A=2
q[0] = 0.5
K0[0] = 1.3
K1[0]=0.01
alpha=[1.1,1.1]
# Run the simulation
for i in range(1, len(t)):
    # Compute the control input and system dynamics
    u = Proposed_sliding_mode_control(q[i - 1],K0[i-1],K1[i-1])
    q_dot = system_dynamics(q[i - 1], u, t[i])  # extract the scalar value
    
    # Update the state variables
    q[i] = q[i - 1] + q_dot * dt
    K_dot = adaptive_gain_law(K0[i - 1],K1[i-1], q[i - 1],alpha)
    K0[i] = K0[i - 1] + K_dot[0] * dt
    K1[i] = K1[i - 1] + K_dot[1] * dt

# Plot the results
plt.figure()
plt.plot(t, q)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position response of the system')
plt.grid()
# plt.ylim(0, 4e-3)  # set the y-axis limits
plt.show()
