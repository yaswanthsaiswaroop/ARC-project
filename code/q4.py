import numpy as np
import matplotlib.pyplot as plt

############## System dynamics ################
def system_dynamics(q, u, t):
    
    # d = 0.05 * np.sin(0.05 * t)  # disturbance function
    # d = 2.5*q+0.05 * np.sin(0.05 * t)  # disturbance function
    d = 3*q+0.05 * np.sin(0.05 * t)  # disturbance function
    
    return -q + u + d

########### Proposed sliding mode control law ################
def Proposed_sliding_mode_control(q, K0,K1):
    global A
    s = q
    rho=K0+K1*abs(q)
    u=  (-A*s)-(rho * np.sign(s))
    return u

######### Proposed adaptive gain law ################
def Proposed_adaptive_gain_law(K0,K1, s,alpha):

    K_dot=np.zeros(2)
    K_dot[0]=abs(s)- alpha[0] * K0
    K_dot[1]=abs(s)*abs(s)- alpha[1] * K1
    return K_dot

############# initial conditions and simulation parameters ################
 
t0, tf = 0,50  # simulation time interval
dt = 0.001      # simulation time step

# Initialize the simulation
t = np.arange(t0, tf, dt)
q_newmodel = np.zeros_like(t)
K0_newmodel = np.zeros_like(t)
K1_newmodel = np.zeros_like(t)
q_newmodel[0] = 0.5  # initial position
K0_newmodel[0] = 1.0
K1_newmodel[0] = 0.01
alpha=[6,6]
A=2


############ Run the Simulation for proposed model ################
for i in range(1, len(t)):

    u = Proposed_sliding_mode_control(q_newmodel[i - 1],K0_newmodel[i-1],K1_newmodel[i-1])
    q_dot = system_dynamics(q_newmodel[i - 1], u, t[i])  # extract the scalar value
    
    q_newmodel[i] = q_newmodel[i - 1] + q_dot * dt
    K_dot = Proposed_adaptive_gain_law(K0_newmodel[i - 1],K1_newmodel[i-1], q_newmodel[i - 1],alpha)
    K0_newmodel[i] = K0_newmodel[i - 1] + K_dot[0] * dt
    K1_newmodel[i] = K1_newmodel[i - 1] + K_dot[1] * dt



plt.figure()
plt.plot(t, K0_newmodel, label='Proposed ASMC K0')
plt.xlabel('Time (s)')
plt.ylabel('K0 ')
plt.title('Proposed ASMC K0 response (K0=1)')
plt.legend()

plt.autoscale()
plt.show()

