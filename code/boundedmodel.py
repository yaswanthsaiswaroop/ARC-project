import numpy as np
import matplotlib.pyplot as plt

############## System dynamics ################
def system_dynamics(q, u, t):
    
    # d = 0.05 * np.sin(0.05 * t)  # disturbance function
    # d = 2.5*q+0.05 * np.sin(0.05 * t)  # disturbance function
    d = 3*q+0.05 * np.sin(0.05 * t)  # disturbance function
    
    return -q + u + d

####### Older Sliding mode control law ################

def sliding_mode_control(q, K):
    s = q
    u = -K * np.sign(s)
    return u

########### Proposed sliding mode control law ################
def Proposed_sliding_mode_control(q, K0,K1):
    global A
    s = q
    rho=K0+K1*abs(q)
    u=  (-A*s)-(rho * np.sign(s))
    return u
######### Older adaptive gain law ################
def adaptive_gain_law(K, s):
    if K >= 0.1:
        K_dot = abs(s) * np.sign(abs(s) - 1)
    else:
        K_dot = 0.1
    return K_dot

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
q_oldmodel = np.zeros_like(t)
q_newmodel = np.zeros_like(t)
K_oldmodel = np.zeros_like(t)
K0_newmodel = np.zeros_like(t)
K1_newmodel = np.zeros_like(t)
q_oldmodel[0] = 0.5  # initial position
q_newmodel[0] = 0.5  # initial position
K_oldmodel[0] = 1.3
K0_newmodel[0] = 1.3
K1_newmodel[0] = 0.01
alpha=[1.1,1.1]
A=2


############# Run the simulation for old model ################
for i in range(1, len(t)):
    # Compute the control input and system dynamics
    u = sliding_mode_control(q_oldmodel[i - 1],K_oldmodel[i-1])
    q_dot = system_dynamics(q_oldmodel[i - 1], u, t[i])  # extract the scalar value
    
    # Update the state variables
    q_oldmodel[i] = q_oldmodel[i - 1] + q_dot * dt
    K_dot = adaptive_gain_law(K_oldmodel[i - 1], q_oldmodel[i - 1])
    K_oldmodel[i] = K_oldmodel[i - 1] + K_dot * dt

############ Run the Simulation for proposed model ################
for i in range(1, len(t)):

    u = Proposed_sliding_mode_control(q_newmodel[i - 1],K0_newmodel[i-1],K1_newmodel[i-1])
    q_dot = system_dynamics(q_newmodel[i - 1], u, t[i])  # extract the scalar value
    
    q_newmodel[i] = q_newmodel[i - 1] + q_dot * dt
    K_dot = Proposed_adaptive_gain_law(K0_newmodel[i - 1],K1_newmodel[i-1], q_newmodel[i - 1],alpha)
    K0_newmodel[i] = K0_newmodel[i - 1] + K_dot[0] * dt
    K1_newmodel[i] = K1_newmodel[i - 1] + K_dot[1] * dt

# Plot the results
plt.figure()
plt.plot(t, q_oldmodel, label='Older ASMC')
plt.plot(t, q_newmodel, label='Proposed ASMC')
plt.xlabel('Time (s)')
plt.ylabel('Position (q)')
# plt.xlim(0,0.1)
# plt.ylim(0, 0.5)
plt.title('Position response of the system')
plt.legend()

plt.figure()
plt.plot(t, K_oldmodel)
# plt.ylim(0, 0.5)
plt.xlabel('Time (s)')
plt.ylabel('K')
plt.title('Older ASMC K response(K0=1) ')

plt.figure()
plt.plot(t, K0_newmodel, label='Proposed ASMC K0')
plt.plot(t, K1_newmodel, label='Proposed ASMC K1')
plt.xlabel('Time (s)')
plt.ylabel('K0 and K1')
plt.title('Proposed ASMC K response (K0=1,K1=0.01)')
plt.legend()

plt.autoscale()
plt.show()
