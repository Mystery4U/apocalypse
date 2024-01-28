#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:50:19 2023

@author: koen
"""

import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime


# Define the different systems of ODEs

def first_system(t, x, a, b, Ri, Bs, Rs, Re):
    S, E, I, R = x
    N = sum(x)
    dSdt = -a*S*I/N
    dEdt = a*S*I/N - b*E
    dIdt = b*E - Ri*I
    dRdt = Ri*I
    return np.array([dSdt, dEdt, dIdt, dRdt])

def second_system(t, x, a, b, Ri, Bs, Rs, Re):
    S, E, I, R = x
    N = sum(x)
    dSdt = Bs*S - Rs*S - a*S*I/N
    dEdt = -Re*E + a*S*I/N - b*E
    dIdt = b*E - Ri*I
    dRdt = Rs*S + Re*E + Ri*I
    return np.array([dSdt, dEdt, dIdt, dRdt])


# Define the functions for the numerical integration methods

def forward_euler(tb, te, x0, dt, a, b, Ri, f):
    w = np.zeros((len(t),len(x0)))
    w[0] = x0
    
    for i in range(1,len(t)):
        w[i] = w[i-1] + dt * f(t[i-1],w[i-1], a, b, Ri, Bs, Rs, Re)
    return w

# # Deze is nog niet goed
# def backward_euler(tb, te, x0, dt, a, b, Ri):
#     w = np.zeros((len(t),len(x0)))
#     w[0] = x0
#     N = sum(x0)
    
#     I = np.eye(len(x0))
    
#     for i in range(1,len(t)):
#         A = np.array([[-a/N*w[i][2],0,-a/N*w[i][0],0], 
#                       [a/N*w[i][2],-b,a/N*w[i][0],0], 
#                       [0,b,-Ri,0], 
#                       [0,0,Ri,0]])

#         w[i] = np.linalg.solve((I-dt*A),w[i-1])
#     return w


def runga_kutta4(tb, te, x0, dt, a, b, Ri, f):
    w = np.zeros((len(t),len(x0)))
    w[0] = x0
    
    for i in range(1,len(t)):
        k1 = dt * f(t[i-1],w[i-1], a, b, Ri, Bs, Rs, Re)
        k2 = dt * f(t[i-1]+0.5*dt,w[i-1]+0.5*k1, a, b, Ri, Bs, Rs, Re)
        k3 = dt * f(t[i-1]+0.5*dt,w[i-1]+0.5*k2, a, b, Ri, Bs, Rs, Re)
        k4 = dt * f(t[i-1]+dt,w[i-1]+k3, a, b, Ri, Bs, Rs, Re)
        
        w[i] = w[i-1] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return w

dt = 1
tb = 0
te = 3*24105600+dt
t = np.arange(tb,te,dt)

start_date = datetime.datetime(1970, 1, 1)
dates = [start_date + datetime.timedelta(seconds=int(time)) for time in t]

formatted_dates = [date.strftime('%Y-%m') for date in dates]

num_ticks = 10
step_size = len(t) // num_ticks


#Set all parameters
a = 2.48e-6/3 #/3 # 3.43e-7 #1.39e-6
b = 4.44e-7 #8.87e-7 #4.02e-7
Bs = 10*4.21e-10
Rs = 7.29e-11 #3.92e-10
Re = 1.55e-8 #6.83e-9
Ri = 2.88e-7 #2.87e-7

x = np.array([500,0,1,0])




# #Calculate the solutions for the first system with different methods
# f1 = first_system

# x1_FE = forward_euler(tb,te,x,dt, a, b, Ri, f1)
# #x1_BE = backward_euler(tb,te,x,dt, a, b, Ri)
# x1_RK4 = runga_kutta4(tb,te,x,dt, a, b, Ri, f1)

# #Calculate solutions first system
# S1 = np.array([x1_FE[:,0],x1_RK4[:, 0]])
# E1 = np.array([x1_FE[:,1],x1_RK4[:, 1]])
# I1 = np.array([x1_FE[:,2],x1_RK4[:, 2]])
# R1 = np.array([x1_FE[:,3],x1_RK4[:, 3]])


# #Plot the results of the first system
# fig, axes = plt.subplots(2, figsize=(10, 10))

# titles = ['Forward Euler', 'Runge-Kutta 4']
# labels = ['S', 'E', 'I', 'R']
# results1 = [S1, E1, I1, R1]

# for i, ax in enumerate(axes):
#     for j, result in enumerate(results1):
#         ax.plot(t, result[i], '--', label=labels[j])
        
#     ax.set_title(titles[i])
#     ax.set_xlabel('Time [months]')
#     ax.set_ylabel('Amount of people')
#     ax.set_xticks(t[::step_size], formatted_dates[::step_size])
#     ax.grid()
#     ax.legend()

# plt.subplots_adjust(hspace=0.4)


# #Dit is voor verschil FE en RK-4
# S1 = abs(x1_FE[:, 0] - x1_RK4[:, 0]) #x1_RK4[:, 0]
# E1 = abs(x1_FE[:, 1] - x1_RK4[:, 1]) #x1_RK4[:, 1]
# I1 = abs(x1_FE[:, 2] - x1_RK4[:, 2]) #x1_RK4[:, 2]
# R1 = abs(x1_FE[:, 3] - x1_RK4[:, 3]) #x1_RK4[:, 3]

# plt.figure(figsize=(10,6))
# plt.plot(t,S1,'--',label="S")
# plt.plot(t,E1,'--',label="E")
# plt.plot(t,I1,'--',label="I")
# plt.plot(t,R1,'--',label="R")
# plt.xlabel('Time [months]')
# plt.ylabel('Amount of people')
# plt.grid()
# plt.legend()

# plt.xticks(t[::step_size], formatted_dates[::step_size])  # Adjust the step size as needed

# plt.show()






#Calculate the solutions for the second system
f2 = second_system

#x2_FE = forward_euler(tb,te,x,dt, a, b, Ri, f2)
#x2_BE = backward_euler(tb,te,x,dt, a, b, Ri)
x2_RK4 = runga_kutta4(tb,te,x,dt, a, b, Ri, f2)

S2 = x2_RK4[:, 0]
E2 = x2_RK4[:, 1]
I2 = x2_RK4[:, 2]
R2 = x2_RK4[:, 3]


plt.figure(figsize=(10,6))
plt.plot(t,S2,'--',label="S")
plt.plot(t,E2,'--',label="E")
plt.plot(t,I2,'--',label="I")
plt.plot(t,R2,'--',label="R")
plt.xlabel('Time [months]')
plt.ylabel('Amount of people')
plt.grid()
plt.legend()

plt.xticks(t[::step_size], formatted_dates[::step_size])  # Adjust the step size as needed

plt.show()


# #Plot the results of the second system
# fig, axes = plt.subplots(2, figsize=(7, 7))
# fig.suptitle('Results of the second model')

# results2 = [S2, E2, I2, R2]

# for i, ax in enumerate(axes):
#     for j, result in enumerate(results2):
#         ax.plot(t, result[i], '--', label=labels[j])

#     ax.set_title(titles[i])
#     ax.legend()

# plt.subplots_adjust(hspace=0.4)
# plt.show()



