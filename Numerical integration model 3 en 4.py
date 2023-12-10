#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:27:24 2023

@author: koen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

#Calculate S, E, I, R
def third_model(S, E, I, R, N):
    for t in range(0, t_max-1):
        for x in range(0, L):
            for y in range(0, L):
                if x == 0 and y == 0:
                    S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + S[t, x, y+1] + S[t, x, y+1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + E[t, x, y+1] + E[t, x, y+1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + I[t, x, y+1] + I[t, x, y+1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                  
                elif x == 0 and y != 0 and y != L:
                    S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                   
                elif x == 0 and y == L:
                    S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + S[t, x, y-1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + E[t, x, y-1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + I[t, x, y-1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
              
                    
                elif x == L and y == 0:
                    S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y+1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y+1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y+1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                   
                elif x == L and y != 0 and y != L:
                    S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                  
                elif x == L and y == L:
                    S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] + S[t, x, y-1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] + E[t, x, y-1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] + I[t, x, y-1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
                elif y == 0 and x != 0 and x != L:
                    S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y+1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y+1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y+1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
                elif y == L and x != 0 and x != L:
                    S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y-1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y-1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y-1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
                else:
                    S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y]
                    I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y]
                    N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
                    
                if E[t+1, x, y] < 0:
                    E[t+1, x, y] = 0
                    
                elif I[t+1, x, y] < 0:
                    I[t+1, x, y] = 0
                
    return S, E, I, R, N

L = 50
dx = 0.1
dy = 0.1

a = 5.22e-7     #0.05 #0.39                             #bite rate
b = 5.09e-7     #0.1 # #0.67                            #infection rate
d = 0.35                                                #diffusion coefficient
Bs = 3.73e-8    #4*1.1e-5 #6.6e-2 #0.29e-9 /4 #         #birth rate S
Rs = 0          #4*5.7e-7 #6.7e-2 #0.95e-10 /4 #        #death rate S
Re = 1.40e-8    #4*5.7e-7 #6.7e-2 #0.95e-10 /4 #        #death rate E
Ri = 5.01e-7    #4*6.7e-2 #0.69e--3 /4                  #death rate I
k = 0.2                                                 #kill rate

dt = dx**2 / (4*d) #=0.25 s
t_max = 100

gamma = d*dt/dx**2


# Initialize solution: the grid of S(t, x, y), etc.
S = np.zeros((t_max, L+1, L+1))
E = np.zeros((t_max, L+1, L+1))
I = np.zeros((t_max, L+1, L+1))
R = np.zeros((t_max, L+1, L+1))
N = np.zeros((t_max, L+1, L+1))

# Initial condition for S
S_initial = 1
S.fill(S_initial)

N_initial = 1
N.fill(N_initial)


# Set the initial conditions for one infected individual
initial_infected_x1 = int(L / 2)
initial_infected_y1 = int(L / 2)
I[0, initial_infected_x1, initial_infected_y1] = 1.0
S[0, initial_infected_x1, initial_infected_y1] = 0.0

# initial_infected_x2 = int(L / 1.5)
# initial_infected_y2 = int(L / 1.5)
# I[0, initial_infected_x2, initial_infected_y2] = 1.0
# S[0, initial_infected_x2, initial_infected_y2] = 0.0

# Calculate the solutions of the third model
S, E, I, R, N = third_model(S,E,I,R,N)

def plotheatmap(data, k, compartment, xlim=None, ylim=None):
    plt.title(f"Densities of {compartment}") #at t = {k * dt:.3f} s")
    plt.xlabel("x")
    plt.ylabel("y")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.pcolormesh(data, cmap=plt.cm.jet, vmin=0, vmax=1)
    plt.colorbar()

def animate(k):
    plt.subplot(2, 2, 1)
    plotheatmap(S[k], k, compartment='S', xlim=(0, L), ylim=(0, L))
    
    plt.subplot(2, 2, 2)
    plotheatmap(E[k], k, compartment='E', xlim=(0, L), ylim=(0, L))

    plt.subplot(2, 2, 3)
    plotheatmap(I[k], k, compartment='I', xlim=(0, L), ylim=(0, L))

    plt.subplot(2, 2, 4)
    plotheatmap(R[k], k, compartment='R', xlim=(0, L), ylim=(0, L))

anim = animation.FuncAnimation(plt.figure(figsize=(10,10)), animate, interval=1, frames=t_max, repeat=False)
anim.save("heat_equation_solution.gif")





# def calculate(S, E, I, R):
#     for t in range(0, t_max-1):
#         for x in range(1, L-1):
#             for y in range(1, L-1):
#                 S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N
#                 E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] - Re*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N - b*E[t, x, y]
#                 I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] - Ri*I[t, x, y] + b*E[t, x, y]
#                 R[t+1, x, y] = Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y]

#     return S, E, I, R





# Set boundary conditions for R explicitly
#R[:, 0, :] = Rs*S[:, 0, :] + Re*E[:, 0, :] + Ri*I[:, 0, :]
#R[:, -1, :] = Rs*S[:, -1, :] + Re*E[:, -1, :] + Ri*I[:, -1, :]
#R[:, :, 0] = Rs*S[:, :, 0] + Re*E[:, :, 0] + Ri*I[:, :, 0]
#R[:, :, -1] = Rs*S[:, :, -1] + Re*E[:, :, -1] + Ri*I[:, :, -1]


# # Update S boundary conditions
# S[:, -1, :] = S[:, 1, :]  # left boundary
# S[:, -1, :] = S[:, -2, :]  # right boundary
# S[:, :, -1] = S[:, :, 1]  # bottom boundary
# S[:, :, -1] = S[:, :, -2]  # top boundary

# # Update E boundary conditions
# E[:, 0, :] = E[:, 1, :]  # left boundary
# E[:, -1, :] = E[:, -2, :]  # right boundary
# E[:, :, 0] = E[:, :, 1]  # bottom boundary
# E[:, :, -1] = E[:, :, -2]  # top boundary

# # Update I boundary conditions
# I[:, 0, :] = I[:, 1, :]  # left boundary
# I[:, -1, :] = I[:, -2, :]  # right boundary
# I[:, :, 0] = I[:, :, 1]  # bottom boundary
# I[:, :, -1] = I[:, :, -2]  # top boundary



# def calculate(S,E,I,R):
    
#     for t in range(1, int(t_max/dt)-1):
#         for x in np.arange(1, L-dx, dx):
#             for y in np.arange(1, L-dx, dx):
#                 S[t+1,x,y] = gamma * (S[t][x+1][y] + S[t][x-1][y] + S[t][x][y+1] + S[t][x][y-1] - 4*S[t][x][y]) + S[t][x][y] + (Bs-Rs)*S[t][x][y] - a*S[t][x][y]*I[t][x][y]/N[t][x][y]
#                 E[t+1,x,y] = gamma * (E[t][x+1][y] + E[t][x-1][y] + E[t][x][y+1] + E[t][x][y-1] - 4*E[t][x][y]) + E[t][x][y] - Re*E[t][x][y] + a*S[t][x][y]*I[t][x][y]/N[t][x][y] - b*E[t][x][y]
#                 I[t+1,x,y] = gamma * (I[t][x+1][y] + I[t][x-1][y] + I[t][x][y+1] + I[t][x][y-1] - 4*I[t][x][y]) + I[t][x][y] - Ri*I[t][x][y] + b*E[t][x][y]
#                 R[t+1,x,y] = Rs*S[t][x][y] + Re*E[t][x][y] + Ri*I[t][x][y]
#     return S,E,I,R








# # Set the boundary conditions
# S[:, (L-1):, :] = S_top
# S[:, :, :1] = S_left
# S[:, :1, 1:] = S_bottom
# S[:, :, (L-1):] = S_right

# E[:, (L-1):, :] = E_top
# E[:, :, :1] = E_left
# E[:, :1, 1:] = E_bottom
# E[:, :, (L-1):] = E_right

# I[:, (L-1):, :] = I_top
# I[:, :, :1] = I_left
# I[:, :1, 1:] = I_bottom
# I[:, :, (L-1):] = I_right

# R[:, (L-1):, :] = R_top
# R[:, :, :1] = R_left
# R[:, :1, 1:] = R_bottom
# R[:, :, (L-1):] = R_right



# # Boundary conditions
# S_top = 100.0
# S_left = 0.0
# S_bottom = 0.0
# S_right = 0.0

# E_top = 0.0
# E_left = 0.0
# E_bottom = 0.0
# E_right = 0.0

# I_top = 1.0
# I_left = 0.0
# I_bottom = 0.0
# I_right = 0.0

# R_top = 0.0
# R_left = 0.0
# R_bottom = 0.0
# R_right = 0.0



# # Prepare the plot function
# def plotheatmap(S_k,E_k,I_k,R_k,k):
#   # Clear the current plot figure
#   plt.clf()
#   plt.title(f"Densities at t = {k*dt:.3f} unit time")
#   plt.xlabel("x")
#   plt.ylabel("y")
  
#   # Set vmin and vmax based on the full range of values in your compartments
#   vmin = min(S_k.min(), E_k.min(), I_k.min(), R_k.min())
#   vmax = max(S_k.max(), E_k.max(), I_k.max(), R_k.max())

  

#   # Plot each compartment separately
#   plt.pcolormesh(S_k, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
#   plt.pcolormesh(E_k, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
#   plt.pcolormesh(I_k, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
#   plt.pcolormesh(R_k, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
#   plt.colorbar()
  
#   return plt

# def plotheatmap(S_k, E_k, I_k, R_k, k, compartment='S'):
#     # Clear the current plot figure
#     plt.clf()
#     plt.title(f"Densities at t = {k * dt:.3f} unit time")
#     plt.xlabel("x")
#     plt.ylabel("y")

#     # Choose which compartment to plot based on the 'compartment' argument
#     if compartment == 'S':
#         data = S_k
#     elif compartment == 'E':
#         data = E_k
#     elif compartment == 'I':
#         data = I_k
#     elif compartment == 'R':
#         data = R_k
#     else:
#         raise ValueError(f"Invalid compartment: {compartment}")

#     # Plot the selected compartment
#     plt.pcolormesh(data, cmap=plt.cm.jet, vmin=0, vmax=100)
#     plt.colorbar()

#     return plt



# # Create a function to animate
# def animate(k):
#   plotheatmap(S[k], E[k], I[k], R[k], k, compartment='S')



# def animate(k):
#     plt.clf()
    
#     for t in range(k+1):
#         plt.subplot(1, k+1, t+1)
#         plt.title(f"Timestep {t}")
#         plt.pcolormesh(I[t], cmap=plt.cm.jet, vmin=0, vmax=100)
#         plt.axis('off')
    
#     plt.tight_layout()
    
# def animate(k):
#     plt.clf()
#     plotheatmap(I[k], k, compartment='I')

# def animate(k):
#     plt.clf()
    
#     plt.title(f"Densities of I at t = {k * dt:.3f} unit time")
#     plt.xlabel("x")
#     plt.ylabel("y")
    
#     plt.pcolormesh(I[k], cmap=plt.cm.jet, vmin=-1e+10, vmax=1e+10)
#     plt.colorbar()


# def plotheatmap(data, k, compartment='S'):
#     plt.clf()
#     plt.title(f"Densities at t = {k * dt:.3f} unit time")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.pcolormesh(data, cmap=plt.cm.jet, vmin=0, vmax=100)
#     plt.colorbar()


# # Create functions to plot the results
# def plotheatmap(data, k, compartment='I'):
#     #plt.clf()
#     plt.title(f"Densities of {compartment} at t = {k * dt:.3f} unit time")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.pcolormesh(data, cmap=plt.cm.jet, vmin=0, vmax=1)
#     plt.colorbar()

# def animate(k):
#     plt.subplot(2, 2, 1)
#     plotheatmap(S[k], k, compartment='S')

#     plt.subplot(2, 2, 2)
#     plotheatmap(E[k], k, compartment='E')

#     plt.subplot(2, 2, 3)
#     plotheatmap(I[k], k, compartment='I')

#     plt.subplot(2, 2, 4)
#     plotheatmap(R[k], k, compartment='R')

