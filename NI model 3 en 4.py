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
from tempfile import TemporaryFile
import calendar
import datetime

#Calculate S, E, I, R
def third_model(S, E, I, R, N):
    S_aantal = np.zeros(len(time)-1)
    E_aantal = np.zeros(len(time)-1)
    I_aantal = np.zeros(len(time)-1)
    R_aantal = np.zeros(len(time)-1)
    for t in range(0, len(time)-1): #for t in range(0, t_max_iter-1):
        #Flux bottom edge
        phiSxy0 = g*-ds*(S[t, 1:-1, 1] - S[t, 1:-1, 0])
        phiExy0 = g*-de*(E[t, 1:-1, 1] - E[t, 1:-1, 0])
        phiIxy0 = g*-di*(I[t, 1:-1, 1] - I[t, 1:-1, 0])
        
        #Flux upper edge
        phiSxyL = g*-ds*(S[t, 1:-1, -1] - S[t, 1:-1, -2])
        phiExyL = g*-de*(E[t, 1:-1, -1] - E[t, 1:-1, -2])
        phiIxyL = g*-di*(I[t, 1:-1, -1] - I[t, 1:-1, -2])

        #Flux left edge
        phiSyx0 = g*-ds*(S[t, 1, 1:-1] - S[t, 0, 1:-1])
        phiEyx0 = g*-de*(E[t, 1, 1:-1] - E[t, 0, 1:-1])
        phiIyx0 = g*-di*(I[t, 1, 1:-1] - I[t, 0, 1:-1])
        
        #Flux right edge
        phiSyxL = g*-ds*(S[t, -1, 1:-1] - S[t, -2, 1:-1])
        phiEyxL = g*-de*(E[t, -1, 1:-1] - E[t, -2, 1:-1])
        phiIyxL = g*-di*(I[t, -1, 1:-1] - I[t, -2, 1:-1])

        #Flux left bottom corner
        xphiSx0y0 = g*-ds*(S[t, 0, 1] - S[t, 0, 0])
        yphiSx0y0 = g*-ds*(S[t, 1, 0] - S[t, 0, 0])
        xphiEx0y0 = g*-de*(E[t, 0, 1] - E[t, 0, 0])
        yphiEx0y0 = g*-de*(E[t, 1, 0] - E[t, 0, 0])
        xphiIx0y0 = g*-di*(I[t, 0, 1] - I[t, 0, 0])
        yphiIx0y0 = g*-di*(I[t, 1, 0] - I[t, 0, 0])

        #Flux right bottom corner
        xphiSxLy0 = g*-ds*(S[t, -1, 1] - S[t, -1, 0])
        yphiSxLy0 = g*-ds*(S[t, -1, 0] - S[t, -2, 0])
        xphiExLy0 = g*-de*(E[t, -1, 1] - E[t, -1, 0])
        yphiExLy0 = g*-de*(E[t, -1, 0] - E[t, -2, 0])
        xphiIxLy0 = g*-di*(I[t, -1, 1] - I[t, -1, 0])
        yphiIxLy0 = g*-di*(I[t, -1, 0] - I[t, -2, 0])

        #Flux left upper corner
        xphiSx0yL = g*-ds*(S[t, 0, -1] - S[t, 0, -2])
        yphiSx0yL = g*-ds*(S[t, 1, -1] - S[t, 0, -1])
        xphiEx0yL = g*-de*(E[t, 0, -1] - E[t, 0, -2])
        yphiEx0yL = g*-de*(E[t, 1, -1] - E[t, 0, -1])
        xphiIx0yL = g*-di*(I[t, 0, -1] - I[t, 0, -2])
        yphiIx0yL = g*-di*(I[t, 1, -1] - I[t, 0, -1])
        
        #Flux right upper corner        
        xphiSxLyL = g*-ds*(S[t, -1, -1] - S[t, -1, -2])
        yphiSxLyL = g*-ds*(S[t, -1, -1] - S[t, -2, -1])
        xphiExLyL = g*-de*(E[t, -1, -1] - E[t, -1, -2])
        yphiExLyL = g*-de*(E[t, -1, -1] - E[t, -2, -1])
        xphiIxLyL = g*-di*(I[t, -1, -1] - I[t, -1, -2])
        yphiIxLyL = g*-di*(I[t, -1, -1] - I[t, -2, -1])
        
        
        #Interior points
        S[t+1, 1:-1, 1:-1] = gammas * (S[t, 2:, 1:-1] + S[t, :-2, 1:-1] + S[t, 1:-1, 2:] + S[t, 1:-1, :-2] - 4 * S[t, 1:-1, 1:-1]) + S[t, 1:-1, 1:-1] + dt * ((Bs - Rs) * S[t, 1:-1, 1:-1] - a * S[t, 1:-1, 1:-1] * I[t, 1:-1, 1:-1] / N[t, 1:-1, 1:-1])
        E[t+1, 1:-1, 1:-1] = gammae * (E[t, 2:, 1:-1] + E[t, :-2, 1:-1] + E[t, 1:-1, 2:] + E[t, 1:-1, :-2] - 4 * E[t, 1:-1, 1:-1]) + E[t, 1:-1, 1:-1] + dt * (-Re*E[t, 1:-1, 1:-1] - b*E[t, 1:-1, 1:-1] + a * S[t, 1:-1, 1:-1] * I[t, 1:-1, 1:-1] / N[t, 1:-1, 1:-1])
        I[t+1, 1:-1, 1:-1] = gammai * (I[t, 2:, 1:-1] + I[t, :-2, 1:-1] + I[t, 1:-1, 2:] + I[t, 1:-1, :-2] - 4 * I[t, 1:-1, 1:-1]) + I[t, 1:-1, 1:-1] + dt * (-Ri*I[t, 1:-1, 1:-1] + b*E[t, 1:-1, 1:-1] - k * I[t, 1:-1, 1:-1] * (S[t, 1:-1, 1:-1] + E[t, 1:-1, 1:-1]) / N[t, 1:-1, 1:-1])   
        R[t+1, 1:-1, 1:-1] = R[t, 1:-1, 1:-1] + dt*(Rs*S[t+1, 1:-1, 1:-1] + Re*E[t+1, 1:-1, 1:-1] + Ri*I[t+1, 1:-1, 1:-1] + k * I[t, 1:-1, 1:-1] * (S[t, 1:-1, 1:-1] + E[t, 1:-1, 1:-1]) / N[t, 1:-1, 1:-1])
        N[t+1, 1:-1, 1:-1] = S[t+1, 1:-1, 1:-1] + E[t+1, 1:-1, 1:-1] + I[t+1, 1:-1, 1:-1]
        
        #Left bottom corner
        S[t+1, 0, 0] = gammas * (2 * S[t, 1, 0] + 2*xphiSx0y0 + 2 * S[t, 0, 1] + 2*yphiSx0y0 - 4 * S[t, 0, 0]) + S[t, 0, 0] + dt * ((Bs - Rs) * S[t, 0, 0] - a * S[t, 0, 0] * I[t, 0, 0] / N[t, 0, 0])
        E[t+1, 0, 0] = gammae * (2 * E[t, 1, 0] + 2*xphiEx0y0 + 2 * E[t, 0, 1] + 2*yphiEx0y0 - 4 * E[t, 0, 0]) + E[t, 0, 0] + dt * (-Re * E[t, 0, 0] - b * E[t, 0, 0] + a * S[t, 0, 0] * I[t, 0, 0] / N[t, 0, 0])
        I[t+1, 0, 0] = gammai * (2 * I[t, 1, 0] + 2*xphiIx0y0 + 2 * I[t, 0, 1] + 2*yphiIx0y0 - 4 * I[t, 0, 0]) + I[t, 0, 0] + dt * (-Ri * I[t, 0, 0] + b * E[t, 0, 0] - k * I[t, 0, 0] * (S[t, 0, 0] + E[t, 0, 0]) / N[t, 0, 0])
        R[t+1, 0, 0] = R[t, 0, 0] + dt * (Rs * S[t, 0, 0] + Re * E[t, 0, 0] + Ri * I[t, 0, 0] + k * I[t, 0, 0] * (S[t, 0, 0] + E[t, 0, 0]) / N[t, 0, 0])
        N[t+1, 0, 0] = S[t+1, 0, 0] + E[t+1, 0, 0] + I[t+1, 0, 0]

        #Right bottom corner
        S[t+1, -1, 0] = gammas * (2 * S[t, -2, 0] + 2*xphiSxLy0 + 2 * S[t, -1, 1] - 2*yphiSxLy0 - 4 * S[t, -1, 0]) + S[t, -1, 0] + dt * ((Bs - Rs) * S[t, -1, 0] - a * S[t, -1, 0] * I[t, -1, 0] / N[t, -1, 0])
        E[t+1, -1, 0] = gammae * (2 * E[t, -2, 0] + 2*xphiExLy0 + 2 * E[t, -1, 1] - 2*yphiExLy0 - 4 * E[t, -1, 0]) + E[t, -1, 0] + dt * (-Re * E[t, -1, 0] - b * E[t, -1, 0] + a * S[t, -1, 0] * I[t, -1, 0] / N[t, -1, 0])
        I[t+1, -1, 0] = gammai * (2 * I[t, -2, 0] + 2*xphiIxLy0 + 2 * I[t, -1, 1] - 2*yphiIxLy0 - 4 * I[t, -1, 0]) + I[t, -1, 0] + dt * (-Ri * I[t, -1, 0] + b * E[t, -1, 0] - k * I[t, -1, 0] * (S[t, -1, 0] + E[t, -1, 0]) / N[t, -1, 0])
        R[t+1, -1, 0] = R[t, -1, 0] + dt * (Rs * S[t, -1, 0] + Re * E[t, -1, 0] + Ri * I[t, -1, 0] + k * I[t, -1, 0] * (S[t, -1, 0] + E[t, -1, 0]) / N[t, -1, 0])
        N[t+1, -1, 0] = S[t+1, -1, 0] + E[t+1, -1, 0] + I[t+1, -1, 0]
        
        #Left upper corner        
        S[t+1, 0, -1] = gammas * (2 * S[t, 1, -1] - 2*xphiSx0yL + 2 * S[t, 0, -2] + 2*yphiSx0yL - 4 * S[t, 0, -1]) + S[t, 0, -1] + dt * ((Bs - Rs) * S[t, 0, -1] - a * S[t, 0, -1] * I[t, 0, -1] / N[t, 0, -1])
        E[t+1, 0, -1] = gammae * (2 * E[t, 1, -1] - 2*xphiEx0yL + 2 * E[t, 0, -2] + 2*yphiEx0yL - 4 * E[t, 0, -1]) + E[t, 0, -1] + dt * (-Re * E[t, 0, -1] - b * E[t, 0, -1] + a * S[t, 0, -1] * I[t, 0, -1] / N[t, 0, -1])
        I[t+1, 0, -1] = gammai * (2 * I[t, 1, -1] - 2*xphiIx0yL + 2 * I[t, 0, -2] + 2*yphiIx0yL - 4 * I[t, 0, -1]) + I[t, 0, -1] + dt * (-Ri * I[t, 0, -1] + b * E[t, 0, -1] - k * I[t, 0, -1] * (S[t, 0, -1] + E[t, 0, -1]) / N[t, 0, -1])
        R[t+1, 0, -1] = R[t, 0, Ly] + dt * (Rs * S[t, 0, -1] + Re * E[t, 0, -1] + Ri * I[t, 0, -1] + k * I[t, 0, -1] * (S[t, 0, -1] + E[t, 0, -1]) / N[t, 0, -1])
        N[t+1, 0, -1] = S[t+1, 0, -1] + E[t+1, 0, -1] + I[t+1, 0, -1]
        
        #Right upper corner
        S[t+1, -1, -1] = gammas * (2 * S[t, -2, -1] - 2*xphiSxLyL + 2 * S[t, -1, -2] - 2*yphiSxLyL - 4 * S[t, -1, -1]) + S[t, -1, -1] + dt * ((Bs - Rs) * S[t, Lx, Ly] - a * S[t, Lx, Ly] * I[t, Lx, Ly] / N[t, Lx, Ly])
        E[t+1, -1, -1] = gammae * (2 * E[t, -2, -1] - 2*xphiExLyL + 2 * E[t, -1, -2] - 2*yphiExLyL - 4 * E[t, -1, -1]) + E[t, -1, -1] + dt * (-Re * E[t, Lx, Ly] - b * E[t, Lx, Ly] + a * S[t, Lx, Ly] * I[t, Lx, Ly] / N[t, Lx, Ly])
        I[t+1, -1, -1] = gammai * (2 * I[t, -2, -1] - 2*xphiIxLyL + 2 * I[t, -1, -2] - 2*yphiIxLyL - 4 * I[t, -1, -1]) + I[t, -1, -1] + dt * (-Ri * I[t, Lx, Ly] + b * E[t, Lx, Ly] - k * I[t, Lx, Ly] * (S[t, Lx, Ly] + E[t, Lx, Ly]) / N[t, Lx, Ly])
        R[t+1, -1, -1] = R[t, Lx, Ly] + dt * (Rs * S[t, Lx, Ly] + Re * E[t, Lx, Ly] + Ri * I[t, Lx, Ly] + k * I[t, Lx, Ly] * (S[t, Lx, Ly] + E[t, Lx, Ly]) / N[t, Lx, Ly])
        N[t+1, -1, -1] = S[t+1, Lx, Ly] + E[t+1, Lx, Ly] + I[t+1, Lx, Ly]
        
        #Bottom edge
        S[t+1, 1:-1, 0] = gammas * (S[t, 2:, 0] + S[t, :-2, 0] + 2 * S[t, 1:-1, 1] + 2*phiSxy0 - 4 * S[t, 1:-1, 0]) + S[t, 1:-1, 0] + dt * ((Bs - Rs) * S[t, 1:-1, 0] - a * S[t, 1:-1, 0] * I[t, 1:-1, 0] / N[t, 1:-1, 0])
        E[t+1, 1:-1, 0] = gammae * (E[t, 2:, 0] + E[t, :-2, 0] + 2 * E[t, 1:-1, 1] + 2*phiExy0 - 4 * E[t, 1:-1, 0]) + E[t, 1:-1, 0] + dt * (-Re * E[t, 1:-1, 0] - b * E[t, 1:-1, 0] + a * S[t, 1:-1, 0] * I[t, 1:-1, 0] / N[t, 1:-1, 0])
        I[t+1, 1:-1, 0] = gammai * (I[t, 2:, 0] + I[t, :-2, 0] + 2 * I[t, 1:-1, 1] + 2*phiIxy0 - 4 * I[t, 1:-1, 0]) + I[t, 1:-1, 0] + dt * (-Ri * I[t, 1:-1, 0] + b * E[t, 1:-1, 0] - k * I[t, 1:-1, 0] * (S[t, 1:-1, 0] + E[t, 1:-1, 0]) / N[t, 1:-1, 0])
        R[t+1, 1:-1, 0] = R[t, 1:-1, 0] + dt * (Rs * S[t, 1:-1, 0] + Re * E[t, 1:-1, 0] + Ri * I[t, 1:-1, 0] + k * I[t, 1:-1, 0] * (S[t, 1:-1, 0] + E[t, 1:-1, 0]) / N[t, 1:-1, 0])
        N[t+1, 1:-1, 0] = S[t+1, 1:-1, 0] + E[t+1, 1:-1, 0] + I[t+1, 1:-1, 0]

        #Upper edge
        S[t+1, 1:-1, -1] = gammas * (S[t, 2:, -1] + S[t, :-2, -1] + 2 * S[t, 1:-1, -2] -2*phiSxyL - 4 * S[t, 1:-1, -1]) + S[t, 1:Lx, Ly] + dt * ((Bs - Rs) * S[t, 1:Lx, Ly] - a * S[t, 1:Lx, Ly] * I[t, 1:Lx, Ly] / N[t, 1:Lx, Ly])
        E[t+1, 1:-1, -1] = gammae * (E[t, 2:, -1] + E[t, :-2, -1] + 2 * E[t, 1:-1, -2] -2*phiExyL - 4 * E[t, 1:-1, -1]) + E[t, 1:Lx, Ly] + dt * (-Re * E[t, 1:Lx, Ly] - b * E[t, 1:Lx, Ly] + a * S[t, 1:Lx, Ly] * I[t, 1:Lx, Ly] / N[t, 1:Lx, Ly])
        I[t+1, 1:-1, -1] = gammai * (I[t, 2:, -1] + I[t, :-2, -1] + 2 * I[t, 1:-1, -2] -2*phiIxyL - 4 * I[t, 1:-1, -1]) + I[t, 1:Lx, Ly] + dt * (-Ri * I[t, 1:Lx, Ly] + b * E[t, 1:Lx, Ly] - k * I[t, 1:Lx, Ly] * (S[t, 1:Lx, Ly] + E[t, 1:Lx, Ly]) / N[t, 1:Lx, Ly])
        R[t+1, 1:-1, -1] = R[t, 1:Lx, Ly] + dt * (Rs * S[t, 1:Lx, Ly] + Re * E[t, 1:Lx, Ly] + Ri * I[t, 1:Lx, Ly] + k * I[t, 1:Lx, Ly] * (S[t, 1:Lx, Ly] + E[t, 1:Lx, Ly]) / N[t, 1:Lx, Ly])
        N[t+1, 1:-1, -1] = S[t+1, 1:Lx, Ly] + E[t+1, 1:Lx, Ly] + I[t+1, 1:Lx, Ly]
        
        #Left edge
        S[t+1, 0, 1:-1] = gammas * (2 * S[t, 1, 1:-1] + 2*phiSyx0 + S[t, 0, 2:] + S[t, 0, :-2] - 4 * S[t, 0, 1:-1]) + S[t, 0, 1:Ly] + dt * ((Bs - Rs) * S[t, 0, 1:Ly] - a * S[t, 0, 1:Ly] * I[t, 0, 1:Ly] / N[t, 0, 1:Ly])
        E[t+1, 0, 1:-1] = gammae * (2 * E[t, 1, 1:-1] + 2*phiEyx0 + E[t, 0, 2:] + E[t, 0, :-2] - 4 * E[t, 0, 1:-1]) + E[t, 0, 1:Ly] + dt * (-Re * E[t, 0, 1:Ly] - b * E[t, 0, 1:Ly] + a * S[t, 0, 1:Ly] * I[t, 0, 1:Ly] / N[t, 0, 1:Ly])
        I[t+1, 0, 1:-1] = gammai * (2 * I[t, 1, 1:-1] + 2*phiIyx0 + I[t, 0, 2:] + I[t, 0, :-2] - 4 * I[t, 0, 1:-1]) + I[t, 0, 1:Ly] + dt * (-Ri * I[t, 0, 1:Ly] + b * E[t, 0, 1:Ly] - k * I[t, 0, 1:Ly] * (S[t, 0, 1:Ly] + E[t, 0, 1:Ly]) / N[t, 0, 1:Ly])
        R[t+1, 0, 1:-1] = R[t, 0, 1:Ly] + dt * (Rs * S[t, 0, 1:Ly] + Re * E[t, 0, 1:Ly] + Ri * I[t, 0, 1:Ly] + k * I[t, 0, 1:Ly] * (S[t, 0, 1:Ly] + E[t, 0, 1:Ly]) / N[t, 0, 1:Ly])
        N[t+1, 0, 1:-1] = S[t+1, 0, 1:Ly] + E[t+1, 0, 1:Ly] + I[t+1, 0, 1:Ly]
        
        #Right edge
        S[t+1, -1, 1:-1] = gammas * (2 * S[t, -2, 1:-1] - 2*phiSyxL + S[t, -1, 2:] + S[t, -1, :-2] - 4 * S[t, -1, 1:-1]) + S[t, Lx, 1:Ly] + dt * ((Bs - Rs) * S[t, Lx, 1:Ly] - a * S[t, Lx, 1:Ly] * I[t, Lx, 1:Ly] / N[t, Lx, 1:Ly])
        E[t+1, -1, 1:-1] = gammae * (2 * E[t, -2, 1:-1] - 2*phiEyxL + E[t, -1, 2:] + E[t, -1, :-2] - 4 * E[t, -1, 1:-1]) + E[t, Lx, 1:Ly] + dt * (-Re * E[t, Lx, 1:Ly] - b * E[t, Lx, 1:Ly] + a * S[t, Lx, 1:Ly] * I[t, Lx, 1:Ly] / N[t, Lx, 1:Ly])
        I[t+1, -1, 1:-1] = gammai * (2 * I[t, -2, 1:-1] - 2*phiIyxL + I[t, -1, 2:] + I[t, -1, :-2] - 4 * I[t, -1, 1:-1]) + I[t, Lx, 1:Ly] + dt * (-Ri * I[t, Lx, 1:Ly] + b * E[t, Lx, 1:Ly] - k * I[t, Lx, 1:Ly] * (S[t, Lx, 1:Ly] + E[t, Lx, 1:Ly]) / N[t, Lx, 1:Ly])
        R[t+1, -1, 1:-1] = R[t, Lx, 1:Ly] + dt * (Rs * S[t, Lx, 1:Ly] + Re * E[t, Lx, 1:Ly] + Ri * I[t, Lx, 1:Ly] + k * I[t, Lx, 1:Ly] * (S[t, Lx, 1:Ly] + E[t, Lx, 1:Ly]) / N[t, Lx, 1:Ly])
        N[t+1, -1, 1:-1] = S[t+1, Lx, 1:Ly] + E[t+1, Lx, 1:Ly] + I[t+1, Lx, 1:Ly]
                    
    
        for x_int in range(0, Lx):
            for y_int in range(0, Ly):
                if S[t+1, x_int, y_int] < 0:
                    S[t+1, x_int, y_int] = 0
                            
                if E[t+1, x_int, y_int] < 0:
                    E[t+1, x_int, y_int] = 0
                            
                if I[t+1, x_int, y_int] < 0:
                    I[t+1, x_int, y_int] = 0
        
        #Time integration
        S_aantal[t] = np.sum(S[t, :, :])
        E_aantal[t] = np.sum(E[t, :, :])
        I_aantal[t] = np.sum(I[t, :, :])
        R_aantal[t] = np.sum(R[t, :, :])

    return S, E, I, R, N, S_aantal, E_aantal, I_aantal, R_aantal

Lx = 50
Ly = 50
dx = 0.1
dy = 0.1

dt = 0.07 #0.00714 #dx**2 / (4*d) #=0.0071428571 s
tb = 0
t_max_iter = 150
time = np.arange(tb, t_max_iter, dt)


a = 0.38 #0.8 #0.38 #0.38 #1 #0.05*140                    #2.48e-3 #*630720 #5.22e-7    #0.39                              #bite rate
Bs = 3.803653001e-5 #dt*0.0053272451 #a*5.0e-5 #0.005 #0                          #3.73e-8*140 #1.88e-3 #*630720 #3.73e-3 #3.73e-8    #4*1.1e-5 #6.6e-2 #0.29e-9 /4 #         #birth rate S
b = Bs*6099.399707 #a*0.29 #a*0.18 #0.1 #0.1*140                     #4.44e-4 #*630720 #5.09e-2 #5.09e-7 #0.1   # #0.67                              #infection rate
ds = 0.035 #0.35                        #*630720 #.35 #0.35                                                #diffusion coefficient
de = ds
di = ds/2 #0.5*ds
Rs = Bs*0.0010936855 #dt*0.0053942161 #a*5.2e-8 #0 #0.01 #0*140                      #3e-3 #*630720 #3.73e-3 #0          #4*5.7e-7 #6.7e-2 #0.95e-10 /4 #        #death rate S
Re = Bs*210.3241278 #a*0.01 #0 #0.001 #1.40e-8*140                #1.40e-5 #*630720 #1.40e-3 #1.40e-8    #4*5.7e-7 #6.7e-2 #0.95e-10 /4 #        #death rate E
Ri = Bs*4416.806684 #a*0.21 #a*0.12 #0.005 #0.003 #5.01e-7*140                #2.88e-4 #*630720 #5.01e-3 #5.01e-7    #4*6.7e-2 #0.69e--3 /4                  #death rate I
k = 0 #1 #0.20 #0.19 #0.18 #0.15 #0.1 #0.05      #0.197 #0.195
g = 0

gammas = ds*dt/dx**2 #=0.25
gammae = de*dt/dx**2 #=0.25
gammai = di*dt/dx**2 #=0.25

# Initialize solution: the grid of S(t, x, y), etc.
S = np.zeros((len(time), Lx+1, Ly+1))
E = np.zeros((len(time), Lx+1, Ly+1))
I = np.zeros((len(time), Lx+1, Ly+1))
R = np.zeros((len(time), Lx+1, Ly+1))
N = np.zeros((len(time), Lx+1, Ly+1))

# Initial condition for S
S_initial = 1
S.fill(S_initial)

N_initial = 1
N.fill(N_initial)


# Set the initial conditions for one infected individual
initial_infected_x1 = 10
initial_infected_y1 = 10
I[0, initial_infected_x1, initial_infected_y1] = 1
S[0, initial_infected_x1, initial_infected_y1] = 1
N[0, initial_infected_x1, initial_infected_y1] = 2


initial_infected_x2 = 40
initial_infected_y2 = 40
I[0, initial_infected_x2, initial_infected_y2] = 1
S[0, initial_infected_x2, initial_infected_y2] = 1
N[0, initial_infected_x2, initial_infected_y2] = 2

# Calculate the solutions of the third model
S, E, I, R, N, S_aantal, E_aantal, I_aantal, R_aantal = third_model(S,E,I,R,N)
#outfile = TemporaryFile()
#np.save(outfile, S, E, I, R, N, S_aantal, E_aantal, I_aantal, R_aantal)



def plotheatmap(data, k, compartment, xlim=None, ylim=None):
    plt.title(f"Density of {compartment}") #at t = {k*dt:.2f} s")
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
    plotheatmap(S[k], k, compartment='S', xlim=(0, Ly), ylim=(0, Lx))
    
    plt.subplot(2, 2, 2)
    plotheatmap(E[k], k, compartment='E', xlim=(0, Ly), ylim=(0, Lx))

    plt.subplot(2, 2, 3)
    plotheatmap(I[k], k, compartment='I', xlim=(0, Ly), ylim=(0, Lx))

    plt.subplot(2, 2, 4)
    plotheatmap(R[k], k, compartment='R', xlim=(0, Ly), ylim=(0, Lx))


# last_time_step = t_max_iter//2 # - 1

# fig = plt.figure(figsize=(10, 10))

# animate(last_time_step)

# writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# output_file = "2 zombies, k=0,2, weglopen 1.mp4"
# with writer.saving(fig, output_file, 100):
#     for k in range(t_max_iter):
#         animate(k)
#         writer.grab_frame()

# plt.show()


t = np.linspace(tb,t_max_iter,len(S_aantal))/60

tplot = t*585365

start_date = datetime.datetime(1970, 1, 1)
dates = [start_date + datetime.timedelta(seconds=int(time)) for time in tplot]

formatted_dates = [date.strftime('%m-%d') for date in dates]

num_ticks = 10
step_size = len(tplot) // num_ticks


plt.figure(figsize=(10,7))
plt.plot(tplot,S_aantal,'--',label='S')
plt.plot(tplot,E_aantal,'--',label='E')
plt.plot(tplot,I_aantal,'--',label='I')
plt.plot(tplot,R_aantal,'--',label='R')
plt.xlabel('Time [months]')
plt.ylabel('Amount of people')
plt.legend()
plt.title('k = {}'.format(k))
plt.grid()
plt.xticks(tplot[::step_size], formatted_dates[::step_size])
plt.show()




















# S_int, E_int, I_int, R_int = 0, 0, 0, 0
               
#         S_int += S[t, x_int, y_int]
#         E_int += E[t, x_int, y_int]
#         I_int += I[t, x_int, y_int]
#         R_int += R[t, x_int, y_int]

# # Multiply by the area element for actual integral values
# S_int *= dx * dy
# E_int *= dx * dy
# I_int *= dx * dy
# R_int *= dx * dy

# # Appending the integrated values
# S_aantal[t] = S_int
# E_aantal[t] = E_int
# I_aantal[t] = I_int
# R_aantal[t] = R_int






        # for x in range(0, Lx):
        #     for y in range(0, Ly):
        #         if x == 0 and y == 0:
        #             phiSx0 = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiEx0 = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIx0 = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSy0 = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEy0 = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIy0 = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + dx*phiSx0 + S[t, x, y+1] + S[t, x, y+1] + dy*phiSy0 - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + dx*phiEx0 + E[t, x, y+1] + E[t, x, y+1] + dy*phiEy0 - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + dx*phiIx0 + I[t, x, y+1] + I[t, x, y+1] + dy*phiIy0 - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
                    
        #         elif x == 0 and y != 0 and y != Ly:
        #             phiSx0 = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiEx0 = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIx0 = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + dx*phiSx0 + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + dx*phiEx0 + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + dx*phiIx0 + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                   

        #         elif x == 0 and y == Ly:
        #             phiSx0 = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiEx0 = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIx0 = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSyL = -d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEyL = -d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIyL = -d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + dx*phiSx0 + S[t, x, y-1] + S[t, x, y-1] - dy*phiSyL - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + dx*phiEx0 + E[t, x, y-1] + E[t, x, y-1] - dy*phiEyL - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + dx*phiIx0 + I[t, x, y-1] + I[t, x, y-1] - dy*phiIyL - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
              

        #         elif x == Lx and y == 0:
        #             phiSxL = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiExL = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIxL = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSy0 = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEy0 = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIy0 = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] - dx*phiSxL + S[t, x, y+1] + S[t, x, y+1] + dy*phiSy0 - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] - dx*phiExL + E[t, x, y+1] + E[t, x, y+1] + dy*phiEy0 - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] - dx*phiIxL + I[t, x, y+1] + I[t, x, y+1] + dy*phiIy0 - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                   

        #         elif x == Lx and y != 0 and y != Ly:
        #             phiSxL = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiExL = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIxL = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
                    
        #             S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] - dx*phiSxL + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] - dx*phiExL + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] - dx*phiIxL + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                  

        #         elif x == Lx and y == Ly:
        #             phiSxL = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiExL = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIxL = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSyL = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEyL = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIyL = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] - dx*phiSxL + S[t, x, y-1] + S[t, x, y-1] - dy*phiSyL - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] - dx*phiExL + E[t, x, y-1] + E[t, x, y-1] - dy*phiEyL - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] - dx*phiIxL + I[t, x, y-1] + I[t, x, y-1] - dy*phiIyL - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    

        #         elif y == 0 and x != 0 and x != Lx:
        #             phiSy0 = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEy0 = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIy0 = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y+1] + dy*phiSy0 - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y+1] + dy*phiEy0 - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y+1] + dy*phiIy0 - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    

        #         elif y == Ly and x != 0 and x != Lx:
        #             phiSyL = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEyL = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIyL = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y-1] + S[t, x, y-1] - dy*phiSyL - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y-1] + E[t, x, y-1] - dy*phiEyL - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y-1] + I[t, x, y-1] - dy*phiIyL - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
        #         else:
        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + dt*((Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] + dt*(a*S[t, x, y]*I[t, x, y]/N[t, x, y] - Re*E[t, x, y] - b*E[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] + dt*(b*E[t, x, y] - Ri*I[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    

        #         if S[t+1, x, y] < 0:
        #             S[t+1, x, y] = 0
                    
        #         if E[t+1, x, y] < 0:
        #             E[t+1, x, y] = 0
                    
        #         if I[t+1, x, y] < 0:
        #             I[t+1, x, y] = 0
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


        # S_int = 0
        # E_int = 0
        # I_int = 0
        # R_int = 0
        # for x_int in range(0,Lx):
        #     for y_int in range(0,Ly):
        #         S_int += S[t,x_int,y_int]
                
        #         E_int += E[t,x_int,y_int]
            
        #         I_int += I[t,x_int,y_int]
                
        #         R_int += R[t,x_int,y_int]
                
        # #S_int = S_int*Lx*Ly
        # #E_int = E_int*Lx*Ly
        # #I_int = I_int*Lx*Ly
        # #R_int = R_int*Lx*Ly
        
        # S_aantal = np.append(S_aantal,S_int)
        # E_aantal = np.append(E_aantal,E_int)
        # I_aantal = np.append(I_aantal,I_int)
        # R_aantal = np.append(R_aantal,R_int)






                # S[t+1, 0, 0] = (gamma * (S[t, 1, 0] + S[t, 1, 0] + dx*phiSx0 + S[t, 0, 1] + S[t, 0, 1] + dy*phiSy0 - 4*S[t, 0, 0])
                #                 + S[t, 0, 0] + dt*(+ (Bs-Rs)*S[t, 0, 0] - a*S[t, 0, 0]*I[t, 0, 0]/N[t, 0, 0]))
                # E[t+1, 0, 0] = (gamma * (E[t, 1, 0] + E[t, 1, 0] + dx*phiEx0 + E[t, 0, 1] + E[t, 0, 1] + dy*phiEy0 - 4*E[t, 0, 0])
                #                 + E[t, 0, 0] + dt*(-Re * E[t, 0, 0] - b * E[t, 0, 0] + a*S[t, 0, 0]*I[t, 0, 0]/N[t, 0, 0]))
                # I[t+1, 0, 0] = (gamma * (I[t, 1, 0] + I[t, 1, 0] + dx*phiIx0 + I[t, 0, 1] + I[t, 0, 1] + dy*phiIy0 - 4*I[t, 0, 0])
                #                 + I[t, 0, 0] + dt*(-Ri * I[t, 0, 0] + b * E[t, 0, 0] - k * I[t, 0, 0] * (S[t, 0, 0] + E[t, 0, 0]) / N[t, 0, 0]))
                # R[t+1, 0, 0] = R[t, 0, 0] + dt*(Rs * S[t+1, 0, 0] + Re * E[t+1, 0, 0] + Ri * I[t+1, 0, 0])











        # for x in range(0, Lx):
        #     for y in range(0, Ly):
        #         if x == 0 and y == 0:
        #             phiSx0 = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiEx0 = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIx0 = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSy0 = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEy0 = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIy0 = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + dx*phiSx0 + S[t, x, y+1] + S[t, x, y+1] + dy*phiSy0 - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + dx*phiEx0 + E[t, x, y+1] + E[t, x, y+1] + dy*phiEy0 - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + dx*phiIx0 + I[t, x, y+1] + I[t, x, y+1] + dy*phiIy0 - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
                    
        #         elif x == 0 and y != 0 and y != Ly:
        #             phiSx0 = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiEx0 = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIx0 = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + dx*phiSx0 + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + dx*phiEx0 + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + dx*phiIx0 + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                   

        #         elif x == 0 and y == Ly:
        #             phiSx0 = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiEx0 = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIx0 = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSyL = -d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEyL = -d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIyL = -d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x+1, y] + dx*phiSx0 + S[t, x, y-1] + S[t, x, y-1] - dy*phiSyL - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x+1, y] + dx*phiEx0 + E[t, x, y-1] + E[t, x, y-1] - dy*phiEyL - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x+1, y] + dx*phiIx0 + I[t, x, y-1] + I[t, x, y-1] - dy*phiIyL - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
              

        #         elif x == Lx and y == 0:
        #             phiSxL = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiExL = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIxL = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSy0 = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEy0 = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIy0 = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] - dx*phiSxL + S[t, x, y+1] + S[t, x, y+1] + dy*phiSy0 - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] - dx*phiExL + E[t, x, y+1] + E[t, x, y+1] + dy*phiEy0 - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] - dx*phiIxL + I[t, x, y+1] + I[t, x, y+1] + dy*phiIy0 - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                   

        #         elif x == Lx and y != 0 and y != Ly:
        #             phiSxL = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiExL = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIxL = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
                    
        #             S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] - dx*phiSxL + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] - dx*phiExL + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] - dx*phiIxL + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                  

        #         elif x == Lx and y == Ly:
        #             phiSxL = 0 #-d*(S[t, x+1, y] - S[t, x, y])/dx
        #             phiExL = 0 #-d*(E[t, x+1, y] - E[t, x, y])/dx
        #             phiIxL = 0 #-d*(I[t, x+1, y] - I[t, x, y])/dx
                    
        #             phiSyL = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEyL = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIyL = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x-1, y] + S[t, x-1, y] - dx*phiSxL + S[t, x, y-1] + S[t, x, y-1] - dy*phiSyL - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x-1, y] + E[t, x-1, y] - dx*phiExL + E[t, x, y-1] + E[t, x, y-1] - dy*phiEyL - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x-1, y] + I[t, x-1, y] - dx*phiIxL + I[t, x, y-1] + I[t, x, y-1] - dy*phiIyL - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    

        #         elif y == 0 and x != 0 and x != Lx:
        #             phiSy0 = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEy0 = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIy0 = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y+1] + dy*phiSy0 - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y+1] + dy*phiEy0 - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y+1] + dy*phiIy0 - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    

        #         elif y == Ly and x != 0 and x != Lx:
        #             phiSyL = 0 #-d*(S[t, x, y+1] - S[t, x, y])/dy
        #             phiEyL = 0 #-d*(E[t, x, y+1] - E[t, x, y])/dy
        #             phiIyL = 0 #-d*(I[t, x, y+1] - I[t, x, y])/dy

        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y-1] + S[t, x, y-1] - dy*phiSyL - 4*S[t, x, y]) + S[t, x, y] + dt*(+ (Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y-1] + E[t, x, y-1] - dy*phiEyL - 4*E[t, x, y]) + E[t, x, y] + dt*(- Re*E[t, x, y] - b*E[t, x, y] + a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y-1] + I[t, x, y-1] - dy*phiIyL - 4*I[t, x, y]) + I[t, x, y] + dt*(- Ri*I[t, x, y] + b*E[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
                    
        #         else:
        #             S[t+1, x, y] = gamma * (S[t, x+1, y] + S[t, x-1, y] + S[t, x, y+1] + S[t, x, y-1] - 4*S[t, x, y]) + S[t, x, y] + dt*((Bs-Rs)*S[t, x, y] - a*S[t, x, y]*I[t, x, y]/N[t, x, y])
        #             E[t+1, x, y] = gamma * (E[t, x+1, y] + E[t, x-1, y] + E[t, x, y+1] + E[t, x, y-1] - 4*E[t, x, y]) + E[t, x, y] + dt*(a*S[t, x, y]*I[t, x, y]/N[t, x, y] - Re*E[t, x, y] - b*E[t, x, y])
        #             I[t+1, x, y] = gamma * (I[t, x+1, y] + I[t, x-1, y] + I[t, x, y+1] + I[t, x, y-1] - 4*I[t, x, y]) + I[t, x, y] + dt*(b*E[t, x, y] - Ri*I[t, x, y] - k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             R[t+1, x, y] = R[t, x, y] + dt*(Rs*S[t, x, y] + Re*E[t, x, y] + Ri*I[t, x, y] + k*I[t, x, y]*(S[t, x, y] + E[t, x, y])/N[t, x, y])
        #             N[t+1, x, y] = S[t+1, x, y] + E[t+1, x, y] + I[t+1, x, y]
        #             # print(I[t, x, y])
        #             # print()
        #             # print(-Ri*I[t, x, y])
        #             # print()
        #             # print(- b*E[t, x, y] )
        #             # print()
        #             # print()
        #         if S[t+1, x, y] < 0:
        #             S[t+1, x, y] = 0
                    
        #         if E[t+1, x, y] < 0:
        #             E[t+1, x, y] = 0
                    
        #         if I[t+1, x, y] < 0:
        #             I[t+1, x, y] = 0
        
        
        
        
        
        
        
        
        
        
        
        
        


# anim = animation.FuncAnimation(plt.figure(figsize=(10,10)), animate, interval=1, frames=t_max, repeat=False)
# anim.save("heat_equation_solution.gif")


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

