# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:19:43 2020

@author: andre
"""

from math import cos
from numpy import arange


def f(x,t):
    return -x**4 + cos(t)

a = 0.0 # Start of the interval
b = 2.0 # End of the interval
N = 10 # Number of steps
h = (b-a)/N # Size of a single step
x = 0.0 # Initial condition
tpoints = arange(a,b,h)
xpoints = []
for t in tpoints:
    xpoints.append(x)
    k1 = h*f(x, t)
    k2 = h*f(x + 0.5*k1, t+0.5*h)
    x += k2
