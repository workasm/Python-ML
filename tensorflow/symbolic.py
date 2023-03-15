
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sym

#sym.init_printing()

def logND(x,mu,sigma):
    return sym.exp(-(sym.ln(x)-mu)**2/(2*sigma**2))/(x*sigma*sym.sqrt(2*sym.pi))


x, w1, mu1, sigma1 = sym.symbols('x w1 mu1 sigma1')
w2, mu2, sigma2 = sym.symbols('w2 mu2 sigma2')
w3, mu3, sigma3 = sym.symbols('w3 mu3 sigma3')

g1=logND(x,mu1,sigma1)
g2=logND(x,mu2,sigma2)
g3=logND(x,mu3,sigma3)

den=w1*g1+w2*g2+w3*g3
y1=sym.simplify(w1*g1/den)
y2=sym.simplify(w2*g2/den)
y3=sym.simplify(w3*g3/den)

L1 = y1*sym.diff(g1,mu1)/g1

d1=sym.simplify(sym.diff(g1,mu1))
d2=sym.simplify(sym.diff(g1,sigma1))