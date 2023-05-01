from sympy import *
init_printing(use_unicode=True)
u = Symbol('u')
v = Symbol('v')
p = Symbol('p')
dp = Symbol('dp')
w = Function('w')
T = Function('T')
I = Function('I')

e = (T(w(u,v,dp)) - I(w(u,v,p)))**2
print(e)

e_lin = (T(w(u,v,0)) + Derivative(T(w(u,v,0)),u,v)*Derivative(w,dp) - I(w(u,v,p)))**2

print(e_lin)