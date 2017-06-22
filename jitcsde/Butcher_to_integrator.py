#!/usr/bin/env python3

from sympy.abc import X, t, h
from sympy import sympify, symbols, Function, sqrt

A0 = sympify("""[
		[ 0 , 0, 0, 0],
		[3/4, 0, 0, 0],
		[ 0 , 0, 0, 0],
		[ 0 , 0, 0, 0],
	]""")

B0 = sympify("""[
		[ 0 , 0, 0, 0],
		[3/2, 0, 0, 0],
		[ 0 , 0, 0, 0],
		[ 0 , 0, 0, 0],
	]""")

A1 = sympify("""[
		[ 0 , 0,  0 , 0],
		[1/4, 0,  0 , 0],
		[ 1 , 0,  0 , 0],
		[ 0 , 0, 1/4, 0],
	]""")

B1 = sympify("""[
		[ 0 , 0,  0 , 0],
		[1/2, 0,  0 , 0],
		[-1 , 0,  0 , 0],
		[-5 , 3, 1/2, 0],
	]""")

c0 = sympify("[ 0, 3/4, 0,  0  ]")
c1 = sympify("[ 0, 1/4, 1, 1/4 ]")

α  = sympify("[ 1/3, 2/3,   0 , 0 ]")
αt = sympify("[ 1/2, 1/2,   0 , 0 ]")
β1 = sympify("[ -1,  4/3,  2/3, 0 ]")
β2 = sympify("[ -1,  4/3, -1/3, 0 ]")
β3 = sympify("[  2, -4/3, -2/3, 0 ]")
β4 = sympify("[ -2,  5/3, -2/3, 1 ]")
β3t = β4t = [0,0,0,0]

H0 = H0_1, H0_2, H0_3, H0_4 = list(symbols("H0_1 H0_2 H0_3 H0_4"))
H1 = H1_1, H1_2, H1_3, H1_4 = list(symbols("H1_1 H1_2 H1_3 H1_4"))

I_1, I_11, I_111, I_10 = symbols("I_1 I_11 I_111 I_10")
f = Function("f")
g = Function("g")

for i in range(4):
	print("\nH^(0)_%i:"%i)
	result  = X
	result += sum( A0[i][j] * f(t+c0[j]*h,H0[j]) * h      for j in range(4) )
	result += sum( B0[i][j] * g(t+c1[j]*h,H1[j]) * I_10/h for j in range(4) )
	print(result)

for i in range(4):
	print("\nH^(1)_%i:"%i)
	result  = X
	result += sum( A1[i][j] * f(t+c0[j]*h,H0[j]) * h       for j in range(4) )
	result += sum( B1[i][j] * g(t+c1[j]*h,H1[j]) * sqrt(h) for j in range(4) )
	print(result)

def solution(α, β1, β2, β3, β4):
	X_bar  = X
	X_bar += sum( α[i] * f(t+c0[i]*h,H0[i]) * h for i in range(4) )
	for i in range(4):
		X_bar += g( t+c1[i]*h, H1[i] ) * (
						  β1[i] * I_1
						+ β2[i] * I_11  / sqrt(h)
						+ β3[i] * I_10  / h
						+ β4[i] * I_111 / h
					)
	return X_bar

X_bar = solution(α, β1, β2, β3, β4)
print("\n\\bar{X}:")
print(X_bar)

X_hat   = solution(αt, β1, β2, β3 , β4 )
X_tilde = solution(α , β1, β2, β3t, β4t)
X_hatil = solution(αt, β1, β2, β3t, β4t)

print("\nE:")
E  = X_bar-X_hatil
print(E.expand())

