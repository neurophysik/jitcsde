from __future__ import division, print_function
from sympy.abc import *
from sympy import *

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
β1 = sympify("[ -1,  4/3,  2/3, 0 ]")
β2 = sympify("[ -1,  4/3, -1/3, 0 ]")
β3 = sympify("[  2, -4/3, -2/3, 0 ]")
β4 = sympify("[  2,  5/3, -2/3, 1 ]")

H0 = DeferredVector("H0")
H1 = DeferredVector("H1")

I_1, I_11, I_111, I_10 = symbols("I_1 I_11 I_111 I_10")
f = Function("f")
g = Function("g")

print("H0_i")
for i in range(4):
	result  = y
	result += sum( A0[i][j] * f(t+c0[j]*h,H0[j]) * h      for j in range(4) )
	result += sum( B0[i][j] * g(t+c1[j]*h,H1[j]) * I_10/h for j in range(4) )
	print(result)

print("H1_i")
for i in range(4):
	result  = y
	result += sum( A1[i][j] * f(t+c0[j]*h,H0[j]) * h       for j in range(4) )
	result += sum( B1[i][j] * g(t+c1[j]*h,H1[j]) * sqrt(h) for j in range(4) )
	print(result)

print("y_new")
y_new  = y
y_new += sum( α[i] * f(t+c0[i]*h,H0[i]) * h for i in range(4) )
y_new += sum(	(
					  β1[i] * I_1
					+ β2[i] * I_11  / sqrt(h)
					+ β3[i] * I_10  / h
					+ β4[i] * I_111 / h
				) * g( t+c1[i]*h, H1[i] )
			for i in range(4)
			)
print(y_new)

print("error")
E  = h/6 * f(t+c0[0]*h,H0[0])
E -= h/6 * f(t+c0[1]*h,H0[1])
E += sum( ( β3[i] * I_10/h + β4[i] * I_111/h ) * g( t+c1[i]*h, H1[i] ) for i in range(4) )
print(E)

