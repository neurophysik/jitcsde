from __future__ import division
from numpy import sqrt, array

A0 = array([
		[ 0 , 0, 0, 0],
		[3/4, 0, 0, 0],
		[ 0 , 0, 0, 0],
		[ 0 , 0, 0, 0],
	])

B0 = array([
		[ 0 , 0, 0, 0],
		[3/2, 0, 0, 0],
		[ 0 , 0, 0, 0],
		[ 0 , 0, 0, 0],
	])

A1 = array([
		[ 0 , 0,  0 , 0],
		[1/4, 0,  0 , 0],
		[ 1 , 0,  0 , 0],
		[ 0 , 0, 1/4, 0],
	])

B1 = array([
		[ 0 , 0,  0 , 0],
		[1/2, 0,  0 , 0],
		[-1 , 0,  0 , 0],
		[-5 , 3, 1/2, 0],
	])

c0 = array([ 0, 3/4, 0,  0  ])
c1 = array([ 0, 1/4, 1, 1/4 ])

alpha_  = array([ 1/3, 2/3,   0 , 0 ])
alpha_t = array([ 1/2, 1/2,   0 , 0 ])
beta_1  = array([ -1,  4/3,  2/3, 0 ])
beta_2  = array([ -1,  4/3, -1/3, 0 ])
beta_3  = array([  2, -4/3, -2/3, 0 ])
beta_4  = array([ -2,  5/3, -2/3, 1 ])
beta_3t = beta_4t = [0,0,0,0]

def perform_step(t,h,f,g,y,I_1,I_11,I_111,I_10):
	H0 = [None,None,None,None]
	H1 = [None,None,None,None]
	for i in range(4):
		H0[i] = ( y
			+ sum( A0[i][j] * f(t+c0[j]*h,H0[j]) * h      for j in range(4) if A0[i][j])
			+ sum( B0[i][j] * g(t+c1[j]*h,H1[j]) * I_10/h for j in range(4) if B0[i][j])
			)
		
		H1[i] = ( y
			+ sum( A1[i][j] * f(t+c0[j]*h,H0[j]) * h       for j in range(4) if A1[i][j])
			+ sum( B1[i][j] * g(t+c1[j]*h,H1[j]) * sqrt(h) for j in range(4) if B1[i][j])
			)
	
	def solution(alpha_,beta_1,beta_2,beta_3,beta_4):
		X_bar = y + sum( alpha_[i] * f(t+c0[i]*h,H0[i]) * h for i in range(4) )
		for i in range(4):
			X_bar += g( t+c1[i]*h, H1[i] ) * (
							  beta_1[i] * I_1
							+ beta_2[i] * I_11  / sqrt(h)
							+ beta_3[i] * I_10  / h
							+ beta_4[i] * I_111 / h
						)
		return X_bar
	
	X_bar   = solution( alpha_ , beta_1, beta_2, beta_3 , beta_4  )
	X_hat   = solution( alpha_t, beta_1, beta_2, beta_3 , beta_4  )
	X_tilde = solution( alpha_ , beta_1, beta_2, beta_3t, beta_4t )
	
	E_D = X_bar-X_hat
	E_N = X_bar-X_tilde
	
	return X_bar, E_D, E_N

