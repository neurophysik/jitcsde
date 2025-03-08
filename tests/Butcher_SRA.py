from numpy import array


A0 = array([
		[ 0 , 0],
		[3/4, 0],
	])

B0 = array([
		[ 0 , 0],
		[1/2, 0],
	])

c0 = array([ 0, 3/4])
c1 = array([ 1,  0 ])

α  = array([ 1/3, 2/3 ])
αt = array([ 1/2, 1/2 ])
β1 = array([ 1,  0])
β2 = array([-1,  1])
β2t = [0,0]

def perform_step(t,h,f,g,y,I_1,I_10):
	H0 = [None,None]
	for i in range(2):
		H0[i] = ( y
			+ sum( A0[i][j] * f(t+c0[j]*h,H0[j]) * h      for j in range(2) if A0[i][j])
			+ sum( B0[i][j] * g(t+c1[j]*h      ) * I_10/h for j in range(2) if B0[i][j])
			)
	
	def solution(α,β1,β2):
		X_bar = y + sum( α[i] * f(t+c0[i]*h,H0[i]) * h for i in range(2) )
		for i in range(2):
			X_bar += g( t+c1[i]*h ) * (
							β1[i] * I_1
							+ β2[i] * I_10  / h
						)
		return X_bar
	
	X_bar   = solution( α , β1, β2  )
	X_hat   = solution( αt, β1, β2  )
	X_tilde = solution( α , β1, β2t )
	
	E_D = X_bar-X_hat
	E_N = X_bar-X_tilde
	
	return X_bar, E_D, E_N

