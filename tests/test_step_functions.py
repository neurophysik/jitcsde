"""
Compares results of the step function from the Python core with the one derived from the Butcher tableaus.
"""

import unittest

import numpy as np
from Butcher import perform_step as step_2
from Butcher_SRA import perform_step as SRA_step_2

from jitcsde._python_core import perform_SRA_step as SRA_step_1, perform_step as step_1


n = 100

class TestStepFunctions(unittest.TestCase):
	def test_step_function(self):
		rng = np.random.default_rng(seed=42)

		# some random coefficients:
		R = rng.uniform( 0.5, 2, (4,n) )
		P = rng.integers( -2, 3, (8,n) )
		# random functions
		def f(t,y):
			return R[0]*y**P[0]*t**P[1] + R[1]*y**P[2]*t**P[3]
		def g(t,y):
			return R[2]*y**P[4]*t**P[5] + R[3]*y**P[6]*t**P[7]
		
		t    = rng.uniform( 0.5, 2 )
		h    = rng.uniform( 0.1, 2 )
		args = rng.uniform( 0.5, 2, (5,n) )
		result_1 = step_1(t,h,f,g,*args)
		result_2 = step_2(t,h,f,g,*args)
		
		np.testing.assert_allclose(result_1,result_2,rtol=1e-5,atol=1e-10)
	
	def test_SRA_step_function(self):
		rng = np.random.default_rng(seed=42)

		# some random coefficients:
		R = rng.uniform( 0.5, 2, (4,n) )
		P = rng.integers( -2, 3, (6,n) )
		# random functions
		def f(t,y):
			return R[0]*y**P[0]*t**P[1] + R[1]*y**P[2]*t**P[3]
		def g(t):
			return R[2]*t**P[4] + R[3]**t**P[5]
		
		t    = rng.uniform( 0.5, 2 )
		h    = rng.uniform( 0.1, 2 )
		args = rng.uniform( 0.5, 2, (3,n) )
		result_1 = SRA_step_1(t,h,f,g,*args)
		result_2 = SRA_step_2(t,h,f,g,*args)
		
		np.testing.assert_allclose(result_1,result_2,rtol=1e-5,atol=1e-10)

if __name__ == "__main__":
	unittest.main(buffer=True)

