import numpy as np
from numpy.testing.utils import assert_allclose
from jitcsde import jitcsde_jump, y, UnsuccessfulIntegration
import platform
from symengine import symbols, exp, Rational
import unittest

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG"]

f = [-y(0)**3 + 4*y(0) + y(0)**2]
g = [5*exp(-y(0)**2+y(0)-Rational(3,2)) + 3]

λ = 1000
initial_value = np.array([1.0])

# Normal noise

result = None

class CompareResults(unittest.TestCase):
	def compare_with_result(self,new_result):
		global result
		if result is None:
			result = new_result
		else:
			assert_allclose(
					result-initial_value,
					new_result-initial_value,
					rtol=1e-3
				)
	
	def setUp(self):
		self.R = np.random.RandomState(42)
		IJI  = lambda t,y: self.R.exponential( 1/λ )
		amp = lambda t,y: self.R.normal( 0.0, 1+1/(1+y**2), (1,) )
		self.SDE = jitcsde_jump( IJI, amp, f, g )
	
	def test_default(self):
		self.SDE.set_seed(42)
		self.SDE.set_initial_value(initial_value,0.0)
	
	def test_reproducability(self):
		self.test_default()
	
	def test_Python_core(self):
		self.test_default()
		self.SDE.generate_lambdas()
	
	def tearDown(self):
		self.SDE.check()
		new_result = self.SDE.integrate(0.001)
		self.compare_with_result(new_result)

IJI = lambda t,y: 1
amp = lambda t,y: np.array([1])

class TestStrat(unittest.TestCase):
	def testError(self):
		with self.assertRaises(NotImplementedError):
			SDE = jitcsde_jump( IJI, amp, f, g, ito=False )

class TestCheck(unittest.TestCase):
	def test_check_index_negative(self):
		SDE = jitcsde_jump( IJI, amp, [y(0)], [y(-1)] )
		with self.assertRaises(ValueError):
			SDE.check()
	
	def test_check_index_too_high(self):
		SDE = jitcsde_jump( IJI, amp, [y(0)], [y(1)] )
		with self.assertRaises(ValueError):
			SDE.check()
	
	def test_check_undefined_variable(self):
		x = symbols("x")
		SDE = jitcsde_jump( IJI, amp, [y(0)], [x] )
		with self.assertRaises(ValueError):
			SDE.check()
	
	def test_wrong_amp_output_type(self):
		amp = lambda t,y: 1
		SDE = jitcsde_jump( IJI, amp, f, g )
		with self.assertRaises(ValueError):
			SDE.check()
	
	def test_wrong_amp_output_size(self):
		amp = lambda t,y: np.array([1,2])
		SDE = jitcsde_jump( IJI, amp, f, g )
		with self.assertRaises(ValueError):
			SDE.check()

# Boilerplate

if __name__ == "__main__":
	unittest.main(buffer=True)

