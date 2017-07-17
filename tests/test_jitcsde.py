from __future__ import print_function, division
import numpy as np
from numpy.testing.utils import assert_allclose
from jitcsde import jitcsde, y
import platform
import sympy
import unittest

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG"]

# Ensures that all kinds of formatting the input actually work and produce the same result. The correctness of this result itself is checked in validation_test.py.

f = [-y(0)**3 + 4*y(0) + y(0)**2]
g = [5*sympy.exp(-y(0)**2+y(0)-1.5) + 3.0]
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
					rtol=1e-3)
	
	def setUp(self):
		self.SDE = jitcsde(f,g)
	
	def test_default(self):
		self.SDE.set_seed(42)
		self.SDE.set_initial_value(initial_value,0.0)
	
	def test_numpy_rng(self):
		self.SDE.compile_C(numpy_rng=True,extra_compile_args=compile_args)
		self.test_default()
	
	def test_reproducability(self):
		self.test_default()
	
	def test_Python_core(self):
		self.test_default()
		self.SDE.generate_lambdas()
	
	def test_no_simplify(self):
		self.test_default()
		self.SDE.compile_C(simplify=False,extra_compile_args=compile_args)
	
	def test_no_cse(self):
		self.test_default()
		self.SDE.compile_C(do_cse=False,extra_compile_args=compile_args)
	
	def test_save_and_load(self):
		filename = self.SDE.save_compiled(overwrite=True)
		self.SDE = jitcsde(module_location=filename,n=len(f))
		self.test_default()

	def tearDown(self):
		new_result = self.SDE.integrate(0.001)
		self.compare_with_result(new_result)

y2_m_y, state, exp_term, polynome = sympy.symbols("y2_m_y, state, exp_term, polynome")

helpers = [
	(state, y(0)),
	(y2_m_y, state**2-state),
	(polynome, -state**3 + 5*state + y2_m_y),
	(exp_term, sympy.exp(-y2_m_y-1.5)),
	]
f_helpers = [helpers[i] for i in (0,1,2)]
g_helpers = [helpers[i] for i in (0,1,3)]
f_with_helpers = [polynome]
g_with_helpers = [5*exp_term + 3.0]

class TestHelpers(CompareResults):
	def setUp(self):
		self.SDE = jitcsde(f_with_helpers,g_with_helpers,helpers=helpers,g_helpers="same")

class TestPrefilteredHelpers(CompareResults):
	def setUp(self):
		self.SDE = jitcsde(f_with_helpers,g_with_helpers,helpers=f_helpers,g_helpers=g_helpers)

class TestAutofilteringHelpers(CompareResults):
	def setUp(self):
		self.SDE = jitcsde(f_with_helpers,g_with_helpers,helpers=helpers,g_helpers="auto")


# Additive Noise

SRA_result = None
g_add = [sympy.sympify(3)]

class TestAdditive(CompareResults):
	def compare_with_result(self,new_result):
		global SRA_result
		if SRA_result is None:
			SRA_result = new_result
		else:
			assert_allclose(
					SRA_result-initial_value,
					new_result-initial_value,
					rtol=1e-3)
	
	def setUp(self):
		self.SDE = jitcsde(f,g_add)

class TestAdditiveAndHelpers(TestAdditive):
	def setUp(self):
		self.SDE = jitcsde(f_with_helpers,g_add,helpers=helpers,g_helpers="same")

class TestAdditiveAndFilteredHelpers(TestAdditive):
	def setUp(self):
		self.SDE = jitcsde(f_with_helpers,g_add,helpers=helpers,g_helpers="auto")


# Boilerplate

if __name__ == "__main__":
	unittest.main(buffer=True)

