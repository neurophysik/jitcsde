from __future__ import print_function, division
import numpy as np
from jitcsde._python_core import sde_integrator
from numpy.testing import assert_allclose, assert_array_equal
import unittest

test_noise = []
for _ in range(10):
	test_noise.append([
		np.random.exponential(),
		np.random.normal(2.0),
	])
max_h = sum(x[0] for x in test_noise)

class TestNoiseMemory(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.SDE = sde_integrator(lambda:[],lambda:[],[1])
		self.SDE.noise_dims = 1
	
	def setUp(self):
		self.SDE.noises = list(test_noise)
	
	def test_unchanged(self):
		# checks that the given noise was not affected
		control_h = 0
		control_DW = 0
		for h,DW in test_noise:
			control_h  += h
			control_DW += DW
			actual_DW = self.SDE.get_noise(control_h)
			print(control_h, self.SDE.noises)
			assert_allclose(actual_DW, control_DW)
	
	def test_insertion(self):
		intermediate_h = np.random.uniform(0,max_h)
		first  = self.SDE.get_noise(intermediate_h)
		second = self.SDE.get_noise(intermediate_h)
		assert_array_equal(first,second)
		self.test_unchanged()
	
	def test_extension(self):
		external_h = np.random.uniform(max_h,2*max_h)
		first  = self.SDE.get_noise(external_h)
		second = self.SDE.get_noise(external_h)
		assert_array_equal(first,second)
		self.test_unchanged()
	
	def test_size_of_noise_memory(self):
		control_h = np.random.uniform(0,max_h)
		self.SDE.get_noise(control_h)
		self.SDE.accept_step()
		current_h = sum(x[0] for x in self.SDE.noises)
		self.assertAlmostEqual(current_h, max_h-control_h)

if __name__ == "__main__":
	unittest.main(buffer=True)

