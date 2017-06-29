#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

from itertools import chain
import sympy
import numpy as np
from numpy import sqrt
from numpy.testing import assert_allclose

def perform_step(t,h,f,g,y,I_1,I_11,I_111,I_10):
	"""
	Optimised step evaluation. Tested against a human-readable version in test_step_functions.py.
	"""
	fh_1 = f( t      , y                             ) * h
	g_1  = g( t      , y                             )
	fh_2 = f( t+3/4*h, y + 3/4*fh_1 + 3/2*g_1*I_10/h ) * h
	g_2  = g( t+1/4*h, y + 1/4*fh_1 + 1/2*g_1*sqrt(h))
	g_3  = g( t+    h, y +     fh_1 -     g_1*sqrt(h))
	g_4  = g( t+1/4*h, y + 1/4*fh_1 + sqrt(h)*(-5*g_1+3*g_2+1/2*g_3) )
	
	E_N = (1/h/3)* (
		+ (  6*I_10 - 6*I_111 ) * g_1
		+ ( -4*I_10 + 5*I_111 ) * g_2
		+ ( -2*I_10 - 2*I_111 ) * g_3
		+ (           3*I_111 ) * g_4
		)
	I_11dbsqh = I_11/sqrt(h)
	new_y = y + E_N + (1/3)*(
		fh_1 + 2*fh_2
		+ (-3*I_1 - 3*I_11dbsqh ) * g_1
		+ ( 4*I_1 + 4*I_11dbsqh ) * g_2
		+ ( 2*I_1 -   I_11dbsqh ) * g_3
		)
	E_D = (fh_2-fh_1)/6
	
	return new_y, E_D, E_N

def perform_SRA_step(t,h,f,g,y,I_1,I_10):
	"""
	Optimised step evaluation. Tested against a human-readable version in test_step_functions.py.
	"""
	fh_1 = f( t      , y                             ) * h
	g_1  = g( t+h                                    )
	fh_2 = f( t+3/4*h, y + 3/4*fh_1 + 1/2*g_1*I_10/h ) * h
	g_2  = g( t                                      )
	
	E_N = I_10/h*(g_2-g_1)
	new_y = y + E_N + (1/3)*(fh_1 + 2*fh_2) + I_1*g_1
	E_D = (fh_2-fh_1)/6
	
	return new_y, E_D, E_N

class sde_integrator(object):
	def __init__(self,
				f,
				g,
				y,
				t = 0.0,
				f_helpers = (),
				g_helpers = (),
				control_pars = (),
				seed = None,
				additive = False
			):
		self.state = y
		self.n = len(self.state)
		self.t = t
		self.parameters = []
		self.noises = []
		self.noise_index = None
		self.new_y = None
		self.new_t = None
		self.RNG = np.random.RandomState(seed)
		self.additive = additive
		
		from jitcsde import t, y
		Y = sympy.DeferredVector("Y")
		basic_subs = [(y(i),Y[i]) for i in range(self.n)]
		f_substitutions = list(reversed(f_helpers)) + basic_subs
		g_substitutions = list(reversed(g_helpers)) + basic_subs
		
		f_wc = [ entry.subs(f_substitutions).simplify(ratio=1.0) for entry in f() ]
		g_wc = [ entry.subs(g_substitutions).simplify(ratio=1.0) for entry in g() ]
		
		F = sympy.lambdify( [t, Y]+list(control_pars), f_wc )
		self.f = lambda t,Y: np.array(F(t,Y,*self.parameters)).flatten()
		
		if self.additive:
			G = sympy.lambdify( [t]+list(control_pars), g_wc )
			self.g = lambda t: np.array(G(t,*self.parameters)).flatten()
		else:
			G = sympy.lambdify( [t, Y]+list(control_pars), g_wc )
			self.g = lambda t,Y: np.array(G(t,Y,*self.parameters)).flatten()
	
	def set_parameters(self, *parameters):
		self.parameters = parameters
	
	def get_numpy_noise(self,scale):
		# Switching the shape and transposing may look pointless, but ensures comparability of with the C backend for testing purposes.
		return self.RNG.normal(0,scale,(self.n,2)).T

	def Brownian_bridge(self, h_need):
		h,noise = self.noises[self.noise_index]
		h_exc = h - h_need
		factor = h_exc/h
		noise_2 = factor*noise + self.get_numpy_noise(sqrt(factor*h_need))
		noise_1 = noise - noise_2
		self.noises[self.noise_index] = [h_exc,noise_2]
		self.noises.insert( self.noise_index, [h_need,noise_1] )
	
	def append_noise(self, h):
		noise = self.get_numpy_noise(sqrt(h))
		self.noises.append([h,noise])
	
	def get_noise(self, h_need):
		noise_acc = 0
		self.noise_index = 0
		while h_need:
			if self.noise_index < len(self.noises):
				# use existing noise
				h,noise = self.noises[self.noise_index]
				if h <= h_need:
					# completely use interval
					noise_acc += noise
					h_need -= h
					self.noise_index += 1
				else:
					# split interval with Brownian bridge
					self.Brownian_bridge(h_need)
			else:
				# create new noise
				self.append_noise(h_need)
		
		return noise_acc
	
	def pin_noise(self, number, step):
		for _ in range(number):
			self.append_noise(step)

	def get_I(self, h):
		DW, DZ = self.get_noise(h)
		I_11  = ( DW**2 -   h    )/2
		I_111 = ( DW**3 - 3*h*DW )/6
		I_10  = ( DW + DZ/sqrt(3))/2*h
		return DW, I_11, I_111, I_10
	
	def get_I_SRA(self, h):
		DW, DZ = self.get_noise(h)
		I_10  = ( DW + DZ/sqrt(3))/2*h
		return DW, I_10
	
	def get_next_step(self, h):
		if self.additive:
			I = self.get_I_SRA(h)
			self.new_y, E_D, E_N = perform_SRA_step(self.t,h,self.f,self.g,self.state,*I)
		else:
			I = self.get_I(h)
			self.new_y, E_D, E_N = perform_step(self.t,h,self.f,self.g,self.state,*I)
		self.error = abs(E_D) + abs(E_N)
		self.new_t = self.t+h
	
	def get_state(self):
		return self.state
	
	def get_p(self, atol, rtol):
		with np.errstate(divide='ignore',invalid='ignore'):
			return np.nanmax(self.error/(atol + rtol*np.abs(self.new_y)))
	
	def print_noises(self):
		if self.noises:
			for noise in self.noises:
				print("%e\t%e\t%e"%(noise[0],noise[1][0][0],noise[1][1][0]))
		else:
			print("no noise")
		print(self.noise_index)
		print("-------")
	
	def accept_step(self):
		self.state = self.new_y
		self.t = self.new_t
		if self.noise_index is not None:
			del self.noises[:self.noise_index]
		self.noise_index = None
	
	def apply_jump(self,change):
		self.state += change
