#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

from itertools import chain
import sympy
import numpy as np

class dde_integrator(object):
	def __init__(self,
				f,
				g,
				y,
				t = 0.0,
				helpers = (),
				control_pars = (),
			):
		self.y = y
		self.n = len(self.y)
		self.t = t
		self.parameters = []
		self.past_noise = []
		
		from jitcode import t, y # TODO: update, once finished
		Y = sympy.DeferredVector("Y")
		substitutions = list(helpers[::-1]) + [(y(i),Y[i]) for i in range(self.n)]
		
		f_wc = [ entry.subs(substitutions).simplify(ratio=1.0) for entry in f() ]
		g_wc = [ entry.subs(substitutions).simplify(ratio=1.0) for entry in g() ]
		
		F = sympy.lambdify( [t, Y]+list(control_pars), f_wc )
		G = sympy.lambdify( [t, Y]+list(control_pars), g_wc )
		
		self.f = lambda t,Y: np.array(F(t,Y,*self.parameters)).flatten()
		self.g = lambda t,Y: np.array(G(t,Y,*self.parameters)).flatten()
	
	def set_parameters(self, *parameters):
		self.parameters = parameters
	
	def get_t(self):
		return self.t
	
	def get_noise(self, t, h):
		if False: #t_in_past_noise:
			pass # TODO
		else:
			DW = np.random.normal( 0, sqrt(h), self.n )
			DZ = np.random.normal( 0, sqrt(h), self.n )
#		self.add_past_noise(t, h, DW, DZ)
		return DW, DZ
	
	def get_I(self, t, h):
		DW, DZ = self.get_noise(t,h)
		I_11  = ( DW**2 -   h    )/2
		I_111 = ( DW**3 - 3*h*DW )/6
		I_10  = ( DW + DZ/sqrt(3))/2*h
		return DW, I_11, I_111, I_10
	
	def get_next_step(self, h):
		I_1, I_11, I_111, I_10 = self.get_I(self.t,h)
		
		t = self.t
		y = self.y
		
		H0_1 = H1_1 = H0_3 = H0_4 = y
		H0_2 = y + 3/4*f(t,H0_1)*h + 3/2*g(t,H1_1)*I_10/h
		H1_2 = y + 1/4*f(t,H0_1)*h + 1/2*g(t,H1_1)*sqrt(h)
		H1_3 = y +     f(t,H0_1)*h -     g(t,H1_1)*sqrt(t)
		H1_4 = y + 1/4*f(t,H0_3)*h + sqrt(h)*(
				-5*g(t,H1_1)+ 3*g(t+h/4,H1_2) + 1/2*g(t+h,H1_3) )
		self.new_y = y + (
			  1/3 * f(t,H0_1) * h  +  2/3 * f(t+3/4*h,H0_2) * h
			+ (  -  I_1 -     I_11/sqrt(h) +  2 *I_10/h +  2 *I_111/h ) * g(t    ,H1_1)
			+ ( 4/3*I_1 + 4/3*I_11/sqrt(h) - 4/3*I_10/h + 5/3*I_111/h ) * g(t+h/4,H1_2)
			+ ( 2/3*I_1 - 1/3*I_11/sqrt(h) - 2/3*I_10/h - 2/3*I_111/h ) * g(t+h  ,H1_3)
			+ (                                               I_111/h ) * g(t+h/4,H1_4)
			)
		self.error = h/6*f(t,H0_1) - h/6*f(t+3*h/4,H0_2) + (
			+ (   2 *I_10/h +  2 *I_111/h ) * g(t    ,H1_1)
			+ ( -4/3*I_10/h + 5/3*I_111/h ) * g(t+h/4,H1_2)
			+ ( -2/3*I_10/h - 2/3*I_111/h ) * g(t+h  ,H1_3)
			+ (                   I_111/h ) * g(t+h/4,H1_4)
			)
		
		self.new_t = t+h
	
	def get_p(self, atol, rtol):
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.nanmax(np.abs(self.error)/(atol + rtol*np.abs()))
	
	def accept_step(self):
		self.y = self.new_y
		self.t = self.new_t


