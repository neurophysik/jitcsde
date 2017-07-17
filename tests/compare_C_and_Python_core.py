#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Creates instances of the Python and C core for the same SDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation.

The argument is the number of runs.
"""

from __future__ import print_function
from jitcsde._python_core import sde_integrator as py_sde_integrator
from jitcsde import t, y, jitcsde

import sympy
import numpy as np
from numpy.testing import assert_allclose
import platform
from random import Random
from sys import argv, stdout

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG","-O0"]

def compare(x,y,rtol=1e-5,atol=1e-5):
	try:
		assert_allclose(x,y,rtol=rtol,atol=atol)
	except AssertionError as error:
		print("\n")
		print (x,y)
		raise error

number_of_runs = int(argv[1])

F = [
	10*(y(1)-y(0)),
	y(0)*(28-y(2))-y(1),
	y(0)*y(1)-8/3.*y(2)
	]

for additive in [True,False]:
	if additive:
		G = [sympy.sympify(i*0.05) for i in range(len(F))]
	else:
		g = [ 0.1*y(i) for i in range(len(F)) ]
	
	RNG = Random()
	
	errors = 0
	
	for realisation in range(number_of_runs):
		print(".", end="")
		stdout.flush()
		
		seed = RNG.randint(0,1000000)
		
		initial_state = np.array([RNG.random() for _ in range(len(F))])
		
		P = py_sde_integrator(
				lambda:F, lambda:G,
				initial_state, 0.0,
				seed=seed, additive=additive
				)
		
		SDE = jitcsde(F,G,additive=additive)
		SDE.compile_C(extra_compile_args=compile_args)
		C = SDE.jitced.sde_integrator(0.0,initial_state,seed)
		
		def get_next_step():
			r = RNG.uniform(1e-7,1e-3)
			P.get_next_step(r)
			C.get_next_step(r)
		
		def time():
			compare(P.t, C.t)
		
		def get_state():
			compare(P.get_state(), C.get_state())
		
		def get_p():
			r = 10**RNG.uniform(-10,-5)
			q = 10**RNG.uniform(-10,-5)
			compare( np.log(P.get_p(r,q)), np.log(C.get_p(r,q)), rtol=1e-2, atol=1e-2 )
		
		def accept_step():
			P.accept_step()
			C.accept_step()
		
		def pin_noise():
			step = RNG.uniform(1e-8,1.0)
			number = RNG.randint(0,5)
			P.pin_noise(number,step)
			C.pin_noise(number,step)
		
		# Many methods cannot be expected to produce anything reasonable without one step being made:
		get_next_step()
		
		actions = [
				get_next_step,
				time,
				get_p,
				accept_step,
				get_state,
				pin_noise
				]
		
		for i in range(10):
			action = RNG.sample(actions,1)[0]
			try:
				action()
			except AssertionError as error:
				print("\n--------------------")
				print("Results did not match in realisation %i in action %i:" % (realisation, i))
				print(action.__name__)
				print("--------------------")
				
				errors += 1
				break
	
	print("Runs with errors: %i / %i" % (errors, number_of_runs))

