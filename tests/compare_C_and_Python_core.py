"""
Creates instances of the Python and C core for the same SDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation.

The argument is the number of runs.
"""

import platform
from random import Random
from sys import argv

import numpy as np
import symengine
from numpy.testing import assert_allclose

from jitcsde import jitcsde, y
from jitcsde._python_core import sde_integrator as py_sde_integrator


if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = [*DEFAULT_COMPILE_ARGS,"-g","-UNDEBUG","-O1"]

def compare(x,y,rtol=1e-4,atol=1e-4):
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
rng = Random()

for additive in [False,True]:
	if additive:
		G = [ symengine.sympify(i*0.05) for i in range(len(F)) ]
	else:
		G = [ 0.1*y(i) for i in range(len(F)) ]
	
	errors = 0
	
	for realisation in range(number_of_runs):
		print( ".", end="", flush=True )
		
		seed = rng.randint(0,1000000)
		
		initial_state = np.array([rng.random() for _ in range(len(F))])
		
		P = py_sde_integrator(
				lambda: F, lambda G=G: G,
				initial_state, 0.0,
				seed=seed, additive=additive
				)
		
		SDE = jitcsde(F,G,additive=additive)
		SDE.compile_C(extra_compile_args=compile_args,chunk_size=1)
		C = SDE.jitced.sde_integrator(0.0,initial_state,seed)
		
		def get_next_step(P=P, C=C):
			r = rng.uniform(1e-7,1e-3)
			P.get_next_step(r)
			C.get_next_step(r)
		
		def time(P=P, C=C):
			compare(P.t, C.t)
		
		def get_state(P=P, C=C):
			compare(P.get_state(), C.get_state())
		
		def get_p(P=P, C=C):
			r = 10**rng.uniform(-10,-5)
			q = 10**rng.uniform(-10,-5)
			compare( np.log(P.get_p(r,q)), np.log(C.get_p(r,q)), rtol=1e-2, atol=1e-2 )
		
		def accept_step(P=P, C=C):
			P.accept_step()
			C.accept_step()
		
		def pin_noise(P=P, C=C):
			step = rng.uniform(1e-8,1.0)
			number = rng.randint(0,5)
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
			action = rng.sample(actions,1)[0]
			try:
				action()
			except AssertionError:
				print("\n--------------------")
				print(f"Results did not match in realisation {realisation} in action {i}:")
				print(action.__name__)
				print("--------------------")
				
				errors += 1
				break
	
	print(f"Runs with errors: {errors} / {number_of_runs}")
