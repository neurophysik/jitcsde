#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Tests several incarnations of the integrator by checking whether the Kramers–Moyal coefficients as estimated from the time series comply with the theoretical expectation. This test produces false negatives from time to time, which is inevitable if we do not want false positives to be too likely. In case of a failure, the specific test is re-run. If the number of re-runs for a specific test or the total number of re-runs are too high, it’s time to worry.
"""

from __future__ import print_function, division
import numpy as np
from jitcsde._python_core import sde_integrator
from jitcsde import jitcsde, t, y
import sympy
from kmc import KMC
from sys import stdout

N  = 100000
kmax = 3
thresholds = [0.6, 0.6, 0.5, 0.0]

times = lambda dt: np.arange(dt,(N+1)*dt,dt)

class KMC_Error(Exception):
	pass

scenarios = [
	{
		"F": -y(0)**3 + 4*y(0) + y(0)**2,
		"G": 5*sympy.exp(-y(0)**2+y(0)-1.5) + 3.0,
		"additive": False
	},
	{
		"F": -y(0)**3 + 4*y(0) + y(0)**2,
		"G": sympy.sympify(3),
		"additive": True
	}
	]

def test_python_core(
		scenario,
		dt = 0.0001,
		each_step = lambda SDE:0
		):
	SDE = sde_integrator(
			lambda:[scenario["F"]],
			lambda:[scenario["G"]],
			np.array([0]),
			additive=scenario["additive"]
			)
	
	for _ in range(N):
		each_step(SDE)
		SDE.get_next_step(dt)
		SDE.accept_step()
		yield SDE.get_state()[0]

def test_integrator(scenario,dt=0.001,python=False,pin=False):
	SDE = jitcsde( [scenario["F"]], [scenario["G"]], verbose=False )
	SDE.set_initial_value( np.array([0.0]) )
	
	if pin:
		size = np.random.exponential(dt)
		number = int(times(dt)[-1]/size)
		SDE.pin_noise(number,size)
	
	if python:
		SDE.generate_lambdas()
	else:
		SDE.compile_C()
	
	for t in times(dt):
		yield SDE.integrate(t)

def cases(scenario):
	dt = 0.0001
	yield dt, lambda: test_python_core(scenario, dt), "Python core"
	
	# Tests the noise memory by making random request in-between the steps.
	each_step = lambda SDE: SDE.get_noise(np.random.exponential(dt))
	yield (
		dt,
		lambda: test_python_core(scenario, dt, each_step),
		"Python core with additional random noise requests"
	)
	
	for dt in (0.0001,0.0003):
		for python in (False,True):
			for pin in (False,True):
				name = "integrator with dt=%f" % dt
				name += " and %s" % ("Python" if python else "C")
				name += " and noise pinning" if pin else ""
				results = lambda: test_integrator(scenario,dt,python)
				yield dt, results, name

def kmc_test(dt, result):
	F = scenario["F"]
	G = scenario["G"]
	
	# Theoretical expectation
	M = [
		sympy.lambdify( y(0), F + G*sympy.diff(G,y(0))/2 ),
		sympy.lambdify( y(0), G**2 ),
		lambda x: 0,
		lambda x: 0
		]
	
	# Numerical estimate
	# bins, *kmcs = KMC( result(), dt, kmax=kmax )
	kmc_result = KMC( result(), dt, kmax=kmax )
	bins = kmc_result[0]
	kmcs = kmc_result[1:]
	
	# Comparing the two
	for k in range(kmax):
		good = 0
		for X,value,error in zip(bins,*kmcs[k]):
			theory = M[k](X)
			if value-error < theory < value+error :
				good += 1
		if good < thresholds[k]*len(bins):
			raise KMC_Error

retries = 0

for scenario in scenarios:
	for dt, results, name in cases(scenario):
		for _ in range(5):
			try:
				with np.errstate(invalid='ignore'):
					kmc_test(dt,results)
			except KMC_Error:
				retries += 1
				print("R", end="")
			else:
				print(".", end="")
				break
			finally:
				stdout.flush()
		else:
			raise AssertionError("Testing %s failed five times. Something is probably really wrong" % name)
print("")

if retries:
	print("Number of reruns: %i. This number should only rarely be larger than 5."% retries)

