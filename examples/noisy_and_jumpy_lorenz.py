#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
As an example, suppose that we want to add jumps to the noisy Lorenz oscillator from `example`.
These shall have exponentially distributed waiting times (i.e. they are a Poisson process) with a scale parameter :math:`β=1.0`.
A function sampling such waiting times (inter-jump intervals) is:

.. literalinclude:: ../examples/noisy_and_jumpy_lorenz.py
	:dedent: 1
	:lines: 49-50

Note that since our waiting times are neither state- nor time-dependent, we do not use the `time` and `state` argument.
Next, we want the jumps to only apply to the last component, being normally distributed with zero mean and the current amplitude of this component as a standard deviation.
A function producing such a jump is:

.. literalinclude:: ../examples/noisy_and_jumpy_lorenz.py
	:dedent: 1
	:lines: 52-57

Finally, we initialise the integrator using `jitcsde_jump` instead of `jitcsde` with the previously defined functions as additional arguments:

.. literalinclude:: ../examples/noisy_and_jumpy_lorenz.py
	:dedent: 1
	:lines: 59

Everything else remains unchanged.
See `the sources <https://raw.githubusercontent.com/neurophysik/jitcsde/master/examples/noisy_and_jumpy_lorenz.py>`_ for the full example.
"""

if __name__ == "__main__":
	from jitcsde import y, jitcsde_jump
	import numpy
	import sympy
	
	ρ = 28
	σ = 10
	β = sympy.Rational(8,3)
	p = 0.1
	
	f = [
		σ * (y(1)-y(0)),
		y(0)*(ρ-y(2)) - y(1),
		y(0)*y(1) - β*y(2)
		]
	
	g = [ p*y(i) for i in range(3) ]
	
	def IJI(time,state):
		return numpy.random.exponential(1.0)
	
	def jump(time,state):
		return numpy.array([
				0.0,
				0.0,
				numpy.random.normal(0.0,abs(state[2]))
			])

	SDE = jitcsde_jump(IJI,jump,f,g)
	
	initial_state = numpy.random.random(3)
	SDE.set_initial_value(initial_state,0.0)
	
	data = []
	for time in numpy.arange(0.0, 100.0, 0.01):
		data.append( SDE.integrate(time) )
	numpy.savetxt("timeseries.dat", data)

