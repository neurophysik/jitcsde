#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose we want to integrate Lorenz oscillator each of whose components is subject to a diffusion that amounts to :math:`p` of the respective component, i.e.:

.. math::

	f(y) = \\left(
	\\begin{matrix}
		σ (y_1-y_0)\\\\
		y_0 (ρ-y_2)-y_1\\\\
		y_0 y_1 - β y_2
	\\end{matrix} \\right),

	\qquad
	
	g(y) = \\left(
	\\begin{matrix}
		p y_0 \\\\
		p y_1 \\\\
		p y_2
	\\end{matrix} \\right),

.. math::

First we do some importing and define the constants, which we want to be :math:`ρ = 28`, :math:`σ = 10`, :math:`β =\\frac{8}{3}`, and :math:`p=0.1`:

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 75-82

Amongst our imports was the symbol for the state (`y`), which have to be used to write down the differential equation such that JiTCSDE can process it.
(For an explicitly time-dependent differential equation, it is also possible to import `t`.)
Using this, we can write down the drift factor :math:`f` as a list of expressions:

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 84-88

For the diffusion factor :math:`g`, we need the same, but due to :math:`g`’s regularity, we can employ a list comprehension:

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 90

We can then initiate JiTCSDE:

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 92

We want the initial condition to be random and the integration to start at :math:`t = 0`.

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 94-95

Finally, we can perform the actual integration. In our case, we integrate for 100 time units with a sampling rate of 0.1 time units.
`integrate` returns the state after integration, which we collect in the list `data`.
Finally, we use `numpy.savetxt` to store this to the file `timeseries.dat`.

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 97-100

Taking everything together, our code is:

.. literalinclude:: ../examples/noisy_lorenz.py
	:dedent: 1
	:lines: 75-100
"""

if __name__ == "__main__":
	from jitcsde import y, jitcsde
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
	
	SDE = jitcsde(f,g)
	
	initial_state = numpy.random.random(3)
	SDE.set_initial_value(initial_state,0.0)
	
	data = []
	for time in numpy.arange(0.0, 100.0, 0.01):
		data.append( SDE.integrate(time) )
	numpy.savetxt("timeseries.dat", data)

