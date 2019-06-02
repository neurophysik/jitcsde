"""
Tests whether things works independent of where symbols are imported from.
"""

import jitcsde
import jitcsde.sympy_symbols
import sympy
import symengine
import random

symengine_manually = [
		symengine.Symbol("t",real=True),
		symengine.Function("y",real=True),
		symengine.cos,
	]

sympy_manually = [
		sympy.Symbol("t",real=True),
		sympy.Function("y",real=True),
		sympy.cos,
	]

jitcsde_provisions = [
		jitcsde.t,
		jitcsde.y,
		symengine.cos,
	]

jitcsde_sympy_provisions = [
		jitcsde.sympy_symbols.t,
		jitcsde.sympy_symbols.y,
		symengine.cos,
	]

mixed = [
		jitcsde.sympy_symbols.t,
		jitcsde.y,
		sympy.cos,
	]

seed = int(random.getrandbits(32))
results = set()

for t,y,cos in [
			symengine_manually,
			sympy_manually,
			jitcsde_provisions,
			jitcsde_sympy_provisions,
			mixed
		]:
	SDE = jitcsde.jitcsde( [cos(t)*y(0)], [cos(t)*y(0)/10], verbose=False )
	SDE.set_seed(seed)
	SDE.set_initial_value([1.0],0.0)
	
	result = SDE.integrate(10)[0]
	results.add(result)

assert len(results)==1

