from jitcsde import jitcsde, y
from jitcode import jitcode
import numpy as np

"""
Tests the SDE integrator by comparing the trajectories of:

* An ODE with the dynamics on an attracting limit-cycle.
* An SDE which is like the ODE, just that the speed of the phase-space flow is subject to stochastic fluctuations.

Inspired by:
	https://math.stackexchange.com/q/2712378
"""

# The dynamics used for testing
# -----------------------------

a = -0.025
b =  0.01
c =  0.02
d =  5     # to make axis scale comparably

F = [
		y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1)/d,
		d*(b*y(0) - c/d*y(1)),
	]

initial_state = np.random.random(2)

# Integrating the ODE
#--------------------

ODE = jitcode(F,verbose=False)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state,0.0)

times = 500+np.arange(0,200,0.005)
ode_data = np.vstack([ODE.integrate(time) for time in times])

# Integrating the SDE
# -------------------

SDE = jitcsde( [1*f for f in F], [0.01*f for f in F], verbose=False )
SDE.set_initial_value(initial_state,0.0)

times = 500+np.arange(0,300,1)
sde_data = np.vstack([SDE.integrate(time) for time in times])

# The actual test
# ---------------

# Each result of the SDE integration should be near a result of the ODE integration.

# Obtain an accuracy threshold from the maximum distance of consecutive ODE samples. Note that the ODE was sampled with a much finer time step to densely capture the attractor.
self_distances = np.linalg.norm(ode_data[1:,:]-ode_data[:-1,:],axis=1)
threshold = 4*np.max(self_distances)

# All distances between SDE samples and ODE samples
distances = np.linalg.norm(
		sde_data[:,None,:] - ode_data[None,:,:],
		axis = 2
	)

assert np.all(np.any(distances<threshold,axis=1))

