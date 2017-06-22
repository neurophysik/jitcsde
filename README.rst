JiTCSDE stands for just-in-time compilation for stochastic differential equations (SDEs).
It makes use of the method described by Rackauckas and Mie.
JiTCSDE is designed in analogy to `JiTCODE <http://github.com/neurophysik/jitcode>`_:
It takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python.

Note that this is work in progress; features and names may change without warning.

* `Documentation <http://jitcsde.readthedocs.io>`_

* `Issue Tracker <http://github.com/neurophysik/jitcsde/issues>`_

* Download from `PyPI <http://pypi.python.org/pypi/jitcsde>`_ or just ``pip install jitcsde``.

This work was supported by the Volkswagen Foundation (Grant No. 88463).

