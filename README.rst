JiTCSDE stands for just-in-time compilation for stochastic differential equations (SDEs).
It makes use of the method described by Rackauckas and Nie.
JiTCSDE is designed in analogy to `JiTCODE <http://github.com/neurophysik/jitcode>`_:
It takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python.
If you want to integrate ordinary or delay differential equations, check out
`JiTCODE <http://github.com/neurophysik/jitcode>`_, or
`JiTCDDE <http://github.com/neurophysik/jitcdde>`_, respectively.

Note that this is work in progress; features and names may change without warning.

* `Documentation <http://jitcsde.readthedocs.io>`_ – Read this to get started and for reference. Don’t miss that some topics are addressed in the `common JiTC*DE documentation <http://jitcde-common.readthedocs.io>`_.

* `Paper <https://doi.org/10.1063/1.5019320>`_ – Read this for the scientific background. Cite this (`BibTeX <https://raw.githubusercontent.com/neurophysik/jitcxde_common/master/citeme.bib>`_) if you wish to give credit or to shift blame.

* `Issue Tracker <http://github.com/neurophysik/jitcsde/issues>`_ – Please report any bugs here. Also feel free to ask for new features.

* `Installation instructions <http://jitcde-common.readthedocs.io/#installation>`_. In most cases, `pip3 install jitcode` or similar should do the job.

This work was supported by the Volkswagen Foundation (Grant No. 88463).

