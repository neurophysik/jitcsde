# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# include <Python.h>
# include <numpy/arrayobject.h>
# include <math.h>
# include <structmember.h>
# include <assert.h>
# include <stdbool.h>

# define TYPE_INDEX NPY_DOUBLE

{% if not numpy_rng %}
# include "random_numbers.c"
{% endif %}

typedef struct noise_item
{
	double h;
	double DW[{{n}}];
	double DZ[{{n}}];
	struct noise_item * next;
} noise_item;

typedef struct
{
	PyObject_HEAD
	noise_item * noises;
	noise_item * current_noise;
	noise_item * last_noise;
	double state[{{n}}];
	double t;
	double new_state[{{n}}];
	double new_t;
	double error[{{n}}];
	{% for control_par in control_pars %}
	double parameter_{{control_par}};
	{% endfor %}
	{% if numpy_rng %}
	PyObject * RNG;
	PyObject * noise_function;
	PyObject * noise_size;
	{% else %}
	rk_state * RNG;
	{% endif %}
} sde_integrator;

/* void print_noises(sde_integrator * const self)
{
	int noise_index = 0;
	bool found = false;
	if (self->noises)
		for (noise_item * cn=self->noises; cn; cn=cn->next)
		{
			printf("%e\t%e\t%e\n", cn->h, cn->DW[0], cn->DZ[0]);
			found |= (cn==self->current_noise);
			if (!found)
				noise_index++;
		}
	else
		printf("no noise\n");
	printf("%i\n", noise_index);
	printf("========\n");
} */

static inline void * safe_malloc(size_t size)
{
	void * pointer = malloc(size);
	if (pointer == NULL)
		PyErr_SetString(PyExc_MemoryError,"Could not allocate memory.");
	return pointer;
}

static inline noise_item * malloc_noise(void)
{
	return safe_malloc(sizeof(noise_item));
}

noise_item * insert_noise_after_current(sde_integrator * const self)
{
	noise_item * new_noise = malloc_noise();
	assert(new_noise!=NULL);
	
	if (self->current_noise)
	{
		assert(self->noises);
		new_noise->next = self->current_noise->next;
		self->current_noise->next = new_noise;
	}
	else
	{
		assert(!self->noises);
		self->current_noise = self->noises = new_noise;
		new_noise->next = NULL;
	}
	
	if (new_noise->next==NULL)
		self->last_noise = new_noise;
	return new_noise;
}

noise_item * append_noise_item(sde_integrator * const self)
{
	noise_item * new_noise = malloc_noise();
	assert(new_noise!=NULL);
	new_noise->next = NULL;
	
	if (self->noises)
		self->last_noise->next = new_noise;
	else
		self->noises = new_noise;
	
	self->last_noise = new_noise;
	return new_noise;
}

{% if numpy_rng %}

void get_gauss(
	sde_integrator * const self,
	double const scale,
	double DW[{{n}}],
	double DZ[{{n}}]
	)
{
	PyArrayObject * noise = (PyArrayObject *) PyObject_CallFunction(
			self->noise_function,
			"ddO",
			0.0,
			scale,
			self->noise_size
		);
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
	{
		DW[i] = * (double *) PyArray_GETPTR2(noise,i,0);
		DZ[i] = * (double *) PyArray_GETPTR2(noise,i,1);
	}
	
	Py_DECREF(noise);
}

{% else %}


void get_gauss(
	sde_integrator * const self,
	double const scale,
	double DW[{{n}}],
	double DZ[{{n}}]
	)
{
	for (int i=0; i<{{n}}; i++)
		rk_gauss( self->RNG, &DW[i], &DZ[i], scale );
}

{% endif %}

void remove_first_noise(sde_integrator * const self)
{
	if (self->noises)
	{
		noise_item * old_first_noise = self->noises;
		self->noises = old_first_noise->next;
		free(old_first_noise);
		if (self->noises==NULL)
			self->last_noise = NULL;
	}
}

void append_noise(sde_integrator * const self, double const h_need)
{
	noise_item * new_noise = append_noise_item(self);
	self->current_noise = new_noise;
	new_noise->h = h_need;
	get_gauss( self, sqrt(h_need), new_noise->DW, new_noise->DZ );
}

void Brownian_bridge(sde_integrator * const self, double const h_need)
{
	noise_item * noise_1 = self->current_noise;
	noise_item * noise_2 = insert_noise_after_current(self);
	double h = noise_1->h;
	double h_exc = h - h_need;
	double const factor = h_exc/h;
	
	noise_2->h = h_exc;
	get_gauss( self, sqrt(factor*h_need), noise_2->DW, noise_2->DZ );
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
	{
		noise_2->DW[i] += noise_1->DW[i]*factor;
		noise_2->DZ[i] += noise_1->DZ[i]*factor;
	}
	
	noise_1->h = h_need;
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
	{
		noise_1->DW[i] -= noise_2->DW[i];
		noise_1->DZ[i] -= noise_2->DZ[i];
	}
}

void get_noise(
		sde_integrator * const self,
		double h_need,
		double DW_acc[{{n}}],
		double DZ_acc[{{n}}]
		)
{
	bool initialised = false;
	
	self->current_noise = self->noises;
	while (h_need)
	{
		if (self->current_noise)
		{
			if (self->current_noise->h <= h_need)
			{
				if (initialised)
					#pragma omp parallel for schedule(dynamic, {{chunk_size}})
					for (int i=0; i<{{n}}; i++)
					{
						DW_acc[i] += self->current_noise->DW[i];
						DZ_acc[i] += self->current_noise->DZ[i];
					}
				else
				{
					initialised = true;
					#pragma omp parallel for schedule(dynamic, {{chunk_size}})
					for (int i=0; i<{{n}}; i++)
					{
						DW_acc[i] = self->current_noise->DW[i];
						DZ_acc[i] = self->current_noise->DZ[i];
					}
				}
				h_need -= self->current_noise->h;
				self->current_noise = self->current_noise->next;
			}
			else
				Brownian_bridge(self, h_need);
		}
		else
			append_noise(self, h_need);
	}
	if (!initialised)
		#pragma omp parallel for schedule(dynamic, {{chunk_size}})
		for (int i=0; i<{{n}}; i++)
			DW_acc[i] = DZ_acc[i] = 0;
}

static PyObject * pin_noise(sde_integrator * self, PyObject * args)
{
	unsigned int number;
	double step;
	
	if (!PyArg_ParseTuple(args,"Id",&number,&step))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	for (unsigned int i=0; i<number; i++)
		append_noise(self, step);
	
	Py_RETURN_NONE;
}

void get_I(
		sde_integrator * const self,
		double const h,
		double DW[{{n}}],
		{% if not additive %}
		double I_11dbsqh[{{n}}],
		double I_111[{{n}}],
		{% endif %}
		double I_10[{{n}}]
		)
{
	double DZ[{{n}}];
	get_noise(self, h, DW, DZ);
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
	{
		{% if not additive %}
		I_11dbsqh[i] = ( DW[i]*DW[i]	   - h				  ) * 0.5/sqrt(h);
		I_111[i]	 = ( DW[i]*DW[i]*DW[i] - 3*h*DW[i]		  ) * (1./6.)	;
		{% endif %}
		I_10 [i]	 = ( DW[i]			 + DZ[i]*(1./sqrt(3)) ) * (h /2.)	;
	}
}

{% if control_pars|length %}

static PyObject * set_parameters(sde_integrator * const self, PyObject * args)
{
	if (!PyArg_ParseTuple(
		args,
		"{{'d'*control_pars|length}}"
		{% for control_par in control_pars %}
		, &(self->parameter_{{control_par}})
		{% endfor %}
		))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	Py_RETURN_NONE;
}

{% endif %}

npy_intp dim[1] = { {{n}} };

{% if callbacks|length %}
PyObject * n_dim_read_only_array_from_data(void * data) {
	PyObject * result = PyArray_SimpleNewFromData( 1, dim, TYPE_INDEX, data );
	PyArray_CLEARFLAGS( (PyArrayObject *) result, NPY_ARRAY_WRITEABLE );
	return result;
}

static inline double callback(PyObject * Python_function, PyObject * arglist)
{
	PyObject * py_result = PyObject_CallObject(Python_function,arglist);
	Py_DECREF(arglist);
	double result = PyFloat_AsDouble(py_result);
	Py_DECREF(py_result);
	return result;
}
{% endif %}

{% for function,nargs in callbacks %}
static PyObject * callback_{{function}};
# define {{function}}(...) callback(\
		callback_{{function}}, \
		Py_BuildValue( \
				{% if nargs -%}
				"(O{{'d'*nargs}})", n_dim_read_only_array_from_data(Y) , __VA_ARGS__ \
				{% else -%}
				"(O)", n_dim_read_only_array_from_data(Y) \
				{% endif -%}
			))
{% endfor %}

# define set_drift(i, value) (drift[i] = (value)*h)
# define set_diffusion(i, value) (diffusion[i] = value)
# define y(i) (Y[i])

# define get_f_helper(i) ((f_helper[i]))
# define set_f_helper(i,value) (f_helper[i] = value)

# define get_g_helper(i) ((g_helper[i]))
# define set_g_helper(i,value) (g_helper[i] = value)

{% if has_any_helpers: %}
# include "helpers_definitions.c"
{% endif %}

# include "f_definitions.c"
void eval_drift(
	sde_integrator * const self,
	double const t,
	double Y[{{n}}],
	double const h,
	double drift[{{n}}])
{
	{% if number_of_f_helpers>0: %}
	double f_helper[{{number_of_f_helpers}}];
	# include "f_helpers.c"
	{% endif %}
	# include "f.c"
}

# include "g_definitions.c"
void eval_diffusion(
	sde_integrator * const self,
	double const t,
	{% if not additive %}
	double Y[{{n}}],
	{% endif %}
	double diffusion[{{n}}])
{
	{% if number_of_g_helpers>0: %}
	double g_helper[{{number_of_g_helpers}}];
	# include "g_helpers.c"
	{% endif %}
	# include "g.c"
}

static PyObject * get_next_step(sde_integrator * const self, PyObject * args)
{
	double h;
	if (!PyArg_ParseTuple(args, "d", &h))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	double I_1[{{n}}];
	double I_10[{{n}}];
	
	{% if not additive %}

	double I_11dbsqh[{{n}}];
	double I_111[{{n}}];
	
	get_I(self,h,I_1,I_11dbsqh,I_111,I_10);
	
	double argument[{{n}}];
	
	double fh_1[{{n}}];
	eval_drift(self,self->t,self->state,h,fh_1);
	
	double g_1[{{n}}];
	eval_diffusion(self,self->t,self->state,g_1);
	
	double fh_2[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->state[i] + 0.75*fh_1[i] + 1.5*g_1[i]*I_10[i]/h;
	eval_drift(self,self->t+0.75*h,argument,h,fh_2);
	
	double g_2[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->state[i] + 0.25*fh_1[i] + 0.5*g_1[i]*sqrt(h);
	eval_diffusion(self,self->t+0.25*h,argument,g_2);
	
	double g_3[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->state[i] + fh_1[i] - g_1[i]*sqrt(h);
	eval_diffusion(self,self->t+h,argument,g_3);
	
	double g_4[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->state[i] + 0.25*fh_1[i] + sqrt(h)*(-5*g_1[i]+3*g_2[i]+0.5*g_3[i]);
	eval_diffusion(self,self->t+0.25*h,argument,g_4);
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
	{
		double E_N = (1./h/3.) * (
			+ (  6*I_10[i] - 6*I_111[i] ) * g_1[i]
			+ ( -4*I_10[i] + 5*I_111[i] ) * g_2[i]
			+ ( -2*I_10[i] - 2*I_111[i] ) * g_3[i]
			+ (              3*I_111[i] ) * g_4[i]
			);
		self->new_state[i] = self->state[i] + E_N + (1./3.)*(
			fh_1[i] + 2*fh_2[i]
			+ (-3*I_1[i] - 3*I_11dbsqh[i] ) * g_1[i]
			+ ( 4*I_1[i] + 4*I_11dbsqh[i] ) * g_2[i]
			+ ( 2*I_1[i] -   I_11dbsqh[i] ) * g_3[i]
			);
		double E_D = (fh_2[i]-fh_1[i])/6;
		self->error[i] = fabs(E_D) + fabs(E_N);
	}
	
	{% else %}

	get_I(self,h,I_1,I_10);
	
	double argument[{{n}}];
	
	double fh_1[{{n}}];
	eval_drift(self,self->t,self->state,h,fh_1);
	
	double g_1[{{n}}];
	eval_diffusion(self,self->t+h,g_1);
	
	double fh_2[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->state[i] + 0.75*fh_1[i] + 0.5*g_1[i]*I_10[i]/h;
	eval_drift(self,self->t+0.75*h,argument,h,fh_2);
	
	double g_2[{{n}}];
	eval_diffusion(self,self->t,g_2);
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
	{
		double E_N = I_10[i] /h * (g_2[i]-g_1[i]);
		self->new_state[i] = self->state[i] + E_N + (1./3.)*(fh_1[i] + 2*fh_2[i]) + I_1[i]*g_1[i];
		double E_D = (fh_2[i]-fh_1[i])/6;
		self->error[i] = fabs(E_D) + fabs(E_N);
	}
	
	{% endif %}
	
	self->new_t = self->t + h;
	Py_RETURN_NONE;
}

static PyObject * get_p(sde_integrator const * const self, PyObject * args)
{
	double atol;
	double rtol;
	if (!PyArg_ParseTuple(args, "dd", &atol, &rtol))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	double p=0.0;
	for (int i=0; i<{{n}}; i++)
	{
		double error = self->error[i];
		double tolerance = atol+rtol*fabs(self->new_state[i]);
		if (error!=0.0 || tolerance!=0.0)
		{
			double x = error/tolerance;
			if (x>p)
				p = x;
		}
	}
	
	return PyFloat_FromDouble(p);
}

static PyObject * accept_step(sde_integrator * const self)
{
	memcpy(self->state, self->new_state, sizeof(self->state));
	self->t = self->new_t;
	while ((self->noises) && (self->noises != self->current_noise))
		remove_first_noise(self);
	self->current_noise = self->noises;
	Py_RETURN_NONE;
}

static PyObject * get_state(sde_integrator * const self)
{
	PyArrayObject * array = (PyArrayObject*) PyArray_SimpleNewFromData(1, dim, TYPE_INDEX, self->state);
	// Copy is necessary because self->state may be overwritten after this.
	PyObject * result = PyArray_NewCopy(array,NPY_ANYORDER);
	Py_DECREF(array);
	return result;
}

static PyObject * apply_jump(sde_integrator * const self, PyObject * args)
{
	PyArrayObject * change;
	
	if (!PyArg_ParseTuple(args,"O!",&PyArray_Type,&change))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input. Note that the function returning jump amplitudes must return a NumPy array.");
		return NULL;
	}
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		self->state[i] += * (double *) PyArray_GETPTR1(change,i);
	
	Py_RETURN_NONE;
}

static void sde_integrator_dealloc(sde_integrator * const self)
{
	while (self->noises)
		remove_first_noise(self);
	{% if numpy_rng %}
	Py_DECREF(self->RNG);
	Py_DECREF(self->noise_function);
	Py_DECREF(self->noise_size);
	{% else %}
	free(self->RNG);
	{% endif %}
	Py_TYPE(self)->tp_free((PyObject *)self);
}

static int sde_integrator_init(sde_integrator * self, PyObject * args)
{
	PyArrayObject * Y;
	PyObject * seed;
	
	if (!PyArg_ParseTuple(
			args,
			"dO!O{{'O'*callbacks|length}}",
			&(self->t),
			&PyArray_Type, &Y,
			&seed
			{% for function,nargs in callbacks %}
			, &callback_{{function}}
			{% endfor %}
		))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return 1;
	}
	
	{% for function,nargs in callbacks %}
	if (!PyCallable_Check(callback_{{function}}))
	{
		PyErr_SetString(PyExc_TypeError,"Callback must be callable.");
		return -1;
	}
	{% endfor %}
	
	self->noises = NULL;
	self->current_noise = NULL;
	self->last_noise = NULL;
	for (int i=0; i<{{n}}; i++)
		self->state[i] = * (double *) PyArray_GETPTR1(Y,i);
	
	{% if numpy_rng %}
	PyObject * nprandom = PyImport_ImportModule("numpy.random");
	self->RNG = PyObject_CallFunctionObjArgs(
			PyObject_GetAttrString(nprandom,"RandomState"),
			seed, NULL
			);
	self->noise_function = PyObject_GetAttrString(self->RNG,"normal");
	self->noise_size = PyTuple_Pack(2,PyLong_FromLong({{n}}),PyLong_FromLong(2));
	{% else %}
	self->RNG = safe_malloc(sizeof(rk_state));
	rk_seed( PyLong_AsUnsignedLong(seed), self->RNG );
	{% endif %}
	
	return 0;
}

// ======================================================

static PyMemberDef sde_integrator_members[] = {
 	{"t", T_DOUBLE, offsetof(sde_integrator,t), 0, "t"},
	{NULL}  /* Sentinel */
};

static PyMethodDef sde_integrator_methods[] = {
	{% if control_pars|length %}
	{"set_parameters", (PyCFunction) set_parameters, METH_VARARGS, NULL},
	{% endif %}
	{"pin_noise"     , (PyCFunction) pin_noise     , METH_VARARGS, NULL},
	{"get_next_step" , (PyCFunction) get_next_step , METH_VARARGS, NULL},
	{"get_p"         , (PyCFunction) get_p         , METH_VARARGS, NULL},
	{"accept_step"   , (PyCFunction) accept_step   , METH_NOARGS , NULL},
	{"apply_jump"    , (PyCFunction) apply_jump    , METH_VARARGS, NULL},
	{"get_state"     , (PyCFunction) get_state     , METH_NOARGS , NULL},
	{ NULL           ,               NULL          , 0           , NULL}
};


static PyTypeObject sde_integrator_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"_jitced.sde_integrator",
	sizeof(sde_integrator),
	0,                         // tp_itemsize
	(destructor) sde_integrator_dealloc,
	0,                         // tp_print
	0,0,0,0,0,0,0,0,0,0,0,0,   // ...
	0,                         // tp_as_buffer
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	0,                         // tp_doc
	0,0,0,0,0,                 // ...
	0,                         // tp_iternext
	sde_integrator_methods,
	sde_integrator_members,
	0,                         // tp_getset
	0,0,0,0,                   // ...
	0,                         // tp_dictoffset
	(initproc) sde_integrator_init,
	0,                         // tp_alloc
	0                          // tp_new
};

static PyMethodDef {{module_name}}_methods[] = {
{NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef =
{
		PyModuleDef_HEAD_INIT,
		"{{module_name}}",
		NULL,
		-1,
		{{module_name}}_methods,
		NULL,
		NULL,
		NULL,
		NULL
};

PyMODINIT_FUNC PyInit_{{module_name}}(void)
{
	sde_integrator_type.tp_new = PyType_GenericNew;
	if (PyType_Ready(&sde_integrator_type) < 0)
		return NULL;
	
	PyObject * module = PyModule_Create(&moduledef);
	
	if (module == NULL)
		return NULL;
	
	Py_INCREF(&sde_integrator_type);
	PyModule_AddObject(module, "sde_integrator", (PyObject *)&sde_integrator_type);
	
	import_array();
	
	return module;
}

