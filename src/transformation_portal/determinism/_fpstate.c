#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#if defined(__aarch64__)
static unsigned long read_fpcr(void) {
    unsigned long fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
}
#endif

static int set_item_owned(PyObject* d, const char* key, PyObject* value) {
    if (value == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, key, value) < 0) {
        Py_DECREF(value);
        return -1;
    }
    Py_DECREF(value);
    return 0;
}

static PyObject* get_fp_state(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;

    PyObject* d = PyDict_New();
    if (d == NULL) {
        return NULL;
    }

#if defined(__x86_64__) || defined(__i386__)
    unsigned int mxcsr = _mm_getcsr();
    int ftz = (mxcsr >> 15) & 1;
    int daz = (mxcsr >> 6) & 1;

    if (set_item_owned(d, "arch", PyUnicode_FromString("x86")) < 0) {
        goto error;
    }
    if (set_item_owned(d, "register", PyUnicode_FromString("MXCSR")) < 0) {
        goto error;
    }
    if (set_item_owned(d, "mxcsr", PyLong_FromUnsignedLong(mxcsr)) < 0) {
        goto error;
    }
    if (set_item_owned(d, "ftz", PyBool_FromLong(ftz)) < 0) {
        goto error;
    }
    if (set_item_owned(d, "daz", PyBool_FromLong(daz)) < 0) {
        goto error;
    }
    return d;

#elif defined(__aarch64__)
    unsigned long fpcr = read_fpcr();
    int ftz = (fpcr >> 24) & 1;

    if (set_item_owned(d, "arch", PyUnicode_FromString("aarch64")) < 0) {
        goto error;
    }
    if (set_item_owned(d, "register", PyUnicode_FromString("FPCR")) < 0) {
        goto error;
    }
    if (set_item_owned(d, "fpcr", PyLong_FromUnsignedLong(fpcr)) < 0) {
        goto error;
    }
    if (set_item_owned(d, "ftz", PyBool_FromLong(ftz)) < 0) {
        goto error;
    }
    if (set_item_owned(d, "daz", PyBool_FromLong(0)) < 0) {
        goto error;
    }
    return d;

#else
    PyErr_SetString(PyExc_RuntimeError, "Unsupported architecture");
    goto error;
#endif

error:
    Py_DECREF(d);
    return NULL;
}

static PyMethodDef Methods[] = {
    {"get_fp_state", get_fp_state, METH_NOARGS, "Return FP register state"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_fpstate",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__fpstate(void) {
    return PyModule_Create(&module);
}
