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

static PyObject* get_fp_state(PyObject* self, PyObject* args) {
    PyObject* d = PyDict_New();

#if defined(__x86_64__) || defined(__i386__)
    unsigned int mxcsr = _mm_getcsr();
    int ftz = (mxcsr >> 15) & 1;
    int daz = (mxcsr >> 6) & 1;

    PyDict_SetItemString(d, "arch", PyUnicode_FromString("x86"));
    PyDict_SetItemString(d, "register", PyUnicode_FromString("MXCSR"));
    PyDict_SetItemString(d, "mxcsr", PyLong_FromUnsignedLong(mxcsr));
    PyDict_SetItemString(d, "ftz", PyBool_FromLong(ftz));
    PyDict_SetItemString(d, "daz", PyBool_FromLong(daz));
    return d;

#elif defined(__aarch64__)
    unsigned long fpcr = read_fpcr();
    int ftz = (fpcr >> 24) & 1;

    PyDict_SetItemString(d, "arch", PyUnicode_FromString("aarch64"));
    PyDict_SetItemString(d, "register", PyUnicode_FromString("FPCR"));
    PyDict_SetItemString(d, "fpcr", PyLong_FromUnsignedLong(fpcr));
    PyDict_SetItemString(d, "ftz", PyBool_FromLong(ftz));
    PyDict_SetItemString(d, "daz", PyBool_FromLong(0));
    return d;

#else
    Py_DECREF(d);
    PyErr_SetString(PyExc_RuntimeError, "Unsupported architecture");
    return NULL;
#endif
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
