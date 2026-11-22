/*
 * eproc_pyext.c — Zero-dependency CPython C-API accelerator for eproc.
 *
 * Links dynamically against libeproc (C11 ABI v1) and eliminates ctypes/libffi
 * and Python dataclass construction overhead on scalar and batch operations.
 *
 * License: Apache-2.0
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include "eproc.h"

#define EPROC_PYCFUNC(fn) ((PyCFunction)(void (*)(void))(fn))

#define EPROC_FL_CAP 32
#ifndef Py_GIL_DISABLED
#define EPROC_DEFINE_FREELIST(NAME, OBJTYPE, PYTYPE) \
    static OBJTYPE *g_fl_##NAME[EPROC_FL_CAP]; \
    static int g_fl_n_##NAME = 0; \
    static inline OBJTYPE *alloc_##NAME(void) { \
        if (g_fl_n_##NAME > 0) { \
            OBJTYPE *o = g_fl_##NAME[--g_fl_n_##NAME]; \
            return (OBJTYPE *)PyObject_Init((PyObject *)o, &(PYTYPE)); \
        } \
        return PyObject_New(OBJTYPE, &(PYTYPE)); \
    } \
    static inline void free_##NAME(OBJTYPE *o) { \
        if (g_fl_n_##NAME < EPROC_FL_CAP) { \
            g_fl_##NAME[g_fl_n_##NAME++] = o; \
            return; \
        } \
        PyObject_Del(o); \
    }
#else
#define EPROC_DEFINE_FREELIST(NAME, OBJTYPE, PYTYPE) \
    static inline OBJTYPE *alloc_##NAME(void) { return PyObject_New(OBJTYPE, &(PYTYPE)); } \
    static inline void free_##NAME(OBJTYPE *o) { PyObject_Del(o); }
#endif

/* ============================================================================
 * Helpers: Error Mapping, Fast Float/Int Parsing, Buffer Extraction, Hashing
 * ============================================================================ */

static int set_eproc_error(int rc, const char *ctx) {
    if (rc == EPROC_OK) {
        return 0;
    }
    switch (rc) {
        case EPROC_ERR_INVALID:
            PyErr_Format(PyExc_ValueError, "%s: invalid argument or parameter out of range", ctx);
            return -1;
        case EPROC_ERR_CAPACITY:
            PyErr_Format(PyExc_OverflowError, "%s: fixed internal capacity exceeded", ctx);
            return -1;
        case EPROC_ERR_EMPTY:
            PyErr_Format(PyExc_ValueError, "%s: operation requires non-empty state", ctx);
            return -1;
        case EPROC_ERR_MISMATCH:
            PyErr_Format(PyExc_ValueError, "%s: incompatible operand or leave-one-out mismatch", ctx);
            return -1;
        default:
            PyErr_Format(PyExc_RuntimeError, "%s: unexpected error code %d", ctx, rc);
            return -1;
    }
}

static inline int fast_as_double(PyObject *obj, double *out) {
    if (PyFloat_CheckExact(obj)) {
        *out = PyFloat_AS_DOUBLE(obj);
        return 0;
    }
    if (PyLong_CheckExact(obj)) {
        double v = PyLong_AsDouble(obj);
        if (v == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        *out = v;
        return 0;
    }
    double v = PyFloat_AsDouble(obj);
    if (v == -1.0 && PyErr_Occurred()) {
        return -1;
    }
    *out = v;
    return 0;
}

static inline int is_native_double_format(const char *fmt) {
    if (fmt == NULL) {
        return 0;
    }
    if (fmt[0] == 'd' && fmt[1] == '\0') {
        return 1;
    }
    if (fmt[0] == '=' && fmt[1] == 'd' && fmt[2] == '\0') {
        return 1;
    }
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (fmt[0] == '>' && fmt[1] == 'd' && fmt[2] == '\0') {
        return 1;
    }
#else
    if (fmt[0] == '<' && fmt[1] == 'd' && fmt[2] == '\0') {
        return 1;
    }
#endif
    return 0;
}

static inline int is_native_uint32_format(const char *fmt) {
    if (fmt == NULL) {
        return 0;
    }
    if (fmt[0] == 'I' && fmt[1] == '\0') {
        return 1;
    }
    if (fmt[0] == '=' && fmt[1] == 'I' && fmt[2] == '\0') {
        return 1;
    }
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (fmt[0] == '>' && fmt[1] == 'I' && fmt[2] == '\0') {
        return 1;
    }
#else
    if (fmt[0] == '<' && fmt[1] == 'I' && fmt[2] == '\0') {
        return 1;
    }
#endif
    return 0;
}

typedef struct {
    const double *ptr;
    Py_ssize_t len;
    Py_buffer view;
    int has_view;
    double *heap_buf;
} double_buf_t;

static void release_double_buf(double_buf_t *b) {
    if (b->has_view) {
        PyBuffer_Release(&b->view);
        b->has_view = 0;
    }
    if (b->heap_buf != NULL) {
        PyMem_Free(b->heap_buf);
        b->heap_buf = NULL;
    }
    b->ptr = NULL;
    b->len = 0;
}

static int acquire_double_buf(PyObject *obj, double_buf_t *b) {
    memset(b, 0, sizeof(*b));
    if (PyUnicode_Check(obj) || PyBytes_Check(obj) || PyByteArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numeric sequence or buffer, not str/bytes");
        return -1;
    }

    /* Fast path 1: PEP 3118 zero-copy buffer (array('d'), memoryview, ctypes, contiguous numpy) */
    if (PyObject_CheckBuffer(obj)) {
        if (PyObject_GetBuffer(obj, &b->view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) == 0) {
            if (b->view.ndim >= 1 &&
                b->view.itemsize == (Py_ssize_t)sizeof(double) &&
                is_native_double_format(b->view.format) &&
                PyBuffer_IsContiguous(&b->view, 'C')) {
                b->ptr = (const double *)b->view.buf;
                b->len = b->view.len / (Py_ssize_t)sizeof(double);
                b->has_view = 1;
                return 0;
            }
            PyBuffer_Release(&b->view);
        } else {
            PyErr_Clear();
        }
    }

    /* Fast path 2: Python list/tuple or general sequence/iterable */
    PyObject *seq = PySequence_Fast(obj, "Expected a numeric sequence or buffer");
    if (seq == NULL) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0) {
        Py_DECREF(seq);
        b->ptr = NULL;
        b->len = 0;
        return 0;
    }
    double *buf = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (buf == NULL) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (fast_as_double(items[i], &buf[i]) < 0) {
            PyMem_Free(buf);
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    b->heap_buf = buf;
    b->ptr = buf;
    b->len = n;
    return 0;
}

static int extract_fixed_doubles(PyObject *obj, double *out, Py_ssize_t expected, const char *ctx) {
    /* Ultra-fast list exact path */
    if (PyList_CheckExact(obj)) {
        Py_ssize_t n = PyList_GET_SIZE(obj);
        if (n != expected) {
            PyErr_Format(PyExc_ValueError, "%s: expected %zd values, got %zd", ctx, expected, n);
            return -1;
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (fast_as_double(PyList_GET_ITEM(obj, i), &out[i]) < 0) {
                return -1;
            }
        }
        return 0;
    }
    /* Ultra-fast tuple exact path */
    if (PyTuple_CheckExact(obj)) {
        Py_ssize_t n = PyTuple_GET_SIZE(obj);
        if (n != expected) {
            PyErr_Format(PyExc_ValueError, "%s: expected %zd values, got %zd", ctx, expected, n);
            return -1;
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (fast_as_double(PyTuple_GET_ITEM(obj, i), &out[i]) < 0) {
                return -1;
            }
        }
        return 0;
    }
    if (PyUnicode_Check(obj) || PyBytes_Check(obj) || PyByteArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numeric sequence or buffer, not str/bytes");
        return -1;
    }
    /* PEP 3118 buffer path */
    if (PyObject_CheckBuffer(obj)) {
        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) == 0) {
            if (view.ndim >= 1 &&
                view.itemsize == (Py_ssize_t)sizeof(double) &&
                is_native_double_format(view.format) &&
                PyBuffer_IsContiguous(&view, 'C')) {
                Py_ssize_t n = view.len / (Py_ssize_t)sizeof(double);
                if (n != expected) {
                    PyBuffer_Release(&view);
                    PyErr_Format(PyExc_ValueError, "%s: expected %zd values, got %zd", ctx, expected, n);
                    return -1;
                }
                memcpy(out, view.buf, (size_t)expected * sizeof(double));
                PyBuffer_Release(&view);
                return 0;
            }
            PyBuffer_Release(&view);
        } else {
            PyErr_Clear();
        }
    }
    /* General sequence / iterable fallback */
    PyObject *seq = PySequence_Fast(obj, "Expected a numeric sequence or buffer");
    if (seq == NULL) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n != expected) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s: expected %zd values, got %zd", ctx, expected, n);
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (fast_as_double(items[i], &out[i]) < 0) {
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}

static int parse_side(PyObject *obj, int32_t *out_code, const char **out_name) {
    if (PyLong_Check(obj) && !PyBool_Check(obj)) {
        long v = PyLong_AsLong(obj);
        if (v == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (v == EPROC_SIDE_UPPER) {
            *out_code = EPROC_SIDE_UPPER;
            *out_name = "upper";
            return 0;
        }
        if (v == EPROC_SIDE_LOWER) {
            *out_code = EPROC_SIDE_LOWER;
            *out_name = "lower";
            return 0;
        }
        if (v == EPROC_SIDE_TWO_SIDED) {
            *out_code = EPROC_SIDE_TWO_SIDED;
            *out_name = "two_sided";
            return 0;
        }
        PyErr_Format(PyExc_ValueError, "Invalid side integer %ld; expected 0, 1, or 2", v);
        return -1;
    }
    if (PyUnicode_Check(obj)) {
        PyObject *lowered = PyObject_CallMethod(obj, "strip", NULL);
        if (lowered == NULL) {
            return -1;
        }
        PyObject *norm = PyObject_CallMethod(lowered, "lower", NULL);
        Py_DECREF(lowered);
        if (norm == NULL) {
            return -1;
        }
        const char *s = PyUnicode_AsUTF8(norm);
        if (s == NULL) {
            Py_DECREF(norm);
            return -1;
        }
        int ok = 1;
        if (strcmp(s, "upper") == 0 || strcmp(s, "greater") == 0) {
            *out_code = EPROC_SIDE_UPPER;
            *out_name = "upper";
        } else if (strcmp(s, "lower") == 0 || strcmp(s, "less") == 0) {
            *out_code = EPROC_SIDE_LOWER;
            *out_name = "lower";
        } else if (strcmp(s, "two_sided") == 0 || strcmp(s, "two-sided") == 0 || strcmp(s, "both") == 0) {
            *out_code = EPROC_SIDE_TWO_SIDED;
            *out_name = "two_sided";
        } else {
            ok = 0;
        }
        Py_DECREF(norm);
        if (ok) {
            return 0;
        }
    }
    PyErr_SetString(PyExc_ValueError, "Invalid side; expected 'upper', 'lower', or 'two_sided'");
    return -1;
}

static int prepare_alphas(PyObject *alphas_obj, double *out, uint32_t *out_n) {
    PyObject *seq = PySequence_Fast(alphas_obj, "alphas must be a sequence of floats");
    if (seq == NULL) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1 || n > EPROC_MAX_RATES) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "Number of alpha levels must be in [1, %d], got %zd", EPROC_MAX_RATES, n);
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double a;
        if (fast_as_double(items[i], &a) < 0) {
            Py_DECREF(seq);
            return -1;
        }
        if (!isfinite(a) || a <= 0.0 || a >= 1.0) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "Each alpha must be finite in (0, 1)");
            return -1;
        }
        out[i] = a;
    }
    Py_DECREF(seq);
    /* Insertion sort descending (n <= 5) */
    for (Py_ssize_t i = 1; i < n; ++i) {
        double key = out[i];
        Py_ssize_t j = i - 1;
        while (j >= 0 && out[j] < key) {
            out[j + 1] = out[j];
            --j;
        }
        out[j + 1] = key;
    }
    *out_n = (uint32_t)n;
    return 0;
}

static PyObject *make_double_tuple(const double *vals, uint32_t n) {
    PyObject *tup = PyTuple_New((Py_ssize_t)n);
    if (tup == NULL) {
        return NULL;
    }
    for (uint32_t i = 0; i < n; ++i) {
        PyObject *f = PyFloat_FromDouble(vals[i]);
        if (f == NULL) {
            Py_DECREF(tup);
            return NULL;
        }
        PyTuple_SET_ITEM(tup, (Py_ssize_t)i, f);
    }
    return tup;
}

/* Lazily cached hashlib.blake2b for string/bytes evidence_id (matches fusion.py bit-for-bit) */
static PyObject *g_blake2b_fn = NULL;

static int normalize_evidence_id(PyObject *obj, uint64_t *out) {
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "evidence_id must be an int, str, or bytes, not bool");
        return -1;
    }
    if (PyLong_Check(obj)) {
        unsigned long long v = PyLong_AsUnsignedLongLong(obj);
        if (v == (unsigned long long)-1 && PyErr_Occurred()) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "evidence_id integer out of uint64 range");
            return -1;
        }
        *out = (uint64_t)v;
        return 0;
    }
    PyObject *raw_bytes = NULL;
    if (PyUnicode_Check(obj)) {
        raw_bytes = PyUnicode_AsUTF8String(obj);
        if (raw_bytes == NULL) {
            return -1;
        }
    } else if (PyBytes_Check(obj)) {
        Py_INCREF(obj);
        raw_bytes = obj;
    } else if (PyByteArray_Check(obj)) {
        raw_bytes = PyBytes_FromStringAndSize(PyByteArray_AS_STRING(obj), PyByteArray_GET_SIZE(obj));
        if (raw_bytes == NULL) {
            return -1;
        }
    } else {
        PyErr_Format(PyExc_TypeError, "evidence_id must be int, str, or bytes, got %s", Py_TYPE(obj)->tp_name);
        return -1;
    }

    if (g_blake2b_fn == NULL) {
        PyObject *hashlib_mod = PyImport_ImportModule("hashlib");
        if (hashlib_mod == NULL) {
            Py_DECREF(raw_bytes);
            return -1;
        }
        g_blake2b_fn = PyObject_GetAttrString(hashlib_mod, "blake2b");
        Py_DECREF(hashlib_mod);
        if (g_blake2b_fn == NULL) {
            Py_DECREF(raw_bytes);
            return -1;
        }
    }

    PyObject *kw = Py_BuildValue("{s:i}", "digest_size", 8);
    if (kw == NULL) {
        Py_DECREF(raw_bytes);
        return -1;
    }
    PyObject *args = PyTuple_Pack(1, raw_bytes);
    Py_DECREF(raw_bytes);
    if (args == NULL) {
        Py_DECREF(kw);
        return -1;
    }
    PyObject *hasher = PyObject_Call(g_blake2b_fn, args, kw);
    Py_DECREF(args);
    Py_DECREF(kw);
    if (hasher == NULL) {
        return -1;
    }
    PyObject *digest = PyObject_CallMethod(hasher, "digest", NULL);
    Py_DECREF(hasher);
    if (digest == NULL) {
        return -1;
    }
    const unsigned char *b = (const unsigned char *)PyBytes_AsString(digest);
    if (b == NULL || PyBytes_GET_SIZE(digest) != 8) {
        Py_DECREF(digest);
        PyErr_SetString(PyExc_RuntimeError, "blake2b digest size mismatch");
        return -1;
    }
    uint64_t u = ((uint64_t)b[0]) |
                 ((uint64_t)b[1] << 8) |
                 ((uint64_t)b[2] << 16) |
                 ((uint64_t)b[3] << 24) |
                 ((uint64_t)b[4] << 32) |
                 ((uint64_t)b[5] << 40) |
                 ((uint64_t)b[6] << 48) |
                 ((uint64_t)b[7] << 56);
    Py_DECREF(digest);
    *out = u;
    return 0;
}

static PyObject *g_str_site = NULL;
static PyObject *g_str_channel = NULL;
static PyObject *g_str_ticks = NULL;

/* Helper: dict insertion stealing value reference */
static inline int dict_set_steal(PyObject *d, const char *key, PyObject *val) {
    if (val == NULL) {
        return -1;
    }
    int rc = PyDict_SetItemString(d, key, val);
    Py_DECREF(val);
    return rc;
}

/* ============================================================================
 * 1. LIGHTWEIGHT C RESULT TYPES
 * ============================================================================ */

/* --- EProcessResult --- */

typedef struct {
    PyObject_HEAD
    unsigned long long n;
    double e_value;
    double log_e;
    double max_log_e;
    double p_value;
    double mean;
    char rejected;
} EProcessResultObj;

static PyMemberDef EProcessResult_members[] = {
    {"n", T_ULONGLONG, offsetof(EProcessResultObj, n), READONLY, NULL},
    {"e_value", T_DOUBLE, offsetof(EProcessResultObj, e_value), READONLY, NULL},
    {"log_e", T_DOUBLE, offsetof(EProcessResultObj, log_e), READONLY, NULL},
    {"max_log_e", T_DOUBLE, offsetof(EProcessResultObj, max_log_e), READONLY, NULL},
    {"p_value", T_DOUBLE, offsetof(EProcessResultObj, p_value), READONLY, NULL},
    {"mean", T_DOUBLE, offsetof(EProcessResultObj, mean), READONLY, NULL},
    {"rejected", T_BOOL, offsetof(EProcessResultObj, rejected), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *EProcessResult_to_dict(EProcessResultObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "n", PyLong_FromUnsignedLongLong(self->n)) < 0 ||
        dict_set_steal(d, "e_value", PyFloat_FromDouble(self->e_value)) < 0 ||
        dict_set_steal(d, "log_e", PyFloat_FromDouble(self->log_e)) < 0 ||
        dict_set_steal(d, "max_log_e", PyFloat_FromDouble(self->max_log_e)) < 0 ||
        dict_set_steal(d, "p_value", PyFloat_FromDouble(self->p_value)) < 0 ||
        dict_set_steal(d, "mean", PyFloat_FromDouble(self->mean)) < 0 ||
        dict_set_steal(d, "rejected", PyBool_FromLong(self->rejected)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyTypeObject EProcessResultType;
EPROC_DEFINE_FREELIST(epres, EProcessResultObj, EProcessResultType)

static void EProcessResult_dealloc(EProcessResultObj *self) {
    free_epres(self);
}

static PyObject *EProcessResult_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(b, &EProcessResultType) || (op != Py_EQ && op != Py_NE)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    EProcessResultObj *x = (EProcessResultObj *)a;
    EProcessResultObj *y = (EProcessResultObj *)b;
    int eq = (x->n == y->n && x->e_value == y->e_value && x->log_e == y->log_e &&
              x->max_log_e == y->max_log_e && x->p_value == y->p_value &&
              x->mean == y->mean && x->rejected == y->rejected);
    return PyBool_FromLong(op == Py_EQ ? eq : !eq);
}

static PyMethodDef EProcessResult_methods[] = {
    {"to_dict", (PyCFunction)EProcessResult_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject EProcessResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.EProcessResult",
    .tp_basicsize = sizeof(EProcessResultObj),
    .tp_dealloc = (destructor)EProcessResult_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = EProcessResult_members,
    .tp_methods = EProcessResult_methods,
    .tp_richcompare = EProcessResult_richcompare,
};

static inline PyObject *new_eprocess_result(const eproc_bet_result_t *r) {
    EProcessResultObj *o = alloc_epres();
    if (o == NULL) return NULL;
    o->n = (unsigned long long)r->n;
    o->e_value = r->e_value;
    o->log_e = r->log_e;
    o->max_log_e = r->max_log_e;
    o->p_value = r->p_value;
    o->mean = r->mean;
    o->rejected = r->rejected ? 1 : 0;
    return (PyObject *)o;
}

/* --- ConfidenceSequenceResult --- */

typedef struct {
    PyObject_HEAD
    unsigned long long n;
    double mean;
    double low;
    double high;
    double width;
} CSResultObj;

static PyMemberDef CSResult_members[] = {
    {"n", T_ULONGLONG, offsetof(CSResultObj, n), READONLY, NULL},
    {"mean", T_DOUBLE, offsetof(CSResultObj, mean), READONLY, NULL},
    {"low", T_DOUBLE, offsetof(CSResultObj, low), READONLY, NULL},
    {"high", T_DOUBLE, offsetof(CSResultObj, high), READONLY, NULL},
    {"width", T_DOUBLE, offsetof(CSResultObj, width), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *CSResult_to_dict(CSResultObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "n", PyLong_FromUnsignedLongLong(self->n)) < 0 ||
        dict_set_steal(d, "mean", PyFloat_FromDouble(self->mean)) < 0 ||
        dict_set_steal(d, "low", PyFloat_FromDouble(self->low)) < 0 ||
        dict_set_steal(d, "high", PyFloat_FromDouble(self->high)) < 0 ||
        dict_set_steal(d, "width", PyFloat_FromDouble(self->width)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyTypeObject CSResultType;
EPROC_DEFINE_FREELIST(csres, CSResultObj, CSResultType)

static void CSResult_dealloc(CSResultObj *self) {
    free_csres(self);
}

static PyObject *CSResult_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(b, &CSResultType) || (op != Py_EQ && op != Py_NE)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    CSResultObj *x = (CSResultObj *)a;
    CSResultObj *y = (CSResultObj *)b;
    int eq = (x->n == y->n && x->mean == y->mean && x->low == y->low &&
              x->high == y->high && x->width == y->width);
    return PyBool_FromLong(op == Py_EQ ? eq : !eq);
}

static PyMethodDef CSResult_methods[] = {
    {"to_dict", (PyCFunction)CSResult_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject CSResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.ConfidenceSequenceResult",
    .tp_basicsize = sizeof(CSResultObj),
    .tp_dealloc = (destructor)CSResult_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = CSResult_members,
    .tp_methods = CSResult_methods,
    .tp_richcompare = CSResult_richcompare,
};

static inline PyObject *new_cs_result(const eproc_cs_result_t *r) {
    CSResultObj *o = alloc_csres();
    if (o == NULL) return NULL;
    o->n = (unsigned long long)r->n;
    o->mean = r->mean;
    o->low = r->low;
    o->high = r->high;
    o->width = r->width;
    return (PyObject *)o;
}

/* --- EDetectorResult --- */

typedef struct {
    PyObject_HEAD
    unsigned long long n;
    double log_stat;
    double stat;
    char alarm;
} EDetectorResultObj;

static PyMemberDef EDetectorResult_members[] = {
    {"n", T_ULONGLONG, offsetof(EDetectorResultObj, n), READONLY, NULL},
    {"log_stat", T_DOUBLE, offsetof(EDetectorResultObj, log_stat), READONLY, NULL},
    {"stat", T_DOUBLE, offsetof(EDetectorResultObj, stat), READONLY, NULL},
    {"alarm", T_BOOL, offsetof(EDetectorResultObj, alarm), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *EDetectorResult_to_dict(EDetectorResultObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "n", PyLong_FromUnsignedLongLong(self->n)) < 0 ||
        dict_set_steal(d, "log_stat", PyFloat_FromDouble(self->log_stat)) < 0 ||
        dict_set_steal(d, "stat", PyFloat_FromDouble(self->stat)) < 0 ||
        dict_set_steal(d, "alarm", PyBool_FromLong(self->alarm)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyTypeObject EDetectorResultType;

static PyObject *EDetectorResult_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(b, &EDetectorResultType) || (op != Py_EQ && op != Py_NE)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    EDetectorResultObj *x = (EDetectorResultObj *)a;
    EDetectorResultObj *y = (EDetectorResultObj *)b;
    int eq = (x->n == y->n && x->log_stat == y->log_stat &&
              x->stat == y->stat && x->alarm == y->alarm);
    return PyBool_FromLong(op == Py_EQ ? eq : !eq);
}

static PyMethodDef EDetectorResult_methods[] = {
    {"to_dict", (PyCFunction)EDetectorResult_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject EDetectorResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.EDetectorResult",
    .tp_basicsize = sizeof(EDetectorResultObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = EDetectorResult_members,
    .tp_methods = EDetectorResult_methods,
    .tp_richcompare = EDetectorResult_richcompare,
};

static inline PyObject *new_edetect_result(uint64_t n, double log_stat, double stat, int alarm) {
    EDetectorResultObj *o = PyObject_New(EDetectorResultObj, &EDetectorResultType);
    if (o == NULL) return NULL;
    o->n = (unsigned long long)n;
    o->log_stat = log_stat;
    o->stat = stat;
    o->alarm = alarm ? 1 : 0;
    return (PyObject *)o;
}

/* --- RaceStatus --- */

typedef struct {
    PyObject_HEAD
    unsigned int n_arms;
    unsigned int n_active;
    int32_t winner_raw;
    int32_t leader_raw;
    unsigned long long total_obs;
    double leader_low;
    double leader_high;
    double runner_up_high;
    PyObject *winner_name;
    PyObject *leader_name;
} RaceStatusObj;

static void RaceStatus_dealloc(RaceStatusObj *self) {
    Py_XDECREF(self->winner_name);
    Py_XDECREF(self->leader_name);
    PyObject_Del(self);
}

static PyObject *RaceStatus_get_winner(RaceStatusObj *self, void *closure) {
    if (self->winner_raw < 0) Py_RETURN_NONE;
    return PyLong_FromLong(self->winner_raw);
}

static PyObject *RaceStatus_get_leader(RaceStatusObj *self, void *closure) {
    if (self->leader_raw < 0) Py_RETURN_NONE;
    return PyLong_FromLong(self->leader_raw);
}

static PyMemberDef RaceStatus_members[] = {
    {"n_arms", T_UINT, offsetof(RaceStatusObj, n_arms), READONLY, NULL},
    {"n_active", T_UINT, offsetof(RaceStatusObj, n_active), READONLY, NULL},
    {"total_obs", T_ULONGLONG, offsetof(RaceStatusObj, total_obs), READONLY, NULL},
    {"leader_low", T_DOUBLE, offsetof(RaceStatusObj, leader_low), READONLY, NULL},
    {"leader_high", T_DOUBLE, offsetof(RaceStatusObj, leader_high), READONLY, NULL},
    {"runner_up_high", T_DOUBLE, offsetof(RaceStatusObj, runner_up_high), READONLY, NULL},
    {"winner_name", T_OBJECT_EX, offsetof(RaceStatusObj, winner_name), READONLY, NULL},
    {"leader_name", T_OBJECT_EX, offsetof(RaceStatusObj, leader_name), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyGetSetDef RaceStatus_getset[] = {
    {"winner", (getter)RaceStatus_get_winner, NULL, NULL, NULL},
    {"leader", (getter)RaceStatus_get_leader, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyObject *RaceStatus_to_dict(RaceStatusObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    PyObject *w = RaceStatus_get_winner(self, NULL);
    PyObject *l = RaceStatus_get_leader(self, NULL);
    Py_INCREF(self->winner_name);
    Py_INCREF(self->leader_name);
    if (dict_set_steal(d, "n_arms", PyLong_FromUnsignedLong(self->n_arms)) < 0 ||
        dict_set_steal(d, "n_active", PyLong_FromUnsignedLong(self->n_active)) < 0 ||
        dict_set_steal(d, "winner", w) < 0 ||
        dict_set_steal(d, "leader", l) < 0 ||
        dict_set_steal(d, "total_obs", PyLong_FromUnsignedLongLong(self->total_obs)) < 0 ||
        dict_set_steal(d, "leader_low", PyFloat_FromDouble(self->leader_low)) < 0 ||
        dict_set_steal(d, "leader_high", PyFloat_FromDouble(self->leader_high)) < 0 ||
        dict_set_steal(d, "runner_up_high", PyFloat_FromDouble(self->runner_up_high)) < 0 ||
        dict_set_steal(d, "winner_name", self->winner_name) < 0 ||
        dict_set_steal(d, "leader_name", self->leader_name) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef RaceStatus_methods[] = {
    {"to_dict", (PyCFunction)RaceStatus_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject RaceStatusType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.RaceStatus",
    .tp_basicsize = sizeof(RaceStatusObj),
    .tp_dealloc = (destructor)RaceStatus_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = RaceStatus_members,
    .tp_getset = RaceStatus_getset,
    .tp_methods = RaceStatus_methods,
};

/* --- RaceArmState --- */

typedef struct {
    PyObject_HEAD
    unsigned int arm;
    PyObject *name;
    char active;
    unsigned long long n;
    double mean;
    double low;
    double high;
    double width;
    unsigned long long eliminated_at;
} RaceArmStateObj;

static void RaceArmState_dealloc(RaceArmStateObj *self) {
    Py_XDECREF(self->name);
    PyObject_Del(self);
}

static PyMemberDef RaceArmState_members[] = {
    {"arm", T_UINT, offsetof(RaceArmStateObj, arm), READONLY, NULL},
    {"name", T_OBJECT_EX, offsetof(RaceArmStateObj, name), READONLY, NULL},
    {"active", T_BOOL, offsetof(RaceArmStateObj, active), READONLY, NULL},
    {"n", T_ULONGLONG, offsetof(RaceArmStateObj, n), READONLY, NULL},
    {"mean", T_DOUBLE, offsetof(RaceArmStateObj, mean), READONLY, NULL},
    {"low", T_DOUBLE, offsetof(RaceArmStateObj, low), READONLY, NULL},
    {"high", T_DOUBLE, offsetof(RaceArmStateObj, high), READONLY, NULL},
    {"width", T_DOUBLE, offsetof(RaceArmStateObj, width), READONLY, NULL},
    {"eliminated_at", T_ULONGLONG, offsetof(RaceArmStateObj, eliminated_at), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *RaceArmState_to_dict(RaceArmStateObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    Py_INCREF(self->name);
    if (dict_set_steal(d, "arm", PyLong_FromUnsignedLong(self->arm)) < 0 ||
        dict_set_steal(d, "name", self->name) < 0 ||
        dict_set_steal(d, "active", PyBool_FromLong(self->active)) < 0 ||
        dict_set_steal(d, "n", PyLong_FromUnsignedLongLong(self->n)) < 0 ||
        dict_set_steal(d, "mean", PyFloat_FromDouble(self->mean)) < 0 ||
        dict_set_steal(d, "low", PyFloat_FromDouble(self->low)) < 0 ||
        dict_set_steal(d, "high", PyFloat_FromDouble(self->high)) < 0 ||
        dict_set_steal(d, "width", PyFloat_FromDouble(self->width)) < 0 ||
        dict_set_steal(d, "eliminated_at", PyLong_FromUnsignedLongLong(self->eliminated_at)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef RaceArmState_methods[] = {
    {"to_dict", (PyCFunction)RaceArmState_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject RaceArmStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.RaceArmState",
    .tp_basicsize = sizeof(RaceArmStateObj),
    .tp_dealloc = (destructor)RaceArmState_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = RaceArmState_members,
    .tp_methods = RaceArmState_methods,
};

/* --- ConformalMeta --- */

typedef struct {
    PyObject_HEAD
    unsigned int n_rates;
    PyObject *alphas;
    PyObject *quantiles_cache;
    PyObject *eff_alphas_cache;
    PyObject *emp_coverage_cache;
    double q_raw[EPROC_MAX_RATES];
    double ea_raw[EPROC_MAX_RATES];
    double ec_raw[EPROC_MAX_RATES];
} ConformalMetaObj;

static PyTypeObject ConformalMetaType;
EPROC_DEFINE_FREELIST(cmeta, ConformalMetaObj, ConformalMetaType)

static void ConformalMeta_dealloc(ConformalMetaObj *self) {
    Py_XDECREF(self->alphas);
    Py_XDECREF(self->quantiles_cache);
    Py_XDECREF(self->eff_alphas_cache);
    Py_XDECREF(self->emp_coverage_cache);
    free_cmeta(self);
}

static PyObject *ConformalMeta_get_quantiles(ConformalMetaObj *self, void *c) {
    if (self->quantiles_cache == NULL) {
        self->quantiles_cache = make_double_tuple(self->q_raw, self->n_rates);
        if (self->quantiles_cache == NULL) return NULL;
    }
    Py_INCREF(self->quantiles_cache);
    return self->quantiles_cache;
}

static PyObject *ConformalMeta_get_eff_alphas(ConformalMetaObj *self, void *c) {
    if (self->eff_alphas_cache == NULL) {
        self->eff_alphas_cache = make_double_tuple(self->ea_raw, self->n_rates);
        if (self->eff_alphas_cache == NULL) return NULL;
    }
    Py_INCREF(self->eff_alphas_cache);
    return self->eff_alphas_cache;
}

static PyObject *ConformalMeta_get_emp_coverage(ConformalMetaObj *self, void *c) {
    if (self->emp_coverage_cache == NULL) {
        self->emp_coverage_cache = make_double_tuple(self->ec_raw, self->n_rates);
        if (self->emp_coverage_cache == NULL) return NULL;
    }
    Py_INCREF(self->emp_coverage_cache);
    return self->emp_coverage_cache;
}

static PyMemberDef ConformalMeta_members[] = {
    {"n_rates", T_UINT, offsetof(ConformalMetaObj, n_rates), READONLY, NULL},
    {"alphas", T_OBJECT_EX, offsetof(ConformalMetaObj, alphas), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyGetSetDef ConformalMeta_getset[] = {
    {"quantiles", (getter)ConformalMeta_get_quantiles, NULL, NULL, NULL},
    {"eff_alphas", (getter)ConformalMeta_get_eff_alphas, NULL, NULL, NULL},
    {"emp_coverage", (getter)ConformalMeta_get_emp_coverage, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyObject *ConformalMeta_to_dict(ConformalMetaObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    PyObject *q = ConformalMeta_get_quantiles(self, NULL);
    PyObject *ea = ConformalMeta_get_eff_alphas(self, NULL);
    PyObject *ec = ConformalMeta_get_emp_coverage(self, NULL);
    if (!q || !ea || !ec) {
        Py_XDECREF(q); Py_XDECREF(ea); Py_XDECREF(ec); Py_DECREF(d);
        return NULL;
    }
    Py_INCREF(self->alphas);
    if (dict_set_steal(d, "n_rates", PyLong_FromUnsignedLong(self->n_rates)) < 0 ||
        dict_set_steal(d, "alphas", self->alphas) < 0 ||
        dict_set_steal(d, "quantiles", q) < 0 ||
        dict_set_steal(d, "eff_alphas", ea) < 0 ||
        dict_set_steal(d, "emp_coverage", ec) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef ConformalMeta_methods[] = {
    {"to_dict", (PyCFunction)ConformalMeta_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject ConformalMetaType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.conformal.ConformalMeta",
    .tp_basicsize = sizeof(ConformalMetaObj),
    .tp_dealloc = (destructor)ConformalMeta_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = ConformalMeta_members,
    .tp_getset = ConformalMeta_getset,
    .tp_methods = ConformalMeta_methods,
};

static inline PyObject *new_conformal_meta(const eproc_conformal_out_t *raw, PyObject *alphas_tuple) {
    ConformalMetaObj *o = alloc_cmeta();
    if (o == NULL) return NULL;
    uint32_t k = raw->n_rates;
    o->n_rates = k;
    Py_INCREF(alphas_tuple);
    o->alphas = alphas_tuple;
    o->quantiles_cache = NULL;
    o->eff_alphas_cache = NULL;
    o->emp_coverage_cache = NULL;
    memcpy(o->q_raw, raw->quantiles, k * sizeof(double));
    memcpy(o->ea_raw, raw->eff_alphas, k * sizeof(double));
    memcpy(o->ec_raw, raw->emp_coverage, k * sizeof(double));
    return (PyObject *)o;
}

/* --- ConformalPrediction --- */

typedef struct {
    PyObject_HEAD
    double point_pred;
    double scale;
    PyObject *alphas;
    PyObject *lows;
    PyObject *highs;
    PyObject *quantiles;
    PyObject *eff_alphas;
    PyObject *emp_coverage;
} ConformalPredObj;

static void ConformalPred_dealloc(ConformalPredObj *self) {
    Py_XDECREF(self->alphas);
    Py_XDECREF(self->lows);
    Py_XDECREF(self->highs);
    Py_XDECREF(self->quantiles);
    Py_XDECREF(self->eff_alphas);
    Py_XDECREF(self->emp_coverage);
    PyObject_Del(self);
}

static PyMemberDef ConformalPred_members[] = {
    {"point_pred", T_DOUBLE, offsetof(ConformalPredObj, point_pred), READONLY, NULL},
    {"scale", T_DOUBLE, offsetof(ConformalPredObj, scale), READONLY, NULL},
    {"alphas", T_OBJECT_EX, offsetof(ConformalPredObj, alphas), READONLY, NULL},
    {"lows", T_OBJECT_EX, offsetof(ConformalPredObj, lows), READONLY, NULL},
    {"highs", T_OBJECT_EX, offsetof(ConformalPredObj, highs), READONLY, NULL},
    {"quantiles", T_OBJECT_EX, offsetof(ConformalPredObj, quantiles), READONLY, NULL},
    {"eff_alphas", T_OBJECT_EX, offsetof(ConformalPredObj, eff_alphas), READONLY, NULL},
    {"emp_coverage", T_OBJECT_EX, offsetof(ConformalPredObj, emp_coverage), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *ConformalPred_intervals(ConformalPredObj *self, PyObject *Py_UNUSED(ignored)) {
    Py_ssize_t k = PyTuple_GET_SIZE(self->lows);
    PyObject *lst = PyList_New(k);
    if (lst == NULL) return NULL;
    for (Py_ssize_t i = 0; i < k; ++i) {
        PyObject *pair = PyTuple_Pack(2, PyTuple_GET_ITEM(self->lows, i), PyTuple_GET_ITEM(self->highs, i));
        if (pair == NULL) {
            Py_DECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, pair);
    }
    return lst;
}

static PyObject *ConformalPred_by_alpha(ConformalPredObj *self, PyObject *Py_UNUSED(ignored)) {
    Py_ssize_t k = PyTuple_GET_SIZE(self->alphas);
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    for (Py_ssize_t i = 0; i < k; ++i) {
        PyObject *pair = PyTuple_Pack(2, PyTuple_GET_ITEM(self->lows, i), PyTuple_GET_ITEM(self->highs, i));
        if (pair == NULL) {
            Py_DECREF(d);
            return NULL;
        }
        int rc = PyDict_SetItem(d, PyTuple_GET_ITEM(self->alphas, i), pair);
        Py_DECREF(pair);
        if (rc < 0) {
            Py_DECREF(d);
            return NULL;
        }
    }
    return d;
}

static PyObject *ConformalPred_to_dict(ConformalPredObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    Py_ssize_t k = PyTuple_GET_SIZE(self->alphas);
    PyObject *iv_list = PyList_New(k);
    if (iv_list == NULL) {
        Py_DECREF(d);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < k; ++i) {
        PyObject *item = PyDict_New();
        if (item == NULL) {
            Py_DECREF(iv_list);
            Py_DECREF(d);
            return NULL;
        }
        double a = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(self->alphas, i));
        double cov = round((1.0 - a) * 1e12) / 1e12;
        PyObject *a_obj = PyTuple_GET_ITEM(self->alphas, i);
        PyObject *lo_obj = PyTuple_GET_ITEM(self->lows, i);
        PyObject *hi_obj = PyTuple_GET_ITEM(self->highs, i);
        Py_INCREF(a_obj);
        Py_INCREF(lo_obj);
        Py_INCREF(hi_obj);
        if (dict_set_steal(item, "alpha", a_obj) < 0 ||
            dict_set_steal(item, "coverage", PyFloat_FromDouble(cov)) < 0 ||
            dict_set_steal(item, "low", lo_obj) < 0 ||
            dict_set_steal(item, "high", hi_obj) < 0) {
            Py_DECREF(item);
            Py_DECREF(iv_list);
            Py_DECREF(d);
            return NULL;
        }
        PyList_SET_ITEM(iv_list, i, item);
    }
    Py_INCREF(self->alphas);
    Py_INCREF(self->lows);
    Py_INCREF(self->highs);
    Py_INCREF(self->quantiles);
    Py_INCREF(self->eff_alphas);
    Py_INCREF(self->emp_coverage);
    if (dict_set_steal(d, "point_pred", PyFloat_FromDouble(self->point_pred)) < 0 ||
        dict_set_steal(d, "scale", PyFloat_FromDouble(self->scale)) < 0 ||
        dict_set_steal(d, "alphas", self->alphas) < 0 ||
        dict_set_steal(d, "lows", self->lows) < 0 ||
        dict_set_steal(d, "highs", self->highs) < 0 ||
        dict_set_steal(d, "quantiles", self->quantiles) < 0 ||
        dict_set_steal(d, "eff_alphas", self->eff_alphas) < 0 ||
        dict_set_steal(d, "emp_coverage", self->emp_coverage) < 0 ||
        dict_set_steal(d, "intervals", iv_list) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef ConformalPred_methods[] = {
    {"intervals", (PyCFunction)ConformalPred_intervals, METH_NOARGS, NULL},
    {"by_alpha", (PyCFunction)ConformalPred_by_alpha, METH_NOARGS, NULL},
    {"to_dict", (PyCFunction)ConformalPred_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject ConformalPredType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.conformal.ConformalPrediction",
    .tp_basicsize = sizeof(ConformalPredObj),
    .tp_dealloc = (destructor)ConformalPred_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = ConformalPred_members,
    .tp_methods = ConformalPred_methods,
};

/* --- SafeFuserStep --- */

typedef struct {
    PyObject_HEAD
    double fused_pred;
    double anchor_pred;
    char fallback_active;
    unsigned int n_quarantined;
    double sup_log_e;
    double deg_log_e;
    unsigned long long n_updates;
} SafeFuserStepObj;

static PyMemberDef SafeFuserStep_members[] = {
    {"fused_pred", T_DOUBLE, offsetof(SafeFuserStepObj, fused_pred), READONLY, NULL},
    {"anchor_pred", T_DOUBLE, offsetof(SafeFuserStepObj, anchor_pred), READONLY, NULL},
    {"fallback_active", T_BOOL, offsetof(SafeFuserStepObj, fallback_active), READONLY, NULL},
    {"n_quarantined", T_UINT, offsetof(SafeFuserStepObj, n_quarantined), READONLY, NULL},
    {"sup_log_e", T_DOUBLE, offsetof(SafeFuserStepObj, sup_log_e), READONLY, NULL},
    {"deg_log_e", T_DOUBLE, offsetof(SafeFuserStepObj, deg_log_e), READONLY, NULL},
    {"n_updates", T_ULONGLONG, offsetof(SafeFuserStepObj, n_updates), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *SafeFuserStep_to_dict(SafeFuserStepObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "fused_pred", PyFloat_FromDouble(self->fused_pred)) < 0 ||
        dict_set_steal(d, "anchor_pred", PyFloat_FromDouble(self->anchor_pred)) < 0 ||
        dict_set_steal(d, "fallback_active", PyBool_FromLong(self->fallback_active)) < 0 ||
        dict_set_steal(d, "n_quarantined", PyLong_FromUnsignedLong(self->n_quarantined)) < 0 ||
        dict_set_steal(d, "sup_log_e", PyFloat_FromDouble(self->sup_log_e)) < 0 ||
        dict_set_steal(d, "deg_log_e", PyFloat_FromDouble(self->deg_log_e)) < 0 ||
        dict_set_steal(d, "n_updates", PyLong_FromUnsignedLongLong(self->n_updates)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef SafeFuserStep_methods[] = {
    {"to_dict", (PyCFunction)SafeFuserStep_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SafeFuserStepType;
EPROC_DEFINE_FREELIST(sfstep, SafeFuserStepObj, SafeFuserStepType)
static void SafeFuserStep_dealloc(SafeFuserStepObj *self) { free_sfstep(self); }

static PyTypeObject SafeFuserStepType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.fusion.SafeFuserStep",
    .tp_basicsize = sizeof(SafeFuserStepObj),
    .tp_dealloc = (destructor)SafeFuserStep_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = SafeFuserStep_members,
    .tp_methods = SafeFuserStep_methods,
};

/* --- EchoSubmitResult --- */

typedef struct {
    PyObject_HEAD
    char accepted;
    double posterior_prob;
    double log_odds;
    double source_reliability;
    unsigned long long unique_count;
} EchoSubmitResultObj;

static PyMemberDef EchoSubmitResult_members[] = {
    {"accepted", T_BOOL, offsetof(EchoSubmitResultObj, accepted), READONLY, NULL},
    {"posterior_prob", T_DOUBLE, offsetof(EchoSubmitResultObj, posterior_prob), READONLY, NULL},
    {"log_odds", T_DOUBLE, offsetof(EchoSubmitResultObj, log_odds), READONLY, NULL},
    {"source_reliability", T_DOUBLE, offsetof(EchoSubmitResultObj, source_reliability), READONLY, NULL},
    {"unique_count", T_ULONGLONG, offsetof(EchoSubmitResultObj, unique_count), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *EchoSubmitResult_to_dict(EchoSubmitResultObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "accepted", PyBool_FromLong(self->accepted)) < 0 ||
        dict_set_steal(d, "posterior_prob", PyFloat_FromDouble(self->posterior_prob)) < 0 ||
        dict_set_steal(d, "log_odds", PyFloat_FromDouble(self->log_odds)) < 0 ||
        dict_set_steal(d, "source_reliability", PyFloat_FromDouble(self->source_reliability)) < 0 ||
        dict_set_steal(d, "unique_count", PyLong_FromUnsignedLongLong(self->unique_count)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef EchoSubmitResult_methods[] = {
    {"to_dict", (PyCFunction)EchoSubmitResult_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject EchoSubmitResultType;
EPROC_DEFINE_FREELIST(echores, EchoSubmitResultObj, EchoSubmitResultType)
static void EchoSubmitResult_dealloc(EchoSubmitResultObj *self) { free_echores(self); }

static PyTypeObject EchoSubmitResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.fusion.EchoSubmitResult",
    .tp_basicsize = sizeof(EchoSubmitResultObj),
    .tp_dealloc = (destructor)EchoSubmitResult_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = EchoSubmitResult_members,
    .tp_methods = EchoSubmitResult_methods,
};

/* --- VarOptItem --- */

typedef struct {
    PyObject_HEAD
    unsigned long long item_id;
    double orig_weight;
    double adj_weight;
    double value;
} VarOptItemObj;

static PyMemberDef VarOptItem_members[] = {
    {"item_id", T_ULONGLONG, offsetof(VarOptItemObj, item_id), READONLY, NULL},
    {"orig_weight", T_DOUBLE, offsetof(VarOptItemObj, orig_weight), READONLY, NULL},
    {"adj_weight", T_DOUBLE, offsetof(VarOptItemObj, adj_weight), READONLY, NULL},
    {"value", T_DOUBLE, offsetof(VarOptItemObj, value), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *VarOptItem_to_dict(VarOptItemObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "item_id", PyLong_FromUnsignedLongLong(self->item_id)) < 0 ||
        dict_set_steal(d, "orig_weight", PyFloat_FromDouble(self->orig_weight)) < 0 ||
        dict_set_steal(d, "adj_weight", PyFloat_FromDouble(self->adj_weight)) < 0 ||
        dict_set_steal(d, "value", PyFloat_FromDouble(self->value)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef VarOptItem_methods[] = {
    {"to_dict", (PyCFunction)VarOptItem_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject VarOptItemType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.subsample.VarOptItem",
    .tp_basicsize = sizeof(VarOptItemObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = VarOptItem_members,
    .tp_methods = VarOptItem_methods,
};

/* --- RLSStepResult --- */

typedef struct {
    PyObject_HEAD
    char keep;
    double leverage;
    double prob;
    double ht_weight;
} RLSStepResultObj;

static PyMemberDef RLSStepResult_members[] = {
    {"keep", T_BOOL, offsetof(RLSStepResultObj, keep), READONLY, NULL},
    {"leverage", T_DOUBLE, offsetof(RLSStepResultObj, leverage), READONLY, NULL},
    {"prob", T_DOUBLE, offsetof(RLSStepResultObj, prob), READONLY, NULL},
    {"ht_weight", T_DOUBLE, offsetof(RLSStepResultObj, ht_weight), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *RLSStepResult_to_dict(RLSStepResultObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    if (dict_set_steal(d, "keep", PyBool_FromLong(self->keep)) < 0 ||
        dict_set_steal(d, "leverage", PyFloat_FromDouble(self->leverage)) < 0 ||
        dict_set_steal(d, "prob", PyFloat_FromDouble(self->prob)) < 0 ||
        dict_set_steal(d, "ht_weight", PyFloat_FromDouble(self->ht_weight)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef RLSStepResult_methods[] = {
    {"to_dict", (PyCFunction)RLSStepResult_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject RLSStepResultType;
EPROC_DEFINE_FREELIST(rlsres, RLSStepResultObj, RLSStepResultType)
static void RLSStepResult_dealloc(RLSStepResultObj *self) { free_rlsres(self); }

static PyTypeObject RLSStepResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.subsample.RLSStepResult",
    .tp_basicsize = sizeof(RLSStepResultObj),
    .tp_dealloc = (destructor)RLSStepResult_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = RLSStepResult_members,
    .tp_methods = RLSStepResult_methods,
};

/* --- SiftAction --- */

typedef struct {
    PyObject_HEAD
    int action_type;
    PyObject *action_name;
    unsigned int target_idx;
    double evoi_per_cost;
} SiftActionObj;

static void SiftAction_dealloc(SiftActionObj *self) {
    Py_XDECREF(self->action_name);
    PyObject_Del(self);
}

static PyMemberDef SiftAction_members[] = {
    {"action_type", T_INT, offsetof(SiftActionObj, action_type), READONLY, NULL},
    {"action_name", T_OBJECT_EX, offsetof(SiftActionObj, action_name), READONLY, NULL},
    {"target_idx", T_UINT, offsetof(SiftActionObj, target_idx), READONLY, NULL},
    {"evoi_per_cost", T_DOUBLE, offsetof(SiftActionObj, evoi_per_cost), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}
};

static PyObject *SiftAction_to_dict(SiftActionObj *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *d = PyDict_New();
    if (d == NULL) return NULL;
    Py_INCREF(self->action_name);
    if (dict_set_steal(d, "action_type", PyLong_FromLong(self->action_type)) < 0 ||
        dict_set_steal(d, "action_name", self->action_name) < 0 ||
        dict_set_steal(d, "target_idx", PyLong_FromUnsignedLong(self->target_idx)) < 0 ||
        dict_set_steal(d, "evoi_per_cost", PyFloat_FromDouble(self->evoi_per_cost)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

static PyMethodDef SiftAction_methods[] = {
    {"to_dict", (PyCFunction)SiftAction_to_dict, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SiftActionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.subsample.SiftAction",
    .tp_basicsize = sizeof(SiftActionObj),
    .tp_dealloc = (destructor)SiftAction_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = SiftAction_members,
    .tp_methods = SiftAction_methods,
};

/* ============================================================================
 * 2. ENGINE WRAPPERS — EProcess, ConfidenceSequence, EDetector, AgentRace
 * ============================================================================ */

/* --- EProcess --- */

typedef struct {
    PyObject_HEAD
    eproc_bet_t state;
    double mu0;
    const char *side_name;
} EProcessObj;

static int EProcess_init(EProcessObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"mu0", "alpha", "lower", "upper", "side", NULL};
    double mu0 = 0.0, alpha = 0.05, lower = 0.0, upper = 1.0;
    PyObject *side_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|dddO", kwlist,
                                     &mu0, &alpha, &lower, &upper, &side_obj)) {
        return -1;
    }
    int32_t side_code = EPROC_SIDE_UPPER;
    const char *side_name = "upper";
    if (side_obj != NULL) {
        if (parse_side(side_obj, &side_code, &side_name) < 0) {
            return -1;
        }
    }
    self->mu0 = mu0;
    self->side_name = side_name;
    int rc = eproc_bet_init(&self->state, mu0, alpha, lower, upper, side_code);
    return set_eproc_error(rc, "EProcess.__init__");
}

static PyObject *EProcess_update(EProcessObj *self, PyObject *arg) {
    double x;
    if (fast_as_double(arg, &x) < 0) return NULL;
    eproc_bet_result_t res;
    int rc = eproc_bet_update(&self->state, x, &res);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "EProcess.update");
        return NULL;
    }
    return new_eprocess_result(&res);
}

static PyObject *EProcess_update_batch(EProcessObj *self, PyObject *arg) {
    double_buf_t b;
    if (acquire_double_buf(arg, &b) < 0) return NULL;
    eproc_bet_result_t res;
    int rc = EPROC_OK;
    if (b.len == 0) {
        eproc_bet_query(&self->state, &res);
    } else if (b.len >= 256) {
        Py_BEGIN_ALLOW_THREADS
        rc = eproc_bet_update_batch(&self->state, b.ptr, (size_t)b.len, &res);
        Py_END_ALLOW_THREADS
    } else {
        rc = eproc_bet_update_batch(&self->state, b.ptr, (size_t)b.len, &res);
    }
    release_double_buf(&b);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "EProcess.update_batch");
        return NULL;
    }
    return new_eprocess_result(&res);
}

static PyObject *EProcess_result(EProcessObj *self, PyObject *Py_UNUSED(ignored)) {
    eproc_bet_result_t res;
    eproc_bet_query(&self->state, &res);
    return new_eprocess_result(&res);
}

static PyObject *EProcess_get_n(EProcessObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n); }
static PyObject *EProcess_get_e_value(EProcessObj *s, void *c) { eproc_bet_result_t r; eproc_bet_query(&s->state, &r); return PyFloat_FromDouble(r.e_value); }
static PyObject *EProcess_get_log_e(EProcessObj *s, void *c) { return PyFloat_FromDouble(s->state.log_e); }
static PyObject *EProcess_get_max_log_e(EProcessObj *s, void *c) { return PyFloat_FromDouble(s->state.max_log_e); }
static PyObject *EProcess_get_p_value(EProcessObj *s, void *c) { eproc_bet_result_t r; eproc_bet_query(&s->state, &r); return PyFloat_FromDouble(r.p_value); }
static PyObject *EProcess_get_rejected(EProcessObj *s, void *c) { eproc_bet_result_t r; eproc_bet_query(&s->state, &r); return PyBool_FromLong(r.rejected); }
static PyObject *EProcess_get_mean(EProcessObj *s, void *c) { eproc_bet_result_t r; eproc_bet_query(&s->state, &r); return PyFloat_FromDouble(r.mean); }
static PyObject *EProcess_get_rejected_at(EProcessObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.rejected_at); }
static PyObject *EProcess_get_n_invalid(EProcessObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_invalid); }
static PyObject *EProcess_get_alpha(EProcessObj *s, void *c) { return PyFloat_FromDouble(s->state.alpha); }
static PyObject *EProcess_get_mu0(EProcessObj *s, void *c) { return PyFloat_FromDouble(s->mu0); }
static PyObject *EProcess_get_lower(EProcessObj *s, void *c) { return PyFloat_FromDouble(s->state.lower); }
static PyObject *EProcess_get_upper(EProcessObj *s, void *c) { return PyFloat_FromDouble(s->state.upper); }
static PyObject *EProcess_get_side(EProcessObj *s, void *c) { return PyUnicode_FromString(s->side_name ? s->side_name : "upper"); }

static PyMethodDef EProcess_methods[] = {
    {"update", (PyCFunction)EProcess_update, METH_O, NULL},
    {"update_batch", (PyCFunction)EProcess_update_batch, METH_O, NULL},
    {"result", (PyCFunction)EProcess_result, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef EProcess_getset[] = {
    {"n", (getter)EProcess_get_n, NULL, NULL, NULL},
    {"e_value", (getter)EProcess_get_e_value, NULL, NULL, NULL},
    {"log_e", (getter)EProcess_get_log_e, NULL, NULL, NULL},
    {"max_log_e", (getter)EProcess_get_max_log_e, NULL, NULL, NULL},
    {"p_value", (getter)EProcess_get_p_value, NULL, NULL, NULL},
    {"rejected", (getter)EProcess_get_rejected, NULL, NULL, NULL},
    {"mean", (getter)EProcess_get_mean, NULL, NULL, NULL},
    {"rejected_at", (getter)EProcess_get_rejected_at, NULL, NULL, NULL},
    {"n_invalid", (getter)EProcess_get_n_invalid, NULL, NULL, NULL},
    {"alpha", (getter)EProcess_get_alpha, NULL, NULL, NULL},
    {"mu0", (getter)EProcess_get_mu0, NULL, NULL, NULL},
    {"lower", (getter)EProcess_get_lower, NULL, NULL, NULL},
    {"upper", (getter)EProcess_get_upper, NULL, NULL, NULL},
    {"side", (getter)EProcess_get_side, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject EProcessType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.EProcess",
    .tp_basicsize = sizeof(EProcessObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)EProcess_init,
    .tp_methods = EProcess_methods,
    .tp_getset = EProcess_getset,
};

/* --- ConfidenceSequence --- */

typedef struct {
    PyObject_HEAD
    eproc_cs_t state;
    const char *mode_name;
} CSObj;

static int CS_init(CSObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"alpha", "lower", "upper", "mode", "sigma", "v_opt", NULL};
    double alpha = 0.05, lower = 0.0, upper = 1.0, sigma = 1.0, v_opt = 100.0;
    const char *mode = "eb";
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dddsdd", kwlist,
                                     &alpha, &lower, &upper, &mode, &sigma, &v_opt)) {
        return -1;
    }
    int rc;
    if (strcmp(mode, "eb") == 0 || strcmp(mode, "empirical_bernstein") == 0 ||
        strcmp(mode, "empirical-bernstein") == 0) {
        self->mode_name = "eb";
        rc = eproc_cs_init_eb(&self->state, alpha, lower, upper);
    } else if (strcmp(mode, "normal") == 0 || strcmp(mode, "normal_mixture") == 0 ||
               strcmp(mode, "normal-mixture") == 0 || strcmp(mode, "subgaussian") == 0) {
        self->mode_name = "normal";
        rc = eproc_cs_init_normal(&self->state, alpha, sigma, v_opt);
    } else {
        PyErr_Format(PyExc_ValueError, "Invalid mode '%s'; expected 'eb' or 'normal'", mode);
        return -1;
    }
    return set_eproc_error(rc, "ConfidenceSequence.__init__");
}

static PyObject *CS_update(CSObj *self, PyObject *arg) {
    double x;
    if (fast_as_double(arg, &x) < 0) return NULL;
    eproc_cs_result_t res;
    int rc = eproc_cs_update(&self->state, x, &res);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "ConfidenceSequence.update");
        return NULL;
    }
    return new_cs_result(&res);
}

static PyObject *CS_update_batch(CSObj *self, PyObject *arg) {
    double_buf_t b;
    if (acquire_double_buf(arg, &b) < 0) return NULL;
    eproc_cs_result_t res;
    int rc = EPROC_OK;
    if (b.len == 0) {
        eproc_cs_query(&self->state, &res);
    } else if (b.len >= 256) {
        Py_BEGIN_ALLOW_THREADS
        rc = eproc_cs_update_batch(&self->state, b.ptr, (size_t)b.len, &res);
        Py_END_ALLOW_THREADS
    } else {
        rc = eproc_cs_update_batch(&self->state, b.ptr, (size_t)b.len, &res);
    }
    release_double_buf(&b);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "ConfidenceSequence.update_batch");
        return NULL;
    }
    return new_cs_result(&res);
}

static PyObject *CS_query(CSObj *self, PyObject *Py_UNUSED(ignored)) {
    eproc_cs_result_t res;
    eproc_cs_query(&self->state, &res);
    return new_cs_result(&res);
}

static PyObject *CS_get_bounds(CSObj *s, void *c) {
    eproc_cs_result_t r;
    eproc_cs_query(&s->state, &r);
    return Py_BuildValue("(dd)", r.low, r.high);
}
static PyObject *CS_get_width(CSObj *s, void *c) { eproc_cs_result_t r; eproc_cs_query(&s->state, &r); return PyFloat_FromDouble(r.width); }
static PyObject *CS_get_mean(CSObj *s, void *c) { eproc_cs_result_t r; eproc_cs_query(&s->state, &r); return PyFloat_FromDouble(r.mean); }
static PyObject *CS_get_n(CSObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n); }
static PyObject *CS_get_alpha(CSObj *s, void *c) { return PyFloat_FromDouble(s->state.alpha); }
static PyObject *CS_get_mode(CSObj *s, void *c) { return PyUnicode_FromString(s->mode_name ? s->mode_name : "eb"); }

static PyMethodDef CS_methods[] = {
    {"update", (PyCFunction)CS_update, METH_O, NULL},
    {"update_batch", (PyCFunction)CS_update_batch, METH_O, NULL},
    {"query", (PyCFunction)CS_query, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef CS_getset[] = {
    {"bounds", (getter)CS_get_bounds, NULL, NULL, NULL},
    {"width", (getter)CS_get_width, NULL, NULL, NULL},
    {"mean", (getter)CS_get_mean, NULL, NULL, NULL},
    {"n", (getter)CS_get_n, NULL, NULL, NULL},
    {"alpha", (getter)CS_get_alpha, NULL, NULL, NULL},
    {"mode", (getter)CS_get_mode, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject CSType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.ConfidenceSequence",
    .tp_basicsize = sizeof(CSObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CS_init,
    .tp_methods = CS_methods,
    .tp_getset = CS_getset,
};

/* --- EDetector --- */

typedef struct {
    PyObject_HEAD
    eproc_edetect_t state;
    double mu0;
    const char *side_name;
} EDetectorObj;

static int EDetector_init(EDetectorObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"mu0", "alpha", "lower", "upper", "side", NULL};
    double mu0 = 0.0, alpha = 0.05, lower = 0.0, upper = 1.0;
    PyObject *side_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|dddO", kwlist,
                                     &mu0, &alpha, &lower, &upper, &side_obj)) {
        return -1;
    }
    int32_t side_code = EPROC_SIDE_UPPER;
    const char *side_name = "upper";
    if (side_obj != NULL) {
        if (parse_side(side_obj, &side_code, &side_name) < 0) return -1;
    }
    self->mu0 = mu0;
    self->side_name = side_name;
    int rc = eproc_edetect_init(&self->state, mu0, alpha, lower, upper, side_code);
    return set_eproc_error(rc, "EDetector.__init__");
}

static PyObject *EDetector_update(EDetectorObj *self, PyObject *arg) {
    double x;
    if (fast_as_double(arg, &x) < 0) return NULL;
    eproc_edetect_result_t res;
    int rc = eproc_edetect_update(&self->state, x, &res);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "EDetector.update");
        return NULL;
    }
    return new_edetect_result(res.n, res.log_stat, res.stat, res.alarm);
}

static PyObject *EDetector_update_batch(EDetectorObj *self, PyObject *arg) {
    double_buf_t b;
    if (acquire_double_buf(arg, &b) < 0) return NULL;
    if (b.len == 0) {
        release_double_buf(&b);
        double ls = self->state.log_cusum;
        double st = exp(ls > 700.0 ? 700.0 : ls);
        int al = (ls >= self->state.log_thresh) ? 1 : 0;
        return new_edetect_result(self->state.n, ls, st, al);
    }
    eproc_edetect_result_t res;
    int rc = EPROC_OK;
    if (b.len >= 256) {
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t i = 0; i < b.len; ++i) {
            rc = eproc_edetect_update(&self->state, b.ptr[i], &res);
            if (rc != EPROC_OK) break;
        }
        Py_END_ALLOW_THREADS
    } else {
        for (Py_ssize_t i = 0; i < b.len; ++i) {
            rc = eproc_edetect_update(&self->state, b.ptr[i], &res);
            if (rc != EPROC_OK) break;
        }
    }
    release_double_buf(&b);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "EDetector.update_batch");
        return NULL;
    }
    return new_edetect_result(res.n, res.log_stat, res.stat, res.alarm);
}

static PyObject *EDetector_reset_alarm(EDetectorObj *self, PyObject *Py_UNUSED(ignored)) {
    eproc_edetect_reset_alarm(&self->state);
    Py_RETURN_NONE;
}

static PyObject *EDetector_get_n(EDetectorObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n); }
static PyObject *EDetector_get_log_stat(EDetectorObj *s, void *c) { return PyFloat_FromDouble(s->state.log_cusum); }
static PyObject *EDetector_get_stat(EDetectorObj *s, void *c) {
    double ls = s->state.log_cusum;
    return PyFloat_FromDouble(exp(ls > 700.0 ? 700.0 : ls));
}
static PyObject *EDetector_get_alarm_at(EDetectorObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.alarm_at); }
static PyObject *EDetector_get_n_alarms(EDetectorObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_alarms); }

static PyMethodDef EDetector_methods[] = {
    {"update", (PyCFunction)EDetector_update, METH_O, NULL},
    {"update_batch", (PyCFunction)EDetector_update_batch, METH_O, NULL},
    {"reset_alarm", (PyCFunction)EDetector_reset_alarm, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef EDetector_getset[] = {
    {"n", (getter)EDetector_get_n, NULL, NULL, NULL},
    {"log_stat", (getter)EDetector_get_log_stat, NULL, NULL, NULL},
    {"stat", (getter)EDetector_get_stat, NULL, NULL, NULL},
    {"alarm_at", (getter)EDetector_get_alarm_at, NULL, NULL, NULL},
    {"n_alarms", (getter)EDetector_get_n_alarms, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject EDetectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.EDetector",
    .tp_basicsize = sizeof(EDetectorObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)EDetector_init,
    .tp_methods = EDetector_methods,
    .tp_getset = EDetector_getset,
};

/* --- AgentRace --- */

typedef struct {
    PyObject_HEAD
    eproc_race_t state;
    PyObject *names;        /* tuple of str or NULL */
    PyObject *name_to_idx;  /* dict[str, int] or NULL */
} AgentRaceObj;

static void AgentRace_dealloc(AgentRaceObj *self) {
    Py_XDECREF(self->names);
    Py_XDECREF(self->name_to_idx);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int AgentRace_init(AgentRaceObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"n_arms", "alpha", "lower", "upper", "margin", "names", NULL};
    long n_arms_l = 0;
    double alpha = 0.05, lower = 0.0, upper = 1.0, margin = 0.0;
    PyObject *names_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|ddddO", kwlist,
                                     &n_arms_l, &alpha, &lower, &upper, &margin, &names_obj)) {
        return -1;
    }
    Py_CLEAR(self->names);
    Py_CLEAR(self->name_to_idx);

    if (names_obj != Py_None) {
        PyObject *seq = PySequence_Fast(names_obj, "names must be a sequence of strings");
        if (seq == NULL) return -1;
        Py_ssize_t nn = PySequence_Fast_GET_SIZE(seq);
        if (nn != n_arms_l) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError, "len(names) (%zd) must equal n_arms (%ld)", nn, n_arms_l);
            return -1;
        }
        PyObject *tup = PyTuple_New(nn);
        PyObject *mp = PyDict_New();
        if (tup == NULL || mp == NULL) {
            Py_XDECREF(tup);
            Py_XDECREF(mp);
            Py_DECREF(seq);
            return -1;
        }
        PyObject **items = PySequence_Fast_ITEMS(seq);
        for (Py_ssize_t i = 0; i < nn; ++i) {
            PyObject *s = PyObject_Str(items[i]);
            if (s == NULL) {
                Py_DECREF(tup);
                Py_DECREF(mp);
                Py_DECREF(seq);
                return -1;
            }
            if (PyDict_Contains(mp, s) == 1) {
                Py_DECREF(s);
                Py_DECREF(tup);
                Py_DECREF(mp);
                Py_DECREF(seq);
                PyErr_SetString(PyExc_ValueError, "Arm names must be unique");
                return -1;
            }
            PyObject *idx_obj = PyLong_FromSsize_t(i);
            if (idx_obj == NULL || PyDict_SetItem(mp, s, idx_obj) < 0) {
                Py_XDECREF(idx_obj);
                Py_DECREF(s);
                Py_DECREF(tup);
                Py_DECREF(mp);
                Py_DECREF(seq);
                return -1;
            }
            Py_DECREF(idx_obj);
            PyTuple_SET_ITEM(tup, i, s);
        }
        Py_DECREF(seq);
        self->names = tup;
        self->name_to_idx = mp;
    }

    if (n_arms_l < 0 || n_arms_l > EPROC_MAX_ARMS) {
        PyErr_Format(PyExc_ValueError, "AgentRace.__init__: n_arms out of range [2, %d]", EPROC_MAX_ARMS);
        return -1;
    }
    int rc = eproc_race_init(&self->state, (uint32_t)n_arms_l, alpha, lower, upper, margin);
    return set_eproc_error(rc, "AgentRace.__init__");
}

static int race_resolve_arm(AgentRaceObj *self, PyObject *arm_obj, uint32_t *out_idx) {
    if (PyUnicode_Check(arm_obj)) {
        if (self->name_to_idx == NULL) {
            PyErr_SetObject(PyExc_KeyError, arm_obj);
            return -1;
        }
        PyObject *found = PyDict_GetItemWithError(self->name_to_idx, arm_obj);
        if (found == NULL) {
            if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, arm_obj);
            return -1;
        }
        *out_idx = (uint32_t)PyLong_AsUnsignedLong(found);
        return 0;
    }
    long long v = PyLong_AsLongLong(arm_obj);
    if (v == -1 && PyErr_Occurred()) return -1;
    if (v < 0 || v >= (long long)self->state.n_arms) {
        PyErr_Format(PyExc_ValueError, "Arm index %lld out of range [0, %u)", v, self->state.n_arms);
        return -1;
    }
    *out_idx = (uint32_t)v;
    return 0;
}

static PyObject *race_status_obj(AgentRaceObj *self, const eproc_race_status_t *st) {
    RaceStatusObj *o = PyObject_New(RaceStatusObj, &RaceStatusType);
    if (o == NULL) return NULL;
    o->n_arms = st->n_arms;
    o->n_active = st->n_active;
    o->winner_raw = st->winner;
    o->leader_raw = st->leader;
    o->total_obs = (unsigned long long)st->total_obs;
    o->leader_low = st->leader_low;
    o->leader_high = st->leader_high;
    o->runner_up_high = st->runner_up_high;
    if (self->names != NULL && st->winner >= 0 && st->winner < (int32_t)st->n_arms) {
        o->winner_name = PyTuple_GET_ITEM(self->names, st->winner);
    } else {
        o->winner_name = Py_None;
    }
    Py_INCREF(o->winner_name);
    if (self->names != NULL && st->leader >= 0 && st->leader < (int32_t)st->n_arms) {
        o->leader_name = PyTuple_GET_ITEM(self->names, st->leader);
    } else {
        o->leader_name = Py_None;
    }
    Py_INCREF(o->leader_name);
    return (PyObject *)o;
}

static PyObject *AgentRace_update(AgentRaceObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *arm_obj = NULL, *rew_obj = NULL;
    if (kwnames == NULL && nargs == 2) {
        arm_obj = args[0];
        rew_obj = args[1];
    } else {
        static char *kwlist[] = {"arm", "reward", NULL};
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            Py_INCREF(args[i]);
            PyTuple_SET_ITEM(tup, i, args[i]);
        }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < nkw; ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "OO", kwlist, &arm_obj, &rew_obj);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
    }
    uint32_t idx;
    double reward;
    if (race_resolve_arm(self, arm_obj, &idx) < 0) return NULL;
    if (fast_as_double(rew_obj, &reward) < 0) return NULL;
    eproc_race_status_t st;
    int rc = eproc_race_update(&self->state, idx, reward, &st);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "AgentRace.update");
        return NULL;
    }
    return race_status_obj(self, &st);
}

static PyObject *AgentRace_update_batch(AgentRaceObj *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "update_batch expects exactly 2 positional arguments (arms, rewards)");
        return NULL;
    }
    PyObject *arms_obj = args[0];
    PyObject *rew_obj = args[1];

    double_buf_t rb;
    if (acquire_double_buf(rew_obj, &rb) < 0) return NULL;

    const uint32_t *arm_ptr = NULL;
    Py_ssize_t arm_len = 0;
    Py_buffer arm_view;
    int has_arm_view = 0;
    uint32_t *arm_heap = NULL;

    if (PyObject_CheckBuffer(arms_obj)) {
        if (PyObject_GetBuffer(arms_obj, &arm_view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) == 0) {
            if (arm_view.ndim >= 1 && arm_view.itemsize == 4 &&
                is_native_uint32_format(arm_view.format) &&
                PyBuffer_IsContiguous(&arm_view, 'C')) {
                arm_ptr = (const uint32_t *)arm_view.buf;
                arm_len = arm_view.len / 4;
                has_arm_view = 1;
            } else {
                PyBuffer_Release(&arm_view);
            }
        } else {
            PyErr_Clear();
        }
    }

    if (!has_arm_view) {
        if (PyUnicode_Check(arms_obj) || PyBytes_Check(arms_obj) || PyByteArray_Check(arms_obj)) {
            release_double_buf(&rb);
            PyErr_SetString(PyExc_TypeError, "Expected a sequence of arm indices or names");
            return NULL;
        }
        PyObject *seq = PySequence_Fast(arms_obj, "arms must be a sequence");
        if (seq == NULL) {
            release_double_buf(&rb);
            return NULL;
        }
        arm_len = PySequence_Fast_GET_SIZE(seq);
        if (arm_len > 0) {
            arm_heap = (uint32_t *)PyMem_Malloc((size_t)arm_len * sizeof(uint32_t));
            if (arm_heap == NULL) {
                Py_DECREF(seq);
                release_double_buf(&rb);
                return PyErr_NoMemory();
            }
            PyObject **items = PySequence_Fast_ITEMS(seq);
            for (Py_ssize_t i = 0; i < arm_len; ++i) {
                if (race_resolve_arm(self, items[i], &arm_heap[i]) < 0) {
                    PyMem_Free(arm_heap);
                    Py_DECREF(seq);
                    release_double_buf(&rb);
                    return NULL;
                }
            }
            arm_ptr = arm_heap;
        }
        Py_DECREF(seq);
    }

    if (arm_len != rb.len) {
        if (has_arm_view) PyBuffer_Release(&arm_view);
        if (arm_heap) PyMem_Free(arm_heap);
        release_double_buf(&rb);
        PyErr_Format(PyExc_ValueError, "arms length (%zd) must match rewards length (%zd)", arm_len, rb.len);
        return NULL;
    }

    eproc_race_status_t st;
    int rc = EPROC_OK;
    if (arm_len == 0) {
        eproc_race_status(&self->state, &st);
    } else if (arm_len >= 256) {
        Py_BEGIN_ALLOW_THREADS
        rc = eproc_race_update_batch(&self->state, arm_ptr, rb.ptr, (size_t)arm_len, &st);
        Py_END_ALLOW_THREADS
    } else {
        rc = eproc_race_update_batch(&self->state, arm_ptr, rb.ptr, (size_t)arm_len, &st);
    }

    if (has_arm_view) PyBuffer_Release(&arm_view);
    if (arm_heap) PyMem_Free(arm_heap);
    release_double_buf(&rb);

    if (rc != EPROC_OK) {
        set_eproc_error(rc, "AgentRace.update_batch");
        return NULL;
    }
    return race_status_obj(self, &st);
}

static PyObject *AgentRace_suggest_arm(AgentRaceObj *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(eproc_race_suggest_arm(&self->state));
}

static PyObject *AgentRace_status(AgentRaceObj *self, PyObject *Py_UNUSED(ignored)) {
    eproc_race_status_t st;
    eproc_race_status(&self->state, &st);
    return race_status_obj(self, &st);
}

static PyObject *AgentRace_arm_state(AgentRaceObj *self, PyObject *arm_obj) {
    uint32_t idx;
    if (race_resolve_arm(self, arm_obj, &idx) < 0) return NULL;
    eproc_cs_result_t cs;
    int32_t is_active = 0;
    int rc = eproc_race_arm_state(&self->state, idx, &cs, &is_active);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "AgentRace.arm_state");
        return NULL;
    }
    RaceArmStateObj *o = PyObject_New(RaceArmStateObj, &RaceArmStateType);
    if (o == NULL) return NULL;
    o->arm = idx;
    o->name = (self->names != NULL) ? PyTuple_GET_ITEM(self->names, idx) : Py_None;
    Py_INCREF(o->name);
    o->active = is_active ? 1 : 0;
    o->n = (unsigned long long)cs.n;
    o->mean = cs.mean;
    o->low = cs.low;
    o->high = cs.high;
    o->width = cs.width;
    o->eliminated_at = (unsigned long long)self->state.eliminated_at[idx];
    return (PyObject *)o;
}

static PyObject *AgentRace_get_n_arms(AgentRaceObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.n_arms); }
static PyObject *AgentRace_get_winner(AgentRaceObj *s, void *c) {
    if (s->state.winner < 0) Py_RETURN_NONE;
    return PyLong_FromLong(s->state.winner);
}
static PyObject *AgentRace_get_winner_name(AgentRaceObj *s, void *c) {
    if (s->state.winner < 0 || s->names == NULL) Py_RETURN_NONE;
    PyObject *nm = PyTuple_GET_ITEM(s->names, s->state.winner);
    Py_INCREF(nm);
    return nm;
}
static PyObject *AgentRace_get_active_arms(AgentRaceObj *s, void *c) {
    PyObject *lst = PyList_New(0);
    if (lst == NULL) return NULL;
    for (uint32_t i = 0; i < s->state.n_arms; ++i) {
        if (s->state.active[i] != 0) {
            PyObject *v = PyLong_FromUnsignedLong(i);
            if (v == NULL || PyList_Append(lst, v) < 0) {
                Py_XDECREF(v);
                Py_DECREF(lst);
                return NULL;
            }
            Py_DECREF(v);
        }
    }
    return lst;
}
static PyObject *AgentRace_get_names(AgentRaceObj *s, void *c) {
    if (s->names == NULL) Py_RETURN_NONE;
    Py_INCREF(s->names);
    return s->names;
}

static PyMethodDef AgentRace_methods[] = {
    {"update", EPROC_PYCFUNC(AgentRace_update), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"update_batch", EPROC_PYCFUNC(AgentRace_update_batch), METH_FASTCALL, NULL},
    {"suggest_arm", (PyCFunction)AgentRace_suggest_arm, METH_NOARGS, NULL},
    {"status", (PyCFunction)AgentRace_status, METH_NOARGS, NULL},
    {"arm_state", (PyCFunction)AgentRace_arm_state, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef AgentRace_getset[] = {
    {"n_arms", (getter)AgentRace_get_n_arms, NULL, NULL, NULL},
    {"winner", (getter)AgentRace_get_winner, NULL, NULL, NULL},
    {"winner_name", (getter)AgentRace_get_winner_name, NULL, NULL, NULL},
    {"active_arms", (getter)AgentRace_get_active_arms, NULL, NULL, NULL},
    {"names", (getter)AgentRace_get_names, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject AgentRaceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.eprocess.AgentRace",
    .tp_basicsize = sizeof(AgentRaceObj),
    .tp_dealloc = (destructor)AgentRace_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)AgentRace_init,
    .tp_methods = AgentRace_methods,
    .tp_getset = AgentRace_getset,
};

/* --- combine_evalues --- */

static PyObject *py_combine_evalues(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"log_es", "independent", NULL};
    PyObject *log_es_obj = NULL;
    int independent = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", kwlist, &log_es_obj, &independent)) {
        return NULL;
    }
    double_buf_t b;
    if (acquire_double_buf(log_es_obj, &b) < 0) return NULL;
    if (b.len == 0) {
        release_double_buf(&b);
        PyErr_SetString(PyExc_ValueError, "combine_evalues: log_es must not be empty");
        return NULL;
    }
    double out;
    if (independent) {
        out = 0.0;
        for (Py_ssize_t i = 0; i < b.len; ++i) {
            out = eproc_combine_independent_log(out, b.ptr[i]);
        }
    } else {
        out = eproc_combine_dependent_log(b.ptr, (size_t)b.len);
    }
    release_double_buf(&b);
    return PyFloat_FromDouble(out);
}

/* ============================================================================
 * 3. CONFORMAL WRAPPERS — MultiRateConformal, Delayed, Budget, ConformalMonoid
 * ============================================================================ */

/* --- MultiRateConformal --- */

typedef struct {
    PyObject_HEAD
    eproc_conformal_t state;
    PyObject *alphas_tuple;
} MultiRateConformalObj;

static void MultiRateConformal_dealloc(MultiRateConformalObj *self) {
    Py_XDECREF(self->alphas_tuple);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int MultiRateConformal_init(MultiRateConformalObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"alphas", "window_size", "pid_ki", NULL};
    PyObject *alphas_obj = NULL;
    long window_size = 128;
    double pid_ki = 0.05;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Old", kwlist, &alphas_obj, &window_size, &pid_ki)) {
        return -1;
    }
    double alphas[EPROC_MAX_RATES] = {0.50, 0.20, 0.10, 0.05, 0.01};
    uint32_t n_rates = 5;
    if (alphas_obj != NULL) {
        if (prepare_alphas(alphas_obj, alphas, &n_rates) < 0) return -1;
    }
    if (window_size < 0) {
        PyErr_SetString(PyExc_ValueError, "MultiRateConformal.__init__: window_size must be positive");
        return -1;
    }
    int rc = eproc_conformal_init(&self->state, alphas, n_rates, (uint32_t)window_size, pid_ki);
    if (set_eproc_error(rc, "MultiRateConformal.__init__") < 0) return -1;

    Py_CLEAR(self->alphas_tuple);
    self->alphas_tuple = make_double_tuple(self->state.target_alphas, self->state.n_rates);
    return (self->alphas_tuple == NULL) ? -1 : 0;
}

static PyObject *MultiRateConformal_predict(MultiRateConformalObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    double pred = 0.0, scale = 1.0;
    if (kwnames == NULL && (nargs == 1 || nargs == 2)) {
        if (fast_as_double(args[0], &pred) < 0) return NULL;
        if (nargs == 2 && fast_as_double(args[1], &scale) < 0) return NULL;
    } else {
        static char *kwlist[] = {"pred", "scale", NULL};
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) { Py_INCREF(args[i]); PyTuple_SET_ITEM(tup, i, args[i]); }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "d|d", kwlist, &pred, &scale);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
    }

    double lows[EPROC_MAX_RATES], highs[EPROC_MAX_RATES];
    eproc_conformal_out_t meta;
    eproc_conformal_predict(&self->state, pred, scale, lows, highs, &meta);

    uint32_t k = meta.n_rates;
    PyObject *lo_t = make_double_tuple(lows, k);
    PyObject *hi_t = make_double_tuple(highs, k);
    PyObject *q_t = make_double_tuple(meta.quantiles, k);
    PyObject *ea_t = make_double_tuple(meta.eff_alphas, k);
    PyObject *ec_t = make_double_tuple(meta.emp_coverage, k);
    if (!lo_t || !hi_t || !q_t || !ea_t || !ec_t) {
        Py_XDECREF(lo_t); Py_XDECREF(hi_t); Py_XDECREF(q_t); Py_XDECREF(ea_t); Py_XDECREF(ec_t);
        return NULL;
    }
    ConformalPredObj *o = PyObject_New(ConformalPredObj, &ConformalPredType);
    if (o == NULL) {
        Py_DECREF(lo_t); Py_DECREF(hi_t); Py_DECREF(q_t); Py_DECREF(ea_t); Py_DECREF(ec_t);
        return NULL;
    }
    o->point_pred = pred;
    o->scale = scale;
    Py_INCREF(self->alphas_tuple);
    o->alphas = self->alphas_tuple;
    o->lows = lo_t;
    o->highs = hi_t;
    o->quantiles = q_t;
    o->eff_alphas = ea_t;
    o->emp_coverage = ec_t;
    return (PyObject *)o;
}

static PyObject *MultiRateConformal_update(MultiRateConformalObj *self, PyObject *arg) {
    double score;
    if (fast_as_double(arg, &score) < 0) return NULL;
    eproc_conformal_out_t meta;
    int rc = eproc_conformal_update(&self->state, score, &meta);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "MultiRateConformal.update");
        return NULL;
    }
    return new_conformal_meta(&meta, self->alphas_tuple);
}

static PyObject *MultiRateConformal_update_batch(MultiRateConformalObj *self, PyObject *arg) {
    double_buf_t b;
    if (acquire_double_buf(arg, &b) < 0) return NULL;
    eproc_conformal_out_t meta;
    int rc = EPROC_OK;
    if (b.len == 0) {
        double lows[EPROC_MAX_RATES], highs[EPROC_MAX_RATES];
        eproc_conformal_predict(&self->state, 0.0, 1.0, lows, highs, &meta);
    } else if (b.len >= 64) {
        Py_BEGIN_ALLOW_THREADS
        rc = eproc_conformal_update_batch(&self->state, b.ptr, (size_t)b.len, &meta);
        Py_END_ALLOW_THREADS
    } else {
        rc = eproc_conformal_update_batch(&self->state, b.ptr, (size_t)b.len, &meta);
    }
    release_double_buf(&b);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "MultiRateConformal.update_batch");
        return NULL;
    }
    return new_conformal_meta(&meta, self->alphas_tuple);
}

static PyObject *MRC_get_alphas(MultiRateConformalObj *s, void *c) { Py_INCREF(s->alphas_tuple); return s->alphas_tuple; }
static PyObject *MRC_get_n_rates(MultiRateConformalObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.n_rates); }
static PyObject *MRC_get_n_updates(MultiRateConformalObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_updates); }
static PyObject *MRC_get_quantiles(MultiRateConformalObj *s, void *c) { return make_double_tuple(s->state.current_q, s->state.n_rates); }

static PyMethodDef MultiRateConformal_methods[] = {
    {"predict", EPROC_PYCFUNC(MultiRateConformal_predict), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"update", (PyCFunction)MultiRateConformal_update, METH_O, NULL},
    {"update_batch", (PyCFunction)MultiRateConformal_update_batch, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef MultiRateConformal_getset[] = {
    {"alphas", (getter)MRC_get_alphas, NULL, NULL, NULL},
    {"n_rates", (getter)MRC_get_n_rates, NULL, NULL, NULL},
    {"n_updates", (getter)MRC_get_n_updates, NULL, NULL, NULL},
    {"quantiles", (getter)MRC_get_quantiles, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject MultiRateConformalType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.conformal.MultiRateConformal",
    .tp_basicsize = sizeof(MultiRateConformalObj),
    .tp_dealloc = (destructor)MultiRateConformal_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)MultiRateConformal_init,
    .tp_methods = MultiRateConformal_methods,
    .tp_getset = MultiRateConformal_getset,
};

/* --- DelayedConformal --- */

typedef struct {
    PyObject_HEAD
    eproc_delayed_conformal_t state;
    PyObject *alphas_tuple;
} DelayedConformalObj;

static void DelayedConformal_dealloc(DelayedConformalObj *self) {
    Py_XDECREF(self->alphas_tuple);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int DelayedConformal_init(DelayedConformalObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"alphas", "window_size", "pid_ki", NULL};
    PyObject *alphas_obj = NULL;
    long window_size = 128;
    double pid_ki = 0.05;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Old", kwlist, &alphas_obj, &window_size, &pid_ki)) {
        return -1;
    }
    double alphas[EPROC_MAX_RATES] = {0.20, 0.10, 0.05, 0.0, 0.0};
    uint32_t n_rates = 3;
    if (alphas_obj != NULL) {
        if (prepare_alphas(alphas_obj, alphas, &n_rates) < 0) return -1;
    }
    if (window_size < 0) {
        PyErr_SetString(PyExc_ValueError, "DelayedConformal.__init__: window_size must be positive");
        return -1;
    }
    int rc = eproc_delayed_conformal_init(&self->state, alphas, n_rates, (uint32_t)window_size, pid_ki);
    if (set_eproc_error(rc, "DelayedConformal.__init__") < 0) return -1;

    Py_CLEAR(self->alphas_tuple);
    self->alphas_tuple = make_double_tuple(self->state.core.target_alphas, self->state.core.n_rates);
    return (self->alphas_tuple == NULL) ? -1 : 0;
}

static PyObject *DelayedConformal_issue(DelayedConformalObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"pred", "scale", NULL};
    double pred = 0.0, scale = 1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|d", kwlist, &pred, &scale)) return NULL;
    uint64_t ticket = 0;
    double lows[EPROC_MAX_RATES], highs[EPROC_MAX_RATES];
    int rc = eproc_delayed_conformal_issue(&self->state, pred, scale, &ticket, lows, highs);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "DelayedConformal.issue");
        return NULL;
    }
    uint32_t k = self->state.core.n_rates;
    PyObject *ivs = PyList_New((Py_ssize_t)k);
    if (ivs == NULL) return NULL;
    for (uint32_t i = 0; i < k; ++i) {
        PyObject *pair = Py_BuildValue("(dd)", lows[i], highs[i]);
        if (pair == NULL) { Py_DECREF(ivs); return NULL; }
        PyList_SET_ITEM(ivs, (Py_ssize_t)i, pair);
    }
    PyObject *tid = PyLong_FromUnsignedLongLong((unsigned long long)ticket);
    if (tid == NULL) { Py_DECREF(ivs); return NULL; }
    PyObject *ret = PyTuple_New(2);
    if (ret == NULL) { Py_DECREF(tid); Py_DECREF(ivs); return NULL; }
    PyTuple_SET_ITEM(ret, 0, tid);
    PyTuple_SET_ITEM(ret, 1, ivs);
    return ret;
}

static PyObject *DelayedConformal_reveal(DelayedConformalObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"ticket_id", "y_true", NULL};
    PyObject *tid_obj = NULL;
    double y_true = 0.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od", kwlist, &tid_obj, &y_true)) return NULL;
    unsigned long long tid = PyLong_AsUnsignedLongLong(tid_obj);
    if (tid == (unsigned long long)-1 && PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "ticket_id must be a non-negative uint64");
        return NULL;
    }
    eproc_conformal_out_t meta;
    int rc = eproc_delayed_conformal_reveal(&self->state, (uint64_t)tid, y_true, &meta);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "DelayedConformal.reveal");
        return NULL;
    }
    return new_conformal_meta(&meta, self->alphas_tuple);
}

static PyObject *DC_get_pending(DelayedConformalObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.pending_count); }
static PyObject *DC_get_alphas(DelayedConformalObj *s, void *c) { Py_INCREF(s->alphas_tuple); return s->alphas_tuple; }
static PyObject *DC_get_n_rates(DelayedConformalObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.core.n_rates); }
static PyObject *DC_get_n_updates(DelayedConformalObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.core.n_updates); }

static PyMethodDef DelayedConformal_methods[] = {
    {"issue", EPROC_PYCFUNC(DelayedConformal_issue), METH_VARARGS | METH_KEYWORDS, NULL},
    {"reveal", EPROC_PYCFUNC(DelayedConformal_reveal), METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef DelayedConformal_getset[] = {
    {"pending_count", (getter)DC_get_pending, NULL, NULL, NULL},
    {"alphas", (getter)DC_get_alphas, NULL, NULL, NULL},
    {"n_rates", (getter)DC_get_n_rates, NULL, NULL, NULL},
    {"n_updates", (getter)DC_get_n_updates, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject DelayedConformalType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.conformal.DelayedConformal",
    .tp_basicsize = sizeof(DelayedConformalObj),
    .tp_dealloc = (destructor)DelayedConformal_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)DelayedConformal_init,
    .tp_methods = DelayedConformal_methods,
    .tp_getset = DelayedConformal_getset,
};

/* --- BudgetConformal --- */

typedef struct {
    PyObject_HEAD
    eproc_budget_conformal_t state;
} BudgetConformalObj;

static int BudgetConformal_init(BudgetConformalObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"alpha", "window_size", "censor_penalty", NULL};
    double alpha = 0.10, censor_penalty = 1.25;
    long window_size = 128;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dld", kwlist, &alpha, &window_size, &censor_penalty)) {
        return -1;
    }
    if (window_size < 0) {
        PyErr_SetString(PyExc_ValueError, "BudgetConformal.__init__: window_size must be positive");
        return -1;
    }
    int rc = eproc_budget_conformal_init(&self->state, alpha, (uint32_t)window_size, censor_penalty);
    return set_eproc_error(rc, "BudgetConformal.__init__");
}

static PyObject *BudgetConformal_bound(BudgetConformalObj *self, PyObject *arg) {
    double base_est;
    if (fast_as_double(arg, &base_est) < 0) return NULL;
    return PyFloat_FromDouble(eproc_budget_conformal_bound(&self->state, base_est));
}

static PyObject *BudgetConformal_update(BudgetConformalObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"base_estimate", "observed_cost", "censored", NULL};
    double base_est = 0.0, obs_cost = 0.0;
    int censored = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd|p", kwlist, &base_est, &obs_cost, &censored)) {
        return NULL;
    }
    int rc = eproc_budget_conformal_update(&self->state, base_est, obs_cost, censored ? 1 : 0);
    if (set_eproc_error(rc, "BudgetConformal.update") < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *BC_get_alpha(BudgetConformalObj *s, void *c) { return PyFloat_FromDouble(s->state.core.target_alphas[0]); }
static PyObject *BC_get_penalty(BudgetConformalObj *s, void *c) { return PyFloat_FromDouble(s->state.censor_penalty); }
static PyObject *BC_get_completed(BudgetConformalObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_completed); }
static PyObject *BC_get_censored(BudgetConformalObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_censored); }

static PyMethodDef BudgetConformal_methods[] = {
    {"bound", (PyCFunction)BudgetConformal_bound, METH_O, NULL},
    {"update", EPROC_PYCFUNC(BudgetConformal_update), METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef BudgetConformal_getset[] = {
    {"alpha", (getter)BC_get_alpha, NULL, NULL, NULL},
    {"censor_penalty", (getter)BC_get_penalty, NULL, NULL, NULL},
    {"n_completed", (getter)BC_get_completed, NULL, NULL, NULL},
    {"n_censored", (getter)BC_get_censored, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject BudgetConformalType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.conformal.BudgetConformal",
    .tp_basicsize = sizeof(BudgetConformalObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)BudgetConformal_init,
    .tp_methods = BudgetConformal_methods,
    .tp_getset = BudgetConformal_getset,
};

/* --- ConformalMonoid --- */

typedef struct {
    PyObject_HEAD
    eproc_cmonoid_t state;
} ConformalMonoidObj;

static PyTypeObject ConformalMonoidType;
EPROC_DEFINE_FREELIST(cmonoid, ConformalMonoidObj, ConformalMonoidType)

static PyObject *ConformalMonoid_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    if (type == &ConformalMonoidType) {
        return (PyObject *)alloc_cmonoid();
    }
    return type->tp_alloc(type, 0);
}

static void ConformalMonoid_dealloc(ConformalMonoidObj *self) {
    if (Py_TYPE(self) == &ConformalMonoidType) {
        free_cmonoid(self);
    } else {
        Py_TYPE(self)->tp_free((PyObject *)self);
    }
}

static int ConformalMonoid_init(ConformalMonoidObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"scale_hint", NULL};
    double scale_hint = 1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &scale_hint)) {
        return -1;
    }
    eproc_cmonoid_init(&self->state, scale_hint);
    return 0;
}

static PyObject *ConformalMonoid_add(ConformalMonoidObj *self, PyObject *arg) {
    double score;
    if (fast_as_double(arg, &score) < 0) return NULL;
    eproc_cmonoid_add(&self->state, score);
    Py_RETURN_NONE;
}

static PyObject *ConformalMonoid_add_batch(ConformalMonoidObj *self, PyObject *arg) {
    double_buf_t b;
    if (acquire_double_buf(arg, &b) < 0) return NULL;
    if (b.len >= 256) {
        Py_BEGIN_ALLOW_THREADS
        eproc_cmonoid_add_batch(&self->state, b.ptr, (size_t)b.len);
        Py_END_ALLOW_THREADS
    } else if (b.len > 0) {
        eproc_cmonoid_add_batch(&self->state, b.ptr, (size_t)b.len);
    }
    release_double_buf(&b);
    Py_RETURN_NONE;
}

static PyObject *ConformalMonoid_merge(ConformalMonoidObj *self, PyObject *other_obj) {
    if (!PyObject_TypeCheck(other_obj, &ConformalMonoidType)) {
        PyErr_Format(PyExc_TypeError, "Expected ConformalMonoid, got %s", Py_TYPE(other_obj)->tp_name);
        return NULL;
    }
    ConformalMonoidObj *other = (ConformalMonoidObj *)other_obj;
    /* W08: reject incompatible scale_hint when both shards are non-empty */
    if (self->state.count > 0ULL && other->state.count > 0ULL) {
        double s1 = self->state.scale_hint;
        double s2 = other->state.scale_hint;
        double m = fmax(1.0, fmax(fabs(s1), fabs(s2)));
        if (fabs(s1 - s2) > 1e-12 * m) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "ConformalMonoid.merge: incompatible scale_hint (%.12g vs %.12g)", s1, s2);
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
    }
    ConformalMonoidObj *out = alloc_cmonoid();
    if (out == NULL) return NULL;
    eproc_cmonoid_merge(&self->state, &other->state, &out->state);
    return (PyObject *)out;
}

static PyObject *ConformalMonoid_subtract(ConformalMonoidObj *self, PyObject *local_obj) {
    if (!PyObject_TypeCheck(local_obj, &ConformalMonoidType)) {
        PyErr_Format(PyExc_TypeError, "Expected ConformalMonoid, got %s", Py_TYPE(local_obj)->tp_name);
        return NULL;
    }
    ConformalMonoidObj *local = (ConformalMonoidObj *)local_obj;
    ConformalMonoidObj *out = alloc_cmonoid();
    if (out == NULL) return NULL;
    int rc = eproc_cmonoid_subtract(&self->state, &local->state, &out->state);
    if (rc != EPROC_OK) {
        Py_DECREF(out);
        set_eproc_error(rc, "ConformalMonoid.subtract");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *ConformalMonoid_nb_add(PyObject *a, PyObject *b) {
    if (!PyObject_TypeCheck(a, &ConformalMonoidType) || !PyObject_TypeCheck(b, &ConformalMonoidType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return ConformalMonoid_merge((ConformalMonoidObj *)a, b);
}

static PyObject *ConformalMonoid_nb_sub(PyObject *a, PyObject *b) {
    if (!PyObject_TypeCheck(a, &ConformalMonoidType) || !PyObject_TypeCheck(b, &ConformalMonoidType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return ConformalMonoid_subtract((ConformalMonoidObj *)a, b);
}

static PyNumberMethods ConformalMonoid_as_number = {
    .nb_add = ConformalMonoid_nb_add,
    .nb_subtract = ConformalMonoid_nb_sub,
};

static PyObject *ConformalMonoid_quantile(ConformalMonoidObj *self, PyObject *arg) {
    double q;
    if (fast_as_double(arg, &q) < 0) return NULL;
    if (!isfinite(q) || q < 0.0 || q > 1.0) {
        PyErr_SetString(PyExc_ValueError, "Quantile probability q must be in [0, 1]");
        return NULL;
    }
    return PyFloat_FromDouble(eproc_cmonoid_quantile(&self->state, q));
}

static PyObject *ConformalMonoid_to_bytes(ConformalMonoidObj *self, PyObject *Py_UNUSED(ignored)) {
    size_t sz = eproc_cmonoid_wire_size();
    PyObject *b = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)sz);
    if (b == NULL) return NULL;
    int rc = eproc_cmonoid_serialize(&self->state, (uint8_t *)PyBytes_AS_STRING(b), sz);
    if (rc != EPROC_OK) {
        Py_DECREF(b);
        set_eproc_error(rc, "ConformalMonoid.to_bytes");
        return NULL;
    }
    return b;
}

static PyObject *ConformalMonoid_from_bytes(PyObject *cls, PyObject *arg) {
    Py_buffer view;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0) return NULL;
    size_t expected = eproc_cmonoid_wire_size();
    /* W11: require exact wire_size bytes, raising ValueError on truncated or trailing garbage */
    if ((size_t)view.len != expected) {
        Py_ssize_t got = view.len;
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError,
                     "ConformalMonoid.from_bytes: expected exactly %zu bytes, got %zd",
                     expected, got);
        return NULL;
    }
    ConformalMonoidObj *out = alloc_cmonoid();
    if (out == NULL) {
        PyBuffer_Release(&view);
        return NULL;
    }
    int rc = eproc_cmonoid_deserialize(&out->state, (const uint8_t *)view.buf, (size_t)view.len);
    PyBuffer_Release(&view);
    if (rc != EPROC_OK) {
        Py_DECREF(out);
        set_eproc_error(rc, "ConformalMonoid.from_bytes");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *CM_get_count(ConformalMonoidObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.count); }
static PyObject *CM_get_mean(ConformalMonoidObj *s, void *c) {
    if (s->state.count == 0ULL) return PyFloat_FromDouble(0.0);
    double sum = s->state.sum;
    if (!isfinite(sum)) return PyFloat_FromDouble(sum);
    return PyFloat_FromDouble((sum + s->state.sum_comp) / (double)s->state.count);
}
static PyObject *CM_get_min_val(ConformalMonoidObj *s, void *c) { return PyFloat_FromDouble(s->state.min_val); }
static PyObject *CM_get_max_val(ConformalMonoidObj *s, void *c) { return PyFloat_FromDouble(s->state.max_val); }
static PyObject *CM_get_scale_hint(ConformalMonoidObj *s, void *c) { return PyFloat_FromDouble(s->state.scale_hint); }
static PyObject *CM_get_bin_counts(ConformalMonoidObj *s, void *c) {
    PyObject *tup = PyTuple_New(EPROC_CMONOID_BINS);
    if (tup == NULL) return NULL;
    for (int i = 0; i < EPROC_CMONOID_BINS; ++i) {
        PyObject *v = PyLong_FromUnsignedLongLong(s->state.bin_counts[i]);
        if (v == NULL) { Py_DECREF(tup); return NULL; }
        PyTuple_SET_ITEM(tup, i, v);
    }
    return tup;
}

static PyMethodDef ConformalMonoid_methods[] = {
    {"add", (PyCFunction)ConformalMonoid_add, METH_O, NULL},
    {"add_batch", (PyCFunction)ConformalMonoid_add_batch, METH_O, NULL},
    {"merge", (PyCFunction)ConformalMonoid_merge, METH_O, NULL},
    {"subtract", (PyCFunction)ConformalMonoid_subtract, METH_O, NULL},
    {"quantile", (PyCFunction)ConformalMonoid_quantile, METH_O, NULL},
    {"to_bytes", (PyCFunction)ConformalMonoid_to_bytes, METH_NOARGS, NULL},
    {"from_bytes", (PyCFunction)ConformalMonoid_from_bytes, METH_CLASS | METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef ConformalMonoid_getset[] = {
    {"count", (getter)CM_get_count, NULL, NULL, NULL},
    {"mean", (getter)CM_get_mean, NULL, NULL, NULL},
    {"min_val", (getter)CM_get_min_val, NULL, NULL, NULL},
    {"max_val", (getter)CM_get_max_val, NULL, NULL, NULL},
    {"scale_hint", (getter)CM_get_scale_hint, NULL, NULL, NULL},
    {"bin_counts", (getter)CM_get_bin_counts, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject ConformalMonoidType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.conformal.ConformalMonoid",
    .tp_basicsize = sizeof(ConformalMonoidObj),
    .tp_dealloc = (destructor)ConformalMonoid_dealloc,
    .tp_as_number = &ConformalMonoid_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ConformalMonoid_new,
    .tp_init = (initproc)ConformalMonoid_init,
    .tp_methods = ConformalMonoid_methods,
    .tp_getset = ConformalMonoid_getset,
};

/* ============================================================================
 * 4. FUSION WRAPPERS — SafeFuser & EchoPool
 * ============================================================================ */

/* --- SafeFuser --- */

typedef struct {
    PyObject_HEAD
    eproc_boa_t state;
    uint32_t n_experts;
} SafeFuserObj;

static int SafeFuser_init(SafeFuserObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"n_experts", "anchor_idx", "alpha_deg", NULL};
    long n_experts = 0, anchor_idx = 0;
    double alpha_deg = 0.05;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|ld", kwlist, &n_experts, &anchor_idx, &alpha_deg)) {
        return -1;
    }
    if (n_experts <= 0 || n_experts > EPROC_BOA_MAX_EXPERTS) {
        PyErr_Format(PyExc_ValueError, "SafeFuser.__init__: n_experts out of range [1, %d]", EPROC_BOA_MAX_EXPERTS);
        return -1;
    }
    if (anchor_idx < 0 || anchor_idx >= n_experts) {
        PyErr_Format(PyExc_ValueError, "SafeFuser.__init__: anchor_idx out of range [0, %ld)", n_experts);
        return -1;
    }
    int rc = eproc_boa_init(&self->state, (uint32_t)n_experts, (uint32_t)anchor_idx, alpha_deg);
    if (set_eproc_error(rc, "SafeFuser.__init__") < 0) return -1;
    self->n_experts = self->state.n_experts;
    return 0;
}

static PyObject *SafeFuser_predict(SafeFuserObj *self, PyObject *arg) {
    double buf[EPROC_BOA_MAX_EXPERTS];
    if (extract_fixed_doubles(arg, buf, (Py_ssize_t)self->n_experts, "SafeFuser.predict") < 0) {
        return NULL;
    }
    return PyFloat_FromDouble(eproc_boa_predict(&self->state, buf));
}

static PyObject *SafeFuser_update(SafeFuserObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *preds_obj = NULL;
    double y_true = 0.0;
    if (kwnames == NULL && nargs == 2) {
        preds_obj = args[0];
        if (fast_as_double(args[1], &y_true) < 0) return NULL;
    } else {
        static char *kwlist[] = {"preds", "y_true", NULL};
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) { Py_INCREF(args[i]); PyTuple_SET_ITEM(tup, i, args[i]); }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "Od", kwlist, &preds_obj, &y_true);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
    }

    double buf[EPROC_BOA_MAX_EXPERTS];
    if (extract_fixed_doubles(preds_obj, buf, (Py_ssize_t)self->n_experts, "SafeFuser.update") < 0) {
        return NULL;
    }
    eproc_boa_step_t step;
    int rc = eproc_boa_update(&self->state, buf, y_true, &step);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "SafeFuser.update");
        return NULL;
    }
    SafeFuserStepObj *o = alloc_sfstep();
    if (o == NULL) return NULL;
    o->fused_pred = step.fused_pred;
    o->anchor_pred = step.anchor_pred;
    o->fallback_active = step.fallback_active ? 1 : 0;
    o->n_quarantined = step.n_quarantined;
    o->sup_log_e = step.sup_log_e;
    o->deg_log_e = step.deg_log_e;
    o->n_updates = (unsigned long long)step.n_updates;
    return (PyObject *)o;
}

/* W02: 100% thread-safe stack-local buffer + GIL released during seqlock spin */
static PyObject *SafeFuser_snapshot(SafeFuserObj *self, PyObject *Py_UNUSED(ignored)) {
    double buf[EPROC_BOA_MAX_EXPERTS];
    int32_t fb = 0;
    uint32_t k = self->n_experts;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = eproc_boa_snapshot_weights(&self->state, buf, k, &fb);
    Py_END_ALLOW_THREADS
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "SafeFuser.snapshot");
        return NULL;
    }
    PyObject *lst = PyList_New((Py_ssize_t)k);
    if (lst == NULL) return NULL;
    for (uint32_t i = 0; i < k; ++i) {
        PyObject *f = PyFloat_FromDouble(buf[i]);
        if (f == NULL) { Py_DECREF(lst); return NULL; }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, f);
    }
    PyObject *fb_obj = PyBool_FromLong(fb);
    PyObject *ret = PyTuple_New(2);
    if (ret == NULL) { Py_DECREF(lst); Py_DECREF(fb_obj); return NULL; }
    PyTuple_SET_ITEM(ret, 0, lst);
    PyTuple_SET_ITEM(ret, 1, fb_obj);
    return ret;
}

static PyObject *SafeFuser_weights(SafeFuserObj *self, PyObject *Py_UNUSED(ignored)) {
    double buf[EPROC_BOA_MAX_EXPERTS];
    int32_t fb = 0;
    uint32_t k = self->n_experts;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = eproc_boa_snapshot_weights(&self->state, buf, k, &fb);
    Py_END_ALLOW_THREADS
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "SafeFuser.weights");
        return NULL;
    }
    PyObject *lst = PyList_New((Py_ssize_t)k);
    if (lst == NULL) return NULL;
    for (uint32_t i = 0; i < k; ++i) {
        PyObject *f = PyFloat_FromDouble(buf[i]);
        if (f == NULL) { Py_DECREF(lst); return NULL; }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, f);
    }
    return lst;
}

static PyObject *SF_get_n_experts(SafeFuserObj *s, void *c) { return PyLong_FromUnsignedLong(s->n_experts); }
static PyObject *SF_get_anchor_idx(SafeFuserObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.anchor_idx); }
static PyObject *SF_get_fallback(SafeFuserObj *s, void *c) {
    double buf[EPROC_BOA_MAX_EXPERTS];
    int32_t fb = 0;
    Py_BEGIN_ALLOW_THREADS
    (void)eproc_boa_snapshot_weights(&s->state, buf, s->n_experts, &fb);
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(fb);
}
static PyObject *SF_get_n_updates(SafeFuserObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_updates); }
static PyObject *SF_get_poison(SafeFuserObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_poison_clipped); }
static PyObject *SF_get_trips(SafeFuserObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_fallback_trips); }
static PyObject *SF_get_quarantined(SafeFuserObj *s, void *c) {
    PyObject *lst = PyList_New((Py_ssize_t)s->n_experts);
    if (lst == NULL) return NULL;
    for (uint32_t i = 0; i < s->n_experts; ++i) {
        PyList_SET_ITEM(lst, (Py_ssize_t)i, PyBool_FromLong(s->state.quarantined[i]));
    }
    return lst;
}

static PyMethodDef SafeFuser_methods[] = {
    {"predict", (PyCFunction)SafeFuser_predict, METH_O, NULL},
    {"update", EPROC_PYCFUNC(SafeFuser_update), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"snapshot", (PyCFunction)SafeFuser_snapshot, METH_NOARGS, NULL},
    {"weights", (PyCFunction)SafeFuser_weights, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef SafeFuser_getset[] = {
    {"n_experts", (getter)SF_get_n_experts, NULL, NULL, NULL},
    {"anchor_idx", (getter)SF_get_anchor_idx, NULL, NULL, NULL},
    {"fallback_active", (getter)SF_get_fallback, NULL, NULL, NULL},
    {"n_updates", (getter)SF_get_n_updates, NULL, NULL, NULL},
    {"n_poison_clipped", (getter)SF_get_poison, NULL, NULL, NULL},
    {"n_fallback_trips", (getter)SF_get_trips, NULL, NULL, NULL},
    {"quarantined", (getter)SF_get_quarantined, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject SafeFuserType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.fusion.SafeFuser",
    .tp_basicsize = sizeof(SafeFuserObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SafeFuser_init,
    .tp_methods = SafeFuser_methods,
    .tp_getset = SafeFuser_getset,
};

/* --- EchoPool --- */

typedef struct {
    PyObject_HEAD
    eproc_echopool_t state;
    uint32_t n_sources;
    uint32_t n_claims;
} EchoPoolObj;

static int EchoPool_init(EchoPoolObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"n_sources", "n_claims", "prior_prob", NULL};
    long n_sources = 0, n_claims = 64;
    double prior_prob = 0.5;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|ld", kwlist, &n_sources, &n_claims, &prior_prob)) {
        return -1;
    }
    if (n_sources <= 0 || n_sources > EPROC_ECHO_MAX_SOURCES ||
        n_claims <= 0 || n_claims > EPROC_ECHO_MAX_CLAIMS) {
        PyErr_SetString(PyExc_ValueError, "EchoPool.__init__: n_sources or n_claims out of range");
        return -1;
    }
    int rc = eproc_echopool_init(&self->state, (uint32_t)n_sources, (uint32_t)n_claims, prior_prob);
    if (set_eproc_error(rc, "EchoPool.__init__") < 0) return -1;
    self->n_sources = self->state.n_sources;
    self->n_claims = self->state.n_claims;
    return 0;
}

static PyObject *EchoPool_submit(EchoPoolObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    long long src = 0, clm = 0;
    PyObject *ev_obj = NULL;
    double llr = 0.0;
    if (kwnames == NULL && nargs == 4) {
        src = PyLong_AsLongLong(args[0]);
        if (src == -1 && PyErr_Occurred()) return NULL;
        clm = PyLong_AsLongLong(args[1]);
        if (clm == -1 && PyErr_Occurred()) return NULL;
        ev_obj = args[2];
        if (fast_as_double(args[3], &llr) < 0) return NULL;
    } else {
        static char *kwlist[] = {"source_id", "claim_id", "evidence_id", "llr", NULL};
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) { Py_INCREF(args[i]); PyTuple_SET_ITEM(tup, i, args[i]); }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "LLOd", kwlist, &src, &clm, &ev_obj, &llr);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
    }

    if (src < 0 || src >= (long long)self->n_sources) {
        PyErr_Format(PyExc_ValueError, "source_id %lld out of range [0, %u)", src, self->n_sources);
        return NULL;
    }
    if (clm < 0 || clm >= (long long)self->n_claims) {
        PyErr_Format(PyExc_ValueError, "claim_id %lld out of range [0, %u)", clm, self->n_claims);
        return NULL;
    }
    uint64_t ev_u64;
    if (normalize_evidence_id(ev_obj, &ev_u64) < 0) return NULL;

    eproc_echo_out_t out;
    int rc = eproc_echopool_submit(&self->state, (uint32_t)src, (uint32_t)clm, ev_u64, llr, &out);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "EchoPool.submit");
        return NULL;
    }
    EchoSubmitResultObj *o = alloc_echores();
    if (o == NULL) return NULL;
    o->accepted = out.accepted ? 1 : 0;
    o->posterior_prob = out.posterior_prob;
    o->log_odds = out.log_odds;
    o->source_reliability = out.source_reliability;
    o->unique_count = (unsigned long long)out.unique_count;
    return (PyObject *)o;
}

static PyObject *EchoPool_resolve(EchoPoolObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"claim_id", "outcome_positive", NULL};
    long long clm = 0;
    int pos = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Lp", kwlist, &clm, &pos)) return NULL;
    if (clm < 0 || clm >= (long long)self->n_claims) {
        PyErr_Format(PyExc_ValueError, "claim_id %lld out of range [0, %u)", clm, self->n_claims);
        return NULL;
    }
    int rc = eproc_echopool_resolve(&self->state, (uint32_t)clm, pos ? 1 : 0);
    if (set_eproc_error(rc, "EchoPool.resolve") < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *EchoPool_posterior(EchoPoolObj *self, PyObject *arg) {
    long long clm = PyLong_AsLongLong(arg);
    if (clm == -1 && PyErr_Occurred()) return NULL;
    if (clm < 0 || clm >= (long long)self->n_claims) {
        PyErr_Format(PyExc_ValueError, "claim_id %lld out of range [0, %u)", clm, self->n_claims);
        return NULL;
    }
    return PyFloat_FromDouble(eproc_echopool_posterior(&self->state, (uint32_t)clm));
}

static PyObject *EchoPool_source_reliability(EchoPoolObj *self, PyObject *arg) {
    long long src = PyLong_AsLongLong(arg);
    if (src == -1 && PyErr_Occurred()) return NULL;
    if (src < 0 || src >= (long long)self->n_sources) {
        PyErr_Format(PyExc_ValueError, "source_id %lld out of range [0, %u)", src, self->n_sources);
        return NULL;
    }
    return PyFloat_FromDouble(eproc_echopool_source_reliability(&self->state, (uint32_t)src));
}

static PyObject *EP_get_n_sources(EchoPoolObj *s, void *c) { return PyLong_FromUnsignedLong(s->n_sources); }
static PyObject *EP_get_n_claims(EchoPoolObj *s, void *c) { return PyLong_FromUnsignedLong(s->n_claims); }
static PyObject *EP_get_n_accepted(EchoPoolObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_accepted); }
static PyObject *EP_get_n_dropped(EchoPoolObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_echo_dropped); }
static PyObject *EP_get_n_full(EchoPoolObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_table_full); }

static PyMethodDef EchoPool_methods[] = {
    {"submit", EPROC_PYCFUNC(EchoPool_submit), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"resolve", EPROC_PYCFUNC(EchoPool_resolve), METH_VARARGS | METH_KEYWORDS, NULL},
    {"posterior", (PyCFunction)EchoPool_posterior, METH_O, NULL},
    {"source_reliability", (PyCFunction)EchoPool_source_reliability, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef EchoPool_getset[] = {
    {"n_sources", (getter)EP_get_n_sources, NULL, NULL, NULL},
    {"n_claims", (getter)EP_get_n_claims, NULL, NULL, NULL},
    {"n_accepted", (getter)EP_get_n_accepted, NULL, NULL, NULL},
    {"n_echo_dropped", (getter)EP_get_n_dropped, NULL, NULL, NULL},
    {"n_table_full", (getter)EP_get_n_full, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject EchoPoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.fusion.EchoPool",
    .tp_basicsize = sizeof(EchoPoolObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)EchoPool_init,
    .tp_methods = EchoPool_methods,
    .tp_getset = EchoPool_getset,
};

/* ============================================================================
 * 5. SUBSAMPLE WRAPPERS — VarOptSampler, OnlineRLSSampler, SiftVOISelector
 * ============================================================================ */

/* --- VarOptSampler --- */

typedef struct {
    PyObject_HEAD
    eproc_varopt_t state;
} VarOptObj;

static int VarOpt_init(VarOptObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"k", "seed", NULL};
    long k = 0;
    PyObject *seed_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|O", kwlist, &k, &seed_obj)) return -1;
    if (k <= 0 || k > EPROC_VAROPT_MAX_K) {
        PyErr_Format(PyExc_ValueError, "VarOptSampler.__init__: k out of range [1, %d]", EPROC_VAROPT_MAX_K);
        return -1;
    }
    uint64_t seed = 42ULL;
    if (seed_obj != NULL) {
        unsigned long long s = PyLong_AsUnsignedLongLongMask(seed_obj);
        if (s == (unsigned long long)-1 && PyErr_Occurred()) return -1;
        seed = (uint64_t)s;
    }
    int rc = eproc_varopt_init(&self->state, (uint32_t)k, seed);
    return set_eproc_error(rc, "VarOptSampler.__init__");
}

static PyObject *VarOpt_push(VarOptObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    unsigned long long item_id = 0;
    double weight = 0.0, value = 0.0;
    if (kwnames == NULL && nargs == 3) {
        item_id = PyLong_AsUnsignedLongLong(args[0]);
        if (item_id == (unsigned long long)-1 && PyErr_Occurred()) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "item_id must be a non-negative uint64");
            return NULL;
        }
        if (fast_as_double(args[1], &weight) < 0) return NULL;
        if (fast_as_double(args[2], &value) < 0) return NULL;
    } else {
        static char *kwlist[] = {"item_id", "weight", "value", NULL};
        PyObject *id_obj = NULL;
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) { Py_INCREF(args[i]); PyTuple_SET_ITEM(tup, i, args[i]); }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "Odd", kwlist, &id_obj, &weight, &value);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
        item_id = PyLong_AsUnsignedLongLong(id_obj);
        if (item_id == (unsigned long long)-1 && PyErr_Occurred()) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "item_id must be a non-negative uint64");
            return NULL;
        }
    }
    int rc = eproc_varopt_push(&self->state, (uint64_t)item_id, weight, value);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "VarOptSampler.push");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *VarOpt_estimate_sum(VarOptObj *self, PyObject *Py_UNUSED(ignored)) {
    return PyFloat_FromDouble(eproc_varopt_estimate_sum(&self->state));
}

static PyObject *VarOpt_estimate_mean(VarOptObj *self, PyObject *Py_UNUSED(ignored)) {
    return PyFloat_FromDouble(eproc_varopt_estimate_mean(&self->state));
}

static PyObject *VarOpt_items(VarOptObj *self, PyObject *Py_UNUSED(ignored)) {
    uint32_t n = self->state.count;
    PyObject *lst = PyList_New((Py_ssize_t)n);
    if (lst == NULL) return NULL;
    for (uint32_t i = 0; i < n; ++i) {
        VarOptItemObj *o = PyObject_New(VarOptItemObj, &VarOptItemType);
        if (o == NULL) { Py_DECREF(lst); return NULL; }
        o->item_id = (unsigned long long)self->state.items[i].item_id;
        o->orig_weight = self->state.items[i].orig_weight;
        o->adj_weight = self->state.items[i].adj_weight;
        o->value = self->state.items[i].value;
        PyList_SET_ITEM(lst, (Py_ssize_t)i, (PyObject *)o);
    }
    return lst;
}

static PyObject *VO_get_k(VarOptObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.k); }
static PyObject *VO_get_count(VarOptObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.count); }
static PyObject *VO_get_n_heavy(VarOptObj *s, void *c) { return PyLong_FromUnsignedLong(s->state.n_heavy); }
static PyObject *VO_get_tau(VarOptObj *s, void *c) { return PyFloat_FromDouble(s->state.tau); }
static PyObject *VO_get_total_weight(VarOptObj *s, void *c) {
    double tw = s->state.total_weight;
    if (!isfinite(tw)) return PyFloat_FromDouble(tw);
    return PyFloat_FromDouble(tw + s->state.total_weight_comp);
}
static PyObject *VO_get_n_seen(VarOptObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_seen); }

static PyMethodDef VarOpt_methods[] = {
    {"push", EPROC_PYCFUNC(VarOpt_push), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"estimate_sum", (PyCFunction)VarOpt_estimate_sum, METH_NOARGS, NULL},
    {"estimate_mean", (PyCFunction)VarOpt_estimate_mean, METH_NOARGS, NULL},
    {"items", (PyCFunction)VarOpt_items, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef VarOpt_getset[] = {
    {"k", (getter)VO_get_k, NULL, NULL, NULL},
    {"count", (getter)VO_get_count, NULL, NULL, NULL},
    {"n_heavy", (getter)VO_get_n_heavy, NULL, NULL, NULL},
    {"tau", (getter)VO_get_tau, NULL, NULL, NULL},
    {"total_weight", (getter)VO_get_total_weight, NULL, NULL, NULL},
    {"n_seen", (getter)VO_get_n_seen, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject VarOptType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.subsample.VarOptSampler",
    .tp_basicsize = sizeof(VarOptObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)VarOpt_init,
    .tp_methods = VarOpt_methods,
    .tp_getset = VarOpt_getset,
};

/* --- OnlineRLSSampler --- */

typedef struct {
    PyObject_HEAD
    eproc_rls_t state;
    uint32_t d;
} OnlineRLSObj;

static int OnlineRLS_init(OnlineRLSObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"d", "lambda_reg", "oversample", "min_prob", "seed", NULL};
    long d = 0;
    double lambda_reg = 1.0, oversample = 2.0, min_prob = 0.02;
    PyObject *seed_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|dddO", kwlist,
                                     &d, &lambda_reg, &oversample, &min_prob, &seed_obj)) {
        return -1;
    }
    if (d <= 0 || d > EPROC_RLS_MAX_D) {
        PyErr_Format(PyExc_ValueError, "OnlineRLSSampler.__init__: d out of range [1, %d]", EPROC_RLS_MAX_D);
        return -1;
    }
    uint64_t seed = 42ULL;
    if (seed_obj != NULL) {
        unsigned long long s = PyLong_AsUnsignedLongLongMask(seed_obj);
        if (s == (unsigned long long)-1 && PyErr_Occurred()) return -1;
        seed = (uint64_t)s;
    }
    int rc = eproc_rls_init(&self->state, (uint32_t)d, lambda_reg, oversample, min_prob, seed);
    if (set_eproc_error(rc, "OnlineRLSSampler.__init__") < 0) return -1;
    self->d = self->state.d;
    return 0;
}

static PyObject *OnlineRLS_step(OnlineRLSObj *self, PyObject *arg) {
    double buf[EPROC_RLS_MAX_D];
    if (extract_fixed_doubles(arg, buf, (Py_ssize_t)self->d, "OnlineRLSSampler.step") < 0) {
        return NULL;
    }
    eproc_rls_out_t out;
    int rc = eproc_rls_step(&self->state, buf, &out);
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "OnlineRLSSampler.step");
        return NULL;
    }
    RLSStepResultObj *o = alloc_rlsres();
    if (o == NULL) return NULL;
    o->keep = out.keep ? 1 : 0;
    o->leverage = out.leverage;
    o->prob = out.prob;
    o->ht_weight = out.ht_weight;
    return (PyObject *)o;
}

static PyObject *RLS_get_d(OnlineRLSObj *s, void *c) { return PyLong_FromUnsignedLong(s->d); }
static PyObject *RLS_get_n_seen(OnlineRLSObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_seen); }
static PyObject *RLS_get_n_kept(OnlineRLSObj *s, void *c) { return PyLong_FromUnsignedLongLong(s->state.n_kept); }
static PyObject *RLS_get_keep_rate(OnlineRLSObj *s, void *c) {
    return PyFloat_FromDouble(s->state.n_seen > 0 ? (double)s->state.n_kept / (double)s->state.n_seen : 0.0);
}

static PyMethodDef OnlineRLS_methods[] = {
    {"step", (PyCFunction)OnlineRLS_step, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef OnlineRLS_getset[] = {
    {"d", (getter)RLS_get_d, NULL, NULL, NULL},
    {"n_seen", (getter)RLS_get_n_seen, NULL, NULL, NULL},
    {"n_kept", (getter)RLS_get_n_kept, NULL, NULL, NULL},
    {"keep_rate", (getter)RLS_get_keep_rate, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject OnlineRLSType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.subsample.OnlineRLSSampler",
    .tp_basicsize = sizeof(OnlineRLSObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)OnlineRLS_init,
    .tp_methods = OnlineRLS_methods,
    .tp_getset = OnlineRLS_getset,
};

/* --- SiftVOISelector + read-only _state view for chan_prob compatibility --- */

typedef struct {
    PyObject_HEAD
    eproc_sift_t state;
    uint32_t n_sites;
    uint32_t per_site;
} SiftObj;

typedef struct {
    PyObject_HEAD
    PyObject *owner;
    const double *data;
    Py_ssize_t len;
} DoubleArrayViewObj;

static void DoubleArrayView_dealloc(DoubleArrayViewObj *self) {
    Py_XDECREF(self->owner);
    PyObject_Del(self);
}

static Py_ssize_t DoubleArrayView_len(DoubleArrayViewObj *self) {
    return self->len;
}

static PyObject *DoubleArrayView_item(DoubleArrayViewObj *self, Py_ssize_t idx) {
    if (idx < 0 || idx >= self->len) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }
    return PyFloat_FromDouble(self->data[idx]);
}

static PySequenceMethods DoubleArrayView_as_seq = {
    .sq_length = (lenfunc)DoubleArrayView_len,
    .sq_item = (ssizeargfunc)DoubleArrayView_item,
};

static PyTypeObject DoubleArrayViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc._native._DoubleArrayView",
    .tp_basicsize = sizeof(DoubleArrayViewObj),
    .tp_dealloc = (destructor)DoubleArrayView_dealloc,
    .tp_as_sequence = &DoubleArrayView_as_seq,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

static PyObject *new_double_view(PyObject *owner, const double *ptr, Py_ssize_t len) {
    DoubleArrayViewObj *v = PyObject_New(DoubleArrayViewObj, &DoubleArrayViewType);
    if (v == NULL) return NULL;
    Py_INCREF(owner);
    v->owner = owner;
    v->data = ptr;
    v->len = len;
    return (PyObject *)v;
}

typedef struct {
    PyObject_HEAD
    SiftObj *owner;
} SiftStateViewObj;

static void SiftStateView_dealloc(SiftStateViewObj *self) {
    Py_XDECREF(self->owner);
    PyObject_Del(self);
}

static PyObject *SiftStateView_chan_prob(SiftStateViewObj *s, void *c) {
    return new_double_view((PyObject *)s->owner, s->owner->state.chan_prob,
                           (Py_ssize_t)(EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE));
}
static PyObject *SiftStateView_site_log_prob(SiftStateViewObj *s, void *c) {
    return new_double_view((PyObject *)s->owner, s->owner->state.site_log_prob,
                           (Py_ssize_t)EPROC_SIFT_MAX_SITES);
}
static PyObject *SiftStateView_chan_noise_scale(SiftStateViewObj *s, void *c) {
    return new_double_view((PyObject *)s->owner, s->owner->state.chan_noise_scale,
                           (Py_ssize_t)(EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE));
}

static PyGetSetDef SiftStateView_getset[] = {
    {"chan_prob", (getter)SiftStateView_chan_prob, NULL, NULL, NULL},
    {"site_log_prob", (getter)SiftStateView_site_log_prob, NULL, NULL, NULL},
    {"chan_noise_scale", (getter)SiftStateView_chan_noise_scale, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject SiftStateViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc._native._SiftStateView",
    .tp_basicsize = sizeof(SiftStateViewObj),
    .tp_dealloc = (destructor)SiftStateView_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = SiftStateView_getset,
};

static int Sift_init(SiftObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"n_sites", "per_site", "signal_mean", "noise_sd", NULL};
    long n_sites = 0, per_site = 8;
    double signal_mean = 1.5, noise_sd = 1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|ldd", kwlist,
                                     &n_sites, &per_site, &signal_mean, &noise_sd)) {
        return -1;
    }
    if (n_sites <= 0 || n_sites > EPROC_SIFT_MAX_SITES ||
        per_site <= 0 || per_site > EPROC_SIFT_MAX_PER_SITE) {
        PyErr_SetString(PyExc_ValueError, "SiftVOISelector.__init__: n_sites or per_site out of range");
        return -1;
    }
    int rc = eproc_sift_init(&self->state, (uint32_t)n_sites, (uint32_t)per_site, signal_mean, noise_sd);
    if (set_eproc_error(rc, "SiftVOISelector.__init__") < 0) return -1;
    self->n_sites = self->state.n_sites;
    self->per_site = self->state.per_site;
    return 0;
}

static PyObject *Sift_set_noise_scale(SiftObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"ch", "scale", NULL};
    long long ch = 0;
    double scale = 1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Ld", kwlist, &ch, &scale)) return NULL;
    uint32_t n_ch = self->n_sites * self->per_site;
    if (ch < 0 || ch >= (long long)n_ch) {
        PyErr_Format(PyExc_ValueError, "channel index %lld out of range [0, %u)", ch, n_ch);
        return NULL;
    }
    int rc = eproc_sift_set_noise_scale(&self->state, (uint32_t)ch, scale);
    if (set_eproc_error(rc, "SiftVOISelector.set_noise_scale") < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *Sift_observe_channel(SiftObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    long long ch = 0;
    double obs_mean = 0.0;
    long long n_ticks = 1;
    if (kwnames == NULL && (nargs == 2 || nargs == 3)) {
        ch = PyLong_AsLongLong(args[0]);
        if (ch == -1 && PyErr_Occurred()) return NULL;
        if (fast_as_double(args[1], &obs_mean) < 0) return NULL;
        if (nargs == 3) {
            n_ticks = PyLong_AsLongLong(args[2]);
            if (n_ticks == -1 && PyErr_Occurred()) return NULL;
        }
    } else {
        static char *kwlist[] = {"ch", "obs_mean", "n_ticks", NULL};
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) { Py_INCREF(args[i]); PyTuple_SET_ITEM(tup, i, args[i]); }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "Ld|L", kwlist, &ch, &obs_mean, &n_ticks);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
    }
    uint32_t n_ch = self->n_sites * self->per_site;
    if (ch < 0 || ch >= (long long)n_ch) {
        PyErr_Format(PyExc_ValueError, "channel index %lld out of range [0, %u)", ch, n_ch);
        return NULL;
    }
    if (n_ticks < 0 || n_ticks > UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "n_ticks out of range");
        return NULL;
    }
    int rc = eproc_sift_observe_channel(&self->state, (uint32_t)ch, obs_mean, (uint32_t)n_ticks);
    if (set_eproc_error(rc, "SiftVOISelector.observe_channel") < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *Sift_observe_site(SiftObj *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    long long site = 0;
    double site_mean = 0.0;
    if (kwnames == NULL && nargs == 2) {
        site = PyLong_AsLongLong(args[0]);
        if (site == -1 && PyErr_Occurred()) return NULL;
        if (fast_as_double(args[1], &site_mean) < 0) return NULL;
    } else {
        static char *kwlist[] = {"site", "site_mean", NULL};
        PyObject *tup = PyTuple_New(nargs);
        if (tup == NULL) return NULL;
        for (Py_ssize_t i = 0; i < nargs; ++i) { Py_INCREF(args[i]); PyTuple_SET_ITEM(tup, i, args[i]); }
        PyObject *kw = NULL;
        if (kwnames != NULL) {
            kw = PyDict_New();
            if (kw == NULL) { Py_DECREF(tup); return NULL; }
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); ++i) {
                PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(tup, kw, "Ld", kwlist, &site, &site_mean);
        Py_DECREF(tup);
        Py_XDECREF(kw);
        if (!ok) return NULL;
    }
    if (site < 0 || site >= (long long)self->n_sites) {
        PyErr_Format(PyExc_ValueError, "site index %lld out of range [0, %u)", site, self->n_sites);
        return NULL;
    }
    int rc = eproc_sift_observe_site(&self->state, (uint32_t)site, site_mean);
    if (set_eproc_error(rc, "SiftVOISelector.observe_site") < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *Sift_select_next(SiftObj *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"cost_site", "cost_channel", "cost_ticks", "ticks_per_read", NULL};
    double cost_site = 1.0, cost_channel = 1.0, cost_ticks = 8.0;
    long long ticks_per_read = 8;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dddL", kwlist,
                                     &cost_site, &cost_channel, &cost_ticks, &ticks_per_read)) {
        return NULL;
    }
    if (ticks_per_read < 0 || ticks_per_read > UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "ticks_per_read out of range");
        return NULL;
    }
    eproc_sift_action_t act;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = eproc_sift_select_next(&self->state, cost_site, cost_channel, cost_ticks,
                                (uint32_t)ticks_per_read, &act);
    Py_END_ALLOW_THREADS
    if (rc != EPROC_OK) {
        set_eproc_error(rc, "SiftVOISelector.select_next");
        return NULL;
    }
    SiftActionObj *o = PyObject_New(SiftActionObj, &SiftActionType);
    if (o == NULL) return NULL;
    o->action_type = (int)act.action_type;
    if (act.action_type == 0) {
        o->action_name = g_str_site;
        Py_INCREF(o->action_name);
    } else if (act.action_type == 1) {
        o->action_name = g_str_channel;
        Py_INCREF(o->action_name);
    } else if (act.action_type == 2) {
        o->action_name = g_str_ticks;
        Py_INCREF(o->action_name);
    } else {
        o->action_name = PyUnicode_FromFormat("unknown(%d)", (int)act.action_type);
        if (o->action_name == NULL) { Py_DECREF(o); return NULL; }
    }
    o->target_idx = act.target_idx;
    o->evoi_per_cost = act.evoi_per_cost;
    return (PyObject *)o;
}

static PyObject *Sift_get_n_sites(SiftObj *s, void *c) { return PyLong_FromUnsignedLong(s->n_sites); }
static PyObject *Sift_get_per_site(SiftObj *s, void *c) { return PyLong_FromUnsignedLong(s->per_site); }
static PyObject *Sift_get_n_channels(SiftObj *s, void *c) { return PyLong_FromUnsignedLong(s->n_sites * s->per_site); }
static PyObject *Sift_get_state(SiftObj *s, void *c) {
    SiftStateViewObj *v = PyObject_New(SiftStateViewObj, &SiftStateViewType);
    if (v == NULL) return NULL;
    Py_INCREF(s);
    v->owner = s;
    return (PyObject *)v;
}

static PyMethodDef Sift_methods[] = {
    {"set_noise_scale", EPROC_PYCFUNC(Sift_set_noise_scale), METH_VARARGS | METH_KEYWORDS, NULL},
    {"observe_channel", EPROC_PYCFUNC(Sift_observe_channel), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"observe_site", EPROC_PYCFUNC(Sift_observe_site), METH_FASTCALL | METH_KEYWORDS, NULL},
    {"select_next", EPROC_PYCFUNC(Sift_select_next), METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef Sift_getset[] = {
    {"n_sites", (getter)Sift_get_n_sites, NULL, NULL, NULL},
    {"per_site", (getter)Sift_get_per_site, NULL, NULL, NULL},
    {"n_channels", (getter)Sift_get_n_channels, NULL, NULL, NULL},
    {"_state", (getter)Sift_get_state, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject SiftType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "eproc.subsample.SiftVOISelector",
    .tp_basicsize = sizeof(SiftObj),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Sift_init,
    .tp_methods = Sift_methods,
    .tp_getset = Sift_getset,
};

/* ============================================================================
 * 6. MODULE INITIALIZATION
 * ============================================================================ */

static PyObject *py_abi_version(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromUnsignedLong(eproc_abi_version());
}

static PyObject *py_simd_flags(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromUnsignedLong(eproc_simd_flags());
}

static PyMethodDef module_methods[] = {
    {"combine_evalues", EPROC_PYCFUNC(py_combine_evalues), METH_VARARGS | METH_KEYWORDS, NULL},
    {"abi_version", (PyCFunction)py_abi_version, METH_NOARGS, NULL},
    {"simd_flags", (PyCFunction)py_simd_flags, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "eproc._native",
    .m_doc = "Zero-dependency native CPython C-API accelerator for libeproc.",
    .m_size = -1,
    .m_methods = module_methods,
};

static int register_type(PyObject *mod, const char *name, PyTypeObject *tp) {
    if (PyType_Ready(tp) < 0) return -1;
    Py_INCREF(tp);
    if (PyModule_AddObject(mod, name, (PyObject *)tp) < 0) {
        Py_DECREF(tp);
        return -1;
    }
    return 0;
}

PyMODINIT_FUNC PyInit__native(void) {
    uint32_t abi = eproc_abi_version();
    if (abi != EPROC_ABI_VERSION) {
        PyErr_Format(PyExc_RuntimeError,
                     "libeproc ABI mismatch: expected %u, got %u",
                     (unsigned)EPROC_ABI_VERSION, (unsigned)abi);
        return NULL;
    }

    PyObject *mod = PyModule_Create(&native_module);
    if (mod == NULL) return NULL;

    g_str_site = PyUnicode_InternFromString("site");
    g_str_channel = PyUnicode_InternFromString("channel");
    g_str_ticks = PyUnicode_InternFromString("ticks");
    if (!g_str_site || !g_str_channel || !g_str_ticks) {
        Py_DECREF(mod);
        return NULL;
    }

    /* Ready internal view types */
    if (PyType_Ready(&DoubleArrayViewType) < 0 || PyType_Ready(&SiftStateViewType) < 0) {
        Py_DECREF(mod);
        return NULL;
    }

    /* Register result types */
    if (register_type(mod, "EProcessResult", &EProcessResultType) < 0 ||
        register_type(mod, "ConfidenceSequenceResult", &CSResultType) < 0 ||
        register_type(mod, "EDetectorResult", &EDetectorResultType) < 0 ||
        register_type(mod, "RaceStatus", &RaceStatusType) < 0 ||
        register_type(mod, "RaceArmState", &RaceArmStateType) < 0 ||
        register_type(mod, "ConformalMeta", &ConformalMetaType) < 0 ||
        register_type(mod, "ConformalPrediction", &ConformalPredType) < 0 ||
        register_type(mod, "SafeFuserStep", &SafeFuserStepType) < 0 ||
        register_type(mod, "EchoSubmitResult", &EchoSubmitResultType) < 0 ||
        register_type(mod, "VarOptItem", &VarOptItemType) < 0 ||
        register_type(mod, "RLSStepResult", &RLSStepResultType) < 0 ||
        register_type(mod, "SiftAction", &SiftActionType) < 0) {
        Py_DECREF(mod);
        return NULL;
    }

    /* Register engine types */
    if (register_type(mod, "EProcess", &EProcessType) < 0 ||
        register_type(mod, "ConfidenceSequence", &CSType) < 0 ||
        register_type(mod, "EDetector", &EDetectorType) < 0 ||
        register_type(mod, "AgentRace", &AgentRaceType) < 0 ||
        register_type(mod, "MultiRateConformal", &MultiRateConformalType) < 0 ||
        register_type(mod, "DelayedConformal", &DelayedConformalType) < 0 ||
        register_type(mod, "BudgetConformal", &BudgetConformalType) < 0 ||
        register_type(mod, "ConformalMonoid", &ConformalMonoidType) < 0 ||
        register_type(mod, "SafeFuser", &SafeFuserType) < 0 ||
        register_type(mod, "EchoPool", &EchoPoolType) < 0 ||
        register_type(mod, "VarOptSampler", &VarOptType) < 0 ||
        register_type(mod, "OnlineRLSSampler", &OnlineRLSType) < 0 ||
        register_type(mod, "SiftVOISelector", &SiftType) < 0) {
        Py_DECREF(mod);
        return NULL;
    }

    /* Attach class/instance wire_size = 640 on ConformalMonoid */
    PyObject *ws = PyLong_FromSize_t(eproc_cmonoid_wire_size());
    if (ws == NULL || PyDict_SetItemString(ConformalMonoidType.tp_dict, "wire_size", ws) < 0) {
        Py_XDECREF(ws);
        Py_DECREF(mod);
        return NULL;
    }
    Py_DECREF(ws);
    PyType_Modified(&ConformalMonoidType);

    return mod;
}
