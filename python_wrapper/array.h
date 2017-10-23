#ifndef VOXEL_MAP_ARRAY_H
#define VOXEL_MAP_ARRAY_H

#include <numpy/arrayobject.h>
#include <voxel_map/strided.h>

namespace
{

template<typename T>
class ConstArray
{
public:
    typedef voxel_map::Strided<const T *> Iter;

    // TODO: Enforce template types.
    ConstArray(PyObject *arr):
        carr_(reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(arr, NPY_DOUBLE,
                                                                 NPY_ARRAY_C_CONTIGUOUS
                                                                 | NPY_ARRAY_ALIGNED)))
    {
        assert(check());
    }
    ~ConstArray()
    {
        Py_XDECREF(carr_);
    }

    const PyArrayObject * array(bool transferOwnership = false) const
    {
        assert(carr_ != NULL);
        if (transferOwnership)
            Py_XINCREF(carr_);
        return carr_;
    }

    size_t ndims() const { return PyArray_NDIM(array()); }
    // PyArray_SIZE cannot handle const pointer.
//    size_t numel() const { return PyArray_SIZE(array()); }
    size_t numel() const
    {
        if (ndims() == 0)
            return 0;
        size_t s = 1;
        for (size_t i = 0; i < ndims(); ++i)
            s *= size(i);
        return s;
    }
    size_t size(size_t dim) const
    {
        assert(dim < ndims());
        return PyArray_DIM(array(), dim);
    }
    size_t stride(size_t dim) const
    {
        assert(dim < ndims());
        assert(PyArray_STRIDE(array(), dim) % sizeof(double) == 0);
        return PyArray_STRIDE(array(), dim) / sizeof(double);
    }

    // Linear iterators
    const T * begin() const { return static_cast<const T *>(PyArray_DATA(array())); }
    const T * end() const { return begin() + numel(); }

    // Strided (dimension) iterators
    Iter begin(const size_t dim, const size_t skip = 0) const
    {
        return Iter(begin() + skip, stride(dim));
    }
    Iter end(const size_t dim, const size_t skip = 0) const
    {
        return begin(dim, skip) + size(dim);
    }

    const T & value() const
    {
        assert(numel() > 0);
        return *begin();
    }
    const T & value(const size_t i) const
    {
        return begin()[i];
    }
    const T & operator[](const size_t i) const { return value(i); }
    const T & operator()(const size_t i) const { return value(i); }
    operator const T &() const { return value(); }
    operator const PyArrayObject *() const { return array(); }
protected:
    const PyArrayObject *carr_;
    ConstArray():
        carr_(NULL) {}
    bool check() const;
private:
    ConstArray(const ConstArray &);
};

template<>
bool ConstArray<double>::check() const
{
//    assert(mxIsDouble(array()));
    return true;
}

template<typename T>
class Array: public ConstArray<T>
{
public:
    typedef voxel_map::Strided<T *> Iter;

    Array(PyObject *arr):
        arr_(reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(arr, NPY_DOUBLE,
                                                                NPY_ARRAY_C_CONTIGUOUS
                                                                | NPY_ARRAY_WRITEABLE
                                                                | NPY_ARRAY_ALIGNED
                                                                | NPY_ARRAY_ENSURECOPY)))
    {
        ConstArray<T>::carr_ = const_cast<const PyArrayObject *>(arr_);
        assert(check());
    }
    Array(const size_t m, const size_t n)
    {
        ConstArray<T>::carr_ = arr_ = create(m, n);
        assert(check());
    }
    Array(const size_t m)
    {
        ConstArray<T>::carr_ = arr_ = create(m);
        assert(check());
    }
    PyArrayObject * array(bool transferOwnership = false)
    {
        assert(arr_ != NULL);
        if (transferOwnership)
            Py_XINCREF(arr_);
        return arr_;
    }

    // Linear iterators
    T * begin() { return static_cast<T *>(PyArray_DATA(array())); }
    T * end() { return begin() + ConstArray<T>::numel(); }

    // Strided (dimension) iterators
    Iter begin(const size_t dim, const size_t skip = 0)
    {
        return Iter(begin() + skip, ConstArray<T>::stride(dim));
    }
    Iter end(const size_t dim, const size_t skip = 0)
    {
        return begin(dim, skip) + ConstArray<T>::size(dim);
    }

    T & value()
    {
        assert(ConstArray<T>::numel() > 0);
        return *begin();
    }
    T & value(const size_t i)
    {
        return begin()[i];
    }
    T & operator[](const size_t i) { return value(i); }
    T & operator()(const size_t i) { return value(i); }
    operator T &() { return value(); }
    operator PyArrayObject *() { return array(); }
    operator ConstArray<T>() { return ConstArray<T>(array()); }
protected:
    PyArrayObject *arr_;
    PyArrayObject * create(const size_t m, const size_t n);
    PyArrayObject * create(const size_t m);
    bool check() const;
private:
    Array(const Array &);
};

template<>
PyArrayObject * Array<double>::create(const size_t m, const size_t n)
{
    npy_intp dims[2] = {m, n};
    PyObject *obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    return reinterpret_cast<PyArrayObject *>(obj);
}

template<>
PyArrayObject * Array<double>::create(const size_t m)
{
    npy_intp dims[1] = {m};
    PyObject *obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    return reinterpret_cast<PyArrayObject *>(obj);
}

template<>
bool Array<double>::check() const
{
//    assert(ConstArray<double>::check());
//    assert(mxIsDouble(array()));
    return true;
}

}

#endif // VOXEL_MAP_ARRAY_H

