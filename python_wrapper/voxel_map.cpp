
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
//#include <boost/range/adaptor/strided.hpp>
#include <boost/python.hpp>
//#include <boost/python/numeric.hpp>
#include <numpy/arrayobject.h>

#include "array.h"
#include <voxel_map/strided.h>
#include <voxel_map/wrapper.h>

namespace {

namespace vm = voxel_map;
namespace bp = boost::python;

typedef vm::Voxel<int> Voxel;
typedef float Value;
typedef vm::VoxelMap<Voxel, Value> VoxelMap;

typedef std::vector<Voxel> VoxelList;
typedef boost::unordered_set<Voxel, typename Voxel::Hash> VoxelSet;

class PyVoxelMap {
private:
    VoxelMap vox_map;

public:
    double voxel_size;
    Value free_update;
    Value hit_update;
    Value occupied_threshold;

    PyVoxelMap():
        vox_map(),
        voxel_size(1.0),
        free_update(-1.0),
        hit_update(1.0),
        occupied_threshold(0.0)
    {}

    PyVoxelMap(const double voxel_size,
               const double free_update,
               const double hit_update,
               const double occupied_threshold):
        voxel_size(voxel_size),
        free_update(free_update),
        hit_update(hit_update),
        occupied_threshold(occupied_threshold)
    {}

    PyObject * size()
    {
        return PyInt_FromSize_t(vox_map.size());
    }

    void clear()
    {
        vox_map.clear();
    }

    PyObject * getVoxels()
    {
        Array<double> x(3, vox_map.size());
        Array<double> v(vox_map.size());
        Array<double> l(vox_map.size());

        vm::getVoxels(vox_map, x.begin(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                      l.begin(), v.begin());

        toVoxelCenters(x.begin(), x.end(), x.begin());

        PyObject *res = PyTuple_New(3);
        PyTuple_SetItem(res, 0, reinterpret_cast<PyObject *>(x.array(true)));
        PyTuple_SetItem(res, 1, reinterpret_cast<PyObject *>(l.array(true)));
        PyTuple_SetItem(res, 2, reinterpret_cast<PyObject *>(v.array(true)));
        return res;
    }
    
    PyObject * getVoxels(PyObject *x_obj, PyObject *l_obj)
    {
        Array<double> x(x_obj);
        Array<double> l(l_obj);
        if (x.size(0) != 3)
            throw std::runtime_error("Argument x must be 2-D array of size 3-by-N.");
        if (l.numel() != x.size(1))
            throw std::runtime_error("Argument level must be vector with N elements.");

        toVoxelIndices(x.begin(), x.end(), x.begin());
        Array<double> v(l.numel());

        vm::getVoxels(vox_map,
                      x.begin(1), x.end(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                      l.begin(), v.begin());

        return reinterpret_cast<PyObject *>(v.array(true));
    }

    void setVoxels(PyObject *x_obj, PyObject *l_obj, PyObject *v_obj)
    {
        Array<double> x(x_obj);
        Array<double> l(l_obj);
        Array<double> v(v_obj);
        if (x.size(0) != 3)
            throw std::runtime_error("Argument x must be 2-D array of size 3-by-N.");
        if (l.numel() != x.size(1))
            throw std::runtime_error("Argument level must be vector with N elements.");
        if (v.numel() != x.size(1))
            throw std::runtime_error("Argument value must be vector with N elements.");

        toVoxelIndices(x.begin(), x.end(), x.begin());

        vm::setVoxels(vox_map,
                      x.begin(1), x.end(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                      l.begin(), v.begin());
    }

    PyObject * updateVoxels(PyObject *x_obj, PyObject *l_obj, PyObject *v_obj)
    {
        Array<double> x(x_obj);
        Array<double> l(l_obj);
        Array<double> v(v_obj);
        Array<double> v1(v.numel());
        if (x.size(0) != 3)
            throw std::runtime_error("Argument x must be 2-D array of size 3-by-N.");
        if (l.numel() != x.size(1))
            throw std::runtime_error("Argument level must be vector with N elements.");
        if (v.numel() != x.size(1))
            throw std::runtime_error("Argument value must be vector with N elements.");

        toVoxelIndices(x.begin(), x.end(), x.begin());

        vm::updateVoxels(vox_map,
                         x.begin(1), x.end(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                         l.begin(), v.begin(), v1.begin());
        return reinterpret_cast<PyObject *>(v1.array(true));
    }

    void updateLines(PyObject *x_obj, PyObject *y_obj)
    {
        Array<double> x(x_obj);
        Array<double> y(y_obj);
        if (x.size(0) != 3)
            throw std::runtime_error("Argument x must be 2-D array of size 3-by-N.");
        if (y.size(0) != 3)
            throw std::runtime_error("Argument y must be 2-D array of size 3-by-N.");
        if (x.size(1) != y.size(1))
            throw std::runtime_error("Arguments x and y must have same size.");

        toVoxelIndices(x.begin(), x.end(), x.begin());
        toVoxelIndices(y.begin(), y.end(), y.begin());

        vm::updateLines(vox_map,
                        x.begin(1), x.end(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                        y.begin(1), y.end(1), y.begin(1, 1*y.stride(0)), y.begin(1, 2*y.stride(0)),
                        free_update, hit_update);
    }

    PyObject * traceLines(PyObject *x_obj, PyObject *y_obj,
                          PyObject *min_val_obj, PyObject *max_val_obj,
                          PyObject *check_unknown_obj)
    {
        Array<double> x(x_obj);
        Array<double> y(y_obj);
        if (x.size(0) != 3)
            throw std::runtime_error("Argument x must be 2-D array of size 3-by-N.");
        if (y.size(0) != 3)
            throw std::runtime_error("Argument y must be 2-D array of size 3-by-N.");
        if (x.size(1) != y.size(1))
            throw std::runtime_error("Arguments x and y must have same size.");
        double min_val = PyFloat_AsDouble(min_val_obj);
        double max_val = PyFloat_AsDouble(max_val_obj);
        bool check_unknown = PyFloat_AsDouble(check_unknown_obj);

        toVoxelIndices(x.begin(), x.end(), x.begin());
        toVoxelIndices(y.begin(), y.end(), y.begin());
        Voxel::Filter filter;
        if (std::isnan(min_val) || std::isnan(max_val))
            filter = &voxel_map::alwaysTrue<Voxel>;
        else
            filter = boost::bind(voxel_map::voxelInRange<Voxel, Value>,
                                 boost::cref(vox_map), _1, min_val, max_val, check_unknown);
        Array<double> h(x.size(0), x.size(1));
        Array<double> v(x.size(1));

        vm::traceLines(vox_map,
                       x.begin(1), x.end(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                       y.begin(1), y.end(1), y.begin(1, 1*y.stride(0)), y.begin(1, 2*y.stride(0)),
                       filter,
                       h.begin(1), h.begin(1, 1*h.stride(0)), h.begin(1, 2*h.stride(0)),
                       v.begin());

        toVoxelCenters(h.begin(), h.end(), h.begin());

        PyObject *res = PyTuple_New(2);
        PyTuple_SetItem(res, 0, reinterpret_cast<PyObject *>(h.array(true)));
        PyTuple_SetItem(res, 1, reinterpret_cast<PyObject *>(v.array(true)));
        return res;
    }

    PyObject * traceRays(PyObject *x_obj, PyObject *y_obj,
                         PyObject *max_range_obj, PyObject *min_val_obj, PyObject *max_val_obj,
                         PyObject *check_unknown_obj)
    {
        Array<double> x(x_obj);
        Array<double> y(y_obj);
        if (x.size(0) != 3)
            throw std::runtime_error("Argument x must be 2-D array of size 3-by-N.");
        if (y.size(0) != 3)
            throw std::runtime_error("Argument y must be 2-D array of size 3-by-N.");
        if (x.size(1) != y.size(1))
            throw std::runtime_error("Arguments x and y must have same size.");
        double max_range = PyFloat_AsDouble(max_range_obj) / voxel_size;
        double min_val = PyFloat_AsDouble(min_val_obj);
        double max_val = PyFloat_AsDouble(max_val_obj);
        bool check_unknown = PyFloat_AsDouble(check_unknown_obj);

        toVoxelIndices(x.begin(), x.end(), x.begin());
        Voxel::Filter filter;
        if (std::isnan(min_val) || std::isnan(max_val))
            filter = &voxel_map::alwaysTrue<Voxel>;
        else
            filter = boost::bind(voxel_map::voxelInRange<Voxel, Value>,
                                 boost::cref(vox_map), _1, min_val, max_val, check_unknown);
        Array<double> h(x.size(0), x.size(1));
        Array<double> v(x.size(1));

        vm::traceRays(vox_map,
                      x.begin(1), x.end(1), x.begin(1, 1*x.stride(0)), x.begin(1, 2*x.stride(0)),
                      y.begin(1), y.end(1), y.begin(1, 1*y.stride(0)), y.begin(1, 2*y.stride(0)),
                      filter, max_range,
                      h.begin(1), h.begin(1, 1*h.stride(0)), h.begin(1, 2*h.stride(0)),
                      v.begin());

        toVoxelCenters(h.begin(), h.end(), h.begin());

        PyObject *res = PyTuple_New(2);
        PyTuple_SetItem(res, 0, reinterpret_cast<PyObject *>(h.array(true)));
        PyTuple_SetItem(res, 1, reinterpret_cast<PyObject *>(v.array(true)));
        return res;
    }

    // TODO: Template?
    Voxel::ElemType voxelIndex(const double x)
    {
        return static_cast<Voxel::ElemType>(floor(x / voxel_size));
    }
    double voxelCenter(const Voxel::ElemType i)
    {
        return voxel_size * (i + 0.5);
    }
    template<typename I, typename O>
    void toVoxelIndices(I from, I to, O dst)
    {
        for (; from != to; ++from, ++dst)
            *dst = voxelIndex(*from);
    }
    template<typename I, typename O>
    void toVoxelCenters(I from, I to, O dst)
    {
        for (; from != to; ++from, ++dst)
            *dst = voxelCenter(*from);
    }
};

}

BOOST_PYTHON_MODULE(voxel_map)
{
    import_array();
    // TODO: Remove boost python dep completely.
    // Using object/handle etc. require compiling boost.python with -fPIC anyway.
    bp::class_<PyVoxelMap>("VoxelMap")
            .def(bp::init<const double, const double, const double, const double>())
            .def_readwrite("voxel_size", &PyVoxelMap::voxel_size)
            .def_readwrite("free_update", &PyVoxelMap::free_update)
            .def_readwrite("hit_update", &PyVoxelMap::hit_update)
            .def_readwrite("occupied_threshold", &PyVoxelMap::occupied_threshold)
            .def("size", &PyVoxelMap::size)
            .def("clear", &PyVoxelMap::clear)
            .def("get_voxels", static_cast<PyObject *(PyVoxelMap::*)()>(&PyVoxelMap::getVoxels))
            .def("get_voxels", static_cast<PyObject *(PyVoxelMap::*)(PyObject *, PyObject *)>(&PyVoxelMap::getVoxels))
            .def("set_voxels", &PyVoxelMap::setVoxels)
            .def("update_voxels", &PyVoxelMap::updateVoxels)
            .def("update_lines", &PyVoxelMap::updateLines)
            .def("trace_lines", &PyVoxelMap::traceLines)
            .def("trace_rays", &PyVoxelMap::traceRays);
}
