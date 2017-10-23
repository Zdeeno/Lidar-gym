#ifndef VOXEL_MAP_VOXEL_MAP_H
#define VOXEL_MAP_VOXEL_MAP_H

#include <boost/function.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

namespace voxel_map
{

template<typename T>
class Voxel {
public:
    typedef boost::function<bool (const Voxel &)> Filter;
    typedef T ElemType;
    typedef const T * const_iterator;
    typedef T * iterator;

    class Hash {
    public:
        size_t operator()(const Voxel &v) const
        {
            size_t h = 0;
            for (size_t i = 0; i < 4; ++i)
                boost::hash_combine(h, v[i]);
            return h;
        }
    };
    typedef boost::unordered_set<Voxel, Hash> Set;

    class ChildIterator {
    private:
        const Voxel &parent;
        int i;
    public:
        ChildIterator(const ChildIterator &o):
            parent(o.parent), i(o.i) {}
        ChildIterator(const Voxel &parent, const int i):
            parent(parent), i(i) {}
        Voxel operator*() { return parent.child(i); }
        ChildIterator & operator++()
        {
            ++i;
            return *this;
        }
        bool operator==(const ChildIterator &it)
        {
            return (parent == it.parent && i == it.i);
        }
        bool operator!=(const ChildIterator &it)
        {
            return !operator==(it);
        }
    };

    union
    {
        struct { T x, y, z, level; };
        T array[4];
    };

    Voxel(const T x, const T y, const T z, const T level):
        x(x), y(y), z(z), level(level) {}
    Voxel(const T x, const T y, const T z):
        x(x), y(y), z(z), level(0) {}
    Voxel(): x(0), y(0), z(0), level(0) {}
    Voxel(const Voxel &o):
        x(o.x), y(o.y), z(o.z), level(o.level) {}

    template<typename T1>
    operator Voxel<T1>() const
    {
        return Voxel<T1>(x, y, z, level);
    }

    T & operator[](const size_t i)             { return array[i]; }
    T & operator()(const size_t i)             { return array[i]; }
    const T & operator[](const size_t i) const { return array[i]; }
    const T & operator()(const size_t i) const { return array[i]; }

    const_iterator begin() const { return array; }
    const_iterator end()   const { return array + 4; }
    iterator begin()             { return array; }
    iterator end()               { return array + 4; }

    Voxel parent() const
    {
        return Voxel((x >= 0) ? x/2 : (x-1)/2,
                     (y >= 0) ? y/2 : (y-1)/2,
                     (z >= 0) ? z/2 : (z-1)/2,
                     level + 1);
    }
    Voxel child(const int i) const
    {
        assert(i >= 0 && i < 8);
        return Voxel(2*x + bool(i & 1),
                     2*y + bool(i & 2),
                     2*z + bool(i & 4),
                     level - 1);
    }
    ChildIterator childBegin() const { return ChildIterator(*this, 0); }
    ChildIterator childEnd()   const { return ChildIterator(*this, 8); }

    Voxel maxChild() const { return Voxel(2*x, 2*y, 2*z, level - 1); }

    size_t hash()
    {
        return Hash()(*this);
    }
    long distanceSquared(const Voxel &o)
    {
        return (long(x-o.x)*(x-o.x) + long(y-o.y)*(y-o.y) + long(z-o.z)*(z-o.z)) << (2*level);
    }
    std::string toString() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
    friend std::ostream& operator<<(std::ostream &out, const Voxel &v) {
        out << "Voxel(" << v.x << ", " << v.y << ", " << v.z << ", " << v.level << ")";
        return out;
    }
    friend bool operator==(const Voxel &a, const Voxel &b) {
        for (size_t i = 0; i < 4; ++i)
        {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }
    friend bool operator<(const Voxel &a, const Voxel &b) {
        for (size_t i = 0; i < 4; ++i)
        {
            if (a[i] < b[i])
                return true;
            else if (a[i] > b[i])
                return false;
        }
        return false;
    }
};

template<typename V, typename T>
class VoxelMap {
public:
    typedef V Voxel;
    typedef T Value;
    typedef boost::function<T (const T &, const T &)> Reduce;
    typedef boost::unordered_map<Voxel, T, typename Voxel::Hash> Map;
    typedef typename Map::iterator Iter;
    typedef typename Map::const_iterator ConstIter;

    Map map;
    typename V::ElemType min_level;
    typename V::ElemType max_level;

    VoxelMap(const typename V::ElemType min_level, const typename V::ElemType max_level, const size_t init_size):
        map(init_size), min_level(min_level), max_level(max_level) {}
    VoxelMap(const typename V::ElemType max_level, const size_t init_size):
        map(init_size), min_level(0), max_level(max_level) {}
    VoxelMap(const typename V::ElemType max_level):
        max_level(max_level) {}
    VoxelMap(): max_level(0) {}
    ~VoxelMap() {}

    size_t    size()  const { return map.size(); }
    void      clear()       {        map.clear(); }
    Iter      begin()       { return map.begin(); }
    ConstIter begin() const { return map.begin(); }
    Iter      end()         { return map.end(); }
    ConstIter end()   const { return map.end(); }

    bool voxelKnown(const Voxel &v) const
    {
        return (map.find(v) != map.end());
    }
    T & value(const Voxel &v)
    {
        assert(voxelKnown(v));
        return map.find(v)->second;
    }
    const T & value(const Voxel &v) const
    {
        assert(voxelKnown(v));
        return map.find(v)->second;
    }
    T & operator()(const Voxel &v)
    {
        return value(v);
    }
    const T & operator()(const Voxel &v) const
    {
        return value(v);
    }
    T & operator[](const Voxel &v)
    {
        return map[v];
    }
    bool ancestorKnown(const Voxel &v) const
    {
        return (findAncestor(v) != map.end());
    }
    Voxel ancestor(const Voxel &v) const
    {
        assert(ancestorKnown(v));
        return findAncestor(v)->first;
    }
    T & ancestorValue(const Voxel &v)
    {
        assert(ancestorKnown(v));
        findAncestor(v)->second;
    }
    const T & ancestorValue(const Voxel &v) const
    {
        assert(ancestorKnown(v));
        findAncestor(v)->second;
    }
    Iter findAncestor(Voxel v)
    {
        while (v.level <= max_level)
        {
             if (map.find(v) != map.end())
                 return map.find(v);
             v = v.parent();
        }
        return map.end();
    }
    ConstIter findAncestor(Voxel v) const
    {
        while (v.level <= max_level)
        {
             if (map.find(v) != map.end())
                 return map.find(v);
             v = v.parent();
        }
        return map.end();
    }
    Iter erase(ConstIter it)
    {
        return map.erase(it);
    }
protected:
    VoxelMap & nonConst() const
    {
        return const_cast<VoxelMap &>(*this);
    }
};

template<typename V>
inline bool alwaysTrue(const V &)  { return true; }
template<typename V>
inline bool alwaysFalse(const V &) { return false; }

template<typename V>
bool voxelPositionInBox(const V &v, const V &min, const V &max)
{
    for (size_t i = 0; i < 3; ++i)
        if (v[i] < min[i] || v[i] > max[i])
            return false;
    return true;
}

template<typename V>
bool voxelPositionInRange(const V &v, const typename V::ElemType min, const typename V::ElemType max)
{
    for (size_t i = 0; i < 3; ++i)
        if (v[i] < min || v[i] > max)
            return false;
    return true;
}

template<typename V, typename T>
bool voxelInRange(const VoxelMap<V, T> &map, const V &v, const T &min, const T &max, const bool check_unknown = true)
{
    return (map.voxelKnown(v) && (map.value(v) >= min) && (map.value(v) <= max))
            || (!map.voxelKnown(v) && !check_unknown);
}

template<typename V, typename T>
bool ancestorInRange(const VoxelMap<V, T> &map, const V &v, const T &min, const T &max, const bool check_unknown = true)
{
    return (map.ancestorKnown(v) && (map.ancestorValue(v) >= min) && (map.ancestorValue(v) <= max))
            || (!map.ancestorKnown(v) && !check_unknown);
}

/// Collapse children of given parent.
/// Pre: parent may not exist, all children exist.
/// Post: parent exists, children do not exist.
template<typename V, typename T>
void collapseChildren(const V &p, VoxelMap<V, T> &map,
                      const boost::function<T (const T &, const T &)> &reduce = std::plus<T>(),
                      const T &initial = -std::numeric_limits<T>::infinity())
{
    assert(p.level >= map.max_level);
    if (!map.voxelKnown(p))
        map[p] = initial;
    for (typename V::ChildIterator it = p.childBegin(); it != p.childEnd(); ++it)
    {
        assert(map.voxelKnown(*it));
        map[p] = reduce(map[p], map[*it]);
        map.map.erase(*it);
    }
}

/// Collapse once.
/// 1) Create set of parents with children to collapse.
/// 2) Collapse children of the parents.
template<typename V, typename T>
void collapseOnce(VoxelMap<V, T> &map, const typename V::Filter &filter,
                  const boost::function<T (const T &, const T &)> &reduce = std::max<T>,
                  const T &initial = -std::numeric_limits<T>::infinity())
{
    // Create a set of parents to check (parents may not yet exist).
    typename V::Set parents;
    for (typename VoxelMap<V, T>::Map::const_iterator it = map.map.begin(); it != map.map.end(); ++it)
    {
        assert(it->first.level <= map.max_level);
        if (it->first.level >= map.max_level)
            continue;
        V parent = it->first.parent();
        parents.insert(parent);
    }
    // Collapse to parent voxel if all the children pass the filter.
    for (typename V::Set::const_iterator p_it = parents.begin(); p_it != parents.end(); ++p_it)
    {
        bool collapse = true;
        for (typename V::ChildIterator c_it = p_it->childBegin(); c_it != p_it->childEnd(); ++c_it)
        {
            if (!filter(*c_it))
            {
                collapse = false;
                break;
            }
        }
        if (collapse)
            collapseChildren(*p_it, map, reduce, initial);
    }
}

/// Collapse all.
template<typename V, typename T>
void collapseMap(VoxelMap<V, T> &map, const typename V::Filter &filter,
                 const boost::function<T (const T &, const T &)> &reduce = std::max<T>,
                 const T &initial = -std::numeric_limits<T>::infinity())
{
    for (typename V::ElemType i = 0; i < map.max_level; ++i)
    {
        collapseOnce(map, filter, reduce, initial);
    }
}

/// Expand children of given parent.
/// Pre: parent exist, children may not exist.
/// Post: all children exist, parent does not exist.
template<typename V, typename T>
void expandParent(const V &v, VoxelMap<V, T> &map,
                    const boost::function<T (const T &, const T &)> &reduce = std::max<T>,
                    const T &initial = -std::numeric_limits<T>::infinity())
{
    assert(map.voxelKnown(v));
    assert(v.level > map.min_level);
    for (typename V::ChildIterator it = v.childBegin(); it != v.childEnd(); ++it)
    {
        assert(map.voxelKnown(*it));
        if (!map.voxelKnown(*it))
            map[*it] = initial;
        map[*it] = reduce(map[v], map[*it]);
    }
    map.map.erase(v);
}

/// Expand once.
/// 1) Create set of parents to expand.
/// 2) Expand and remove these parents.
template<typename V, typename T>
void expandOnce(VoxelMap<V, T> &map, const typename V::Filter &filter,
                const boost::function<T (const T &, const T &)> &reduce = std::max<T>,
                const T &initial = -std::numeric_limits<T>::infinity())
{
    // Create a set of voxels to expand,
    // i.e., a set of the nearest ancestors of the voxels passing the filter.
    typename V::Set to_expand;
    for (typename VoxelMap<V, T>::Map::const_iterator it = map.map.begin(); it != map.map.end(); ++it)
    {
        if (!filter(it->first) || !map.ancestorKnown(it->first.parent()))
            continue;
        to_expand.insert(map.ancestor(it->first.parent()));
    }
    // Expand and remove the parents.
    for (typename V::Set::const_iterator p_it = to_expand.begin(); p_it != to_expand.end(); ++p_it)
    {
        expandParent(*p_it, map, reduce, initial);
    }
}

/// Expand all.
template<typename V, typename T>
void expandMap(VoxelMap<V, T> &map, const typename V::Filter &filter,
               const boost::function<T (const T &, const T &)> &reduce = std::max<T>,
               const T &initial = -std::numeric_limits<T>::infinity())
{
    for (typename V::ElemType i = map.max_level; i > 0; --i)
    {
        expandOnce(map, filter, reduce, initial);
    }
}

/// Merge children to closest ancestors.
template<typename V, typename T>
void mergeOnce(VoxelMap<V, T> &map,
               const boost::function<T (const T &, const T &)> &reduce = std::max<T>)
{
    for (typename VoxelMap<V, T>::Iter it = map.begin(); it != map.end();)
    {
        typename VoxelMap<V, T>::Iter ancestor_it = map.findAncestor(it->first.parent());
        if (ancestor_it == map.end())
             ++it;
        else
        {
            ancestor_it->second = reduce(ancestor_it->second, it->second);
            map.erase(it++);
        }
    }
}

template<typename V>
class VoxelTraits {};

template<>
class VoxelTraits<Voxel<char> >
{
public:
    typedef Voxel<char>::ElemType ElemType;
    typedef short CompType;
    typedef float CompFloatType;
    static char lo() { return std::numeric_limits<char>::min(); }
    static char hi() { return std::numeric_limits<char>::max(); }
};

template<>
class VoxelTraits<Voxel<short> >
{
public:
    typedef Voxel<short>::ElemType ElemType;
    typedef int CompType;
    typedef double CompFloatType;
    static short lo() { return std::numeric_limits<short>::min(); }
    static short hi() { return std::numeric_limits<short>::max(); }
};

template<>
class VoxelTraits<Voxel<int> >
{
public:
    typedef Voxel<int>::ElemType ElemType;
    typedef long CompType;
    typedef double CompFloatType;
    static int lo() { return std::numeric_limits<int>::min(); }
    static int hi() { return std::numeric_limits<int>::max(); }
};

template<>
class VoxelTraits<Voxel<long> >
{
public:
    typedef Voxel<long>::ElemType ElemType;
    typedef long CompType;
    typedef double CompFloatType;
//    static long lo() { return std::numeric_limits<long>::min() / 2; }
//    static long hi() { return std::numeric_limits<long>::max() / 2; }
    // Or half highest double representable, 9007199254740992 / 2?
    static long lo() { return -9007199254740992l / 2; }
    static long hi() { return +9007199254740992l / 2; }
};

/// Rasterize line from v0 to v1 until the filter returns false, return the last voxel checked.
/// \brief line
/// \param v0
/// \param v1
/// \param filter
/// \param voxels All voxels checked.
/// \return The last voxel checked.
///
template<typename V>
V line(const V &v0, const V &v1, const typename V::Filter &filter = &alwaysTrue<V>, std::vector<V> *voxels = NULL)
{
    assert(v0.level == v1.level);
    assert(v0.level == 0);
    typedef typename VoxelTraits<V>::CompType T1;
    T1 i, dx, dy, dz, l, m, n, x_inc, y_inc, z_inc, err_1, err_2, dx2, dy2, dz2;
    dx = T1(v1.x) - v0.x;
    dy = T1(v1.y) - v0.y;
    dz = T1(v1.z) - v0.z;
    x_inc = (dx < 0) ? -1 : 1;
    l = std::abs(dx);
    y_inc = (dy < 0) ? -1 : 1;
    m = std::abs(dy);
    z_inc = (dz < 0) ? -1 : 1;
    n = std::abs(dz);
    dx2 = l << 1;
    dy2 = m << 1;
    dz2 = n << 1;
    // Insert every new voxel into the container if provided.
    // Rather not reserve the capacity to max(abs(v1-v0) because the distance may be too large.
    V v = v0;
    if (voxels)
        voxels->push_back(v);
    if ((l >= m) && (l >= n))
    {
        err_1 = dy2 - l;
        err_2 = dz2 - l;
        for (i = 0; i < l; i++)
        {
            if (!filter(v))
                break;
            if (err_1 > 0)
            {
                v[1] += y_inc;
                err_1 -= dx2;
            }
            if (err_2 > 0)
            {
                v[2] += z_inc;
                err_2 -= dx2;
            }
            err_1 += dy2;
            err_2 += dz2;
            v[0] += x_inc;
            if (voxels)
                voxels->push_back(v);
        }
    }
    else if ((m >= l) && (m >= n))
    {
        err_1 = dx2 - m;
        err_2 = dz2 - m;
        for (i = 0; i < m; i++)
        {
            if (!filter(v))
                break;
            if (err_1 > 0)
            {
                v[0] += x_inc;
                err_1 -= dy2;
            }
            if (err_2 > 0)
            {
                v[2] += z_inc;
                err_2 -= dy2;
            }
            err_1 += dx2;
            err_2 += dz2;
            v[1] += y_inc;
            if (voxels)
                voxels->push_back(v);
        }
    }
    else
    {
        err_1 = dy2 - n;
        err_2 = dx2 - n;
        for (i = 0; i < n; i++)
        {
            if (!filter(v))
                break;
            if (err_1 > 0)
            {
                v[1] += y_inc;
                err_1 -= dz2;
            }
            if (err_2 > 0)
            {
                v[0] += x_inc;
                err_2 -= dz2;
            }
            err_1 += dy2;
            err_2 += dx2;
            v[2] += z_inc;
            if (voxels)
                voxels->push_back(v);
        }
    }
    return v;
}

template<typename V>
V endPoint(const V &v0, const double *d, const double max_range = std::numeric_limits<double>::infinity())
{
    assert(v0.level == 0);
    // Compute unit direction.
    const double r = hypot(hypot(d[0], d[1]), d[2]);
    assert(r > 0);
    const double d1[3] = {d[0]/r, d[1]/r, d[2]/r};
    // Keep voxel coordinates in representable limits.
    const double lo = VoxelTraits<V>::lo();
    const double hi = VoxelTraits<V>::hi();
    double range = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < 3; ++i)
    {
        if (d1[i] > 0)
        {
            const double r_i = (hi - v0[i]) / d1[i];
            range = std::min(r_i, range);
        }
        else if (d1[i] < 0)
        {
            const double r_i = (lo - v0[i]) / d1[i];
            range = std::min(r_i, range);
        }
    }
    range = std::min(range, max_range);
    return V(std::floor(v0[0] + range * d1[0] + 0.5),
             std::floor(v0[1] + range * d1[1] + 0.5),
             std::floor(v0[2] + range * d1[2] + 0.5),
             v0.level);
}

template<typename V>
V lineFloat(const V &v0, const V &v1, const typename V::Filter &filter = &alwaysTrue<V>, std::vector<V> *voxels = NULL)
{
    typedef typename VoxelTraits<V>::CompFloatType T1;
    typedef Voxel<T1> VF;
    // Find the fastest changing coordinate.
    const T1 d[3] = {T1(v1[0]) - v0[0], T1(v1[1]) - v0[1], T1(v1[2]) - v0[2]};
    const T1 da[3] = {std::abs(d[0]), std::abs(d[1]), std::abs(d[2])};
    const size_t i_max = std::distance(da, std::max_element(da, da + 3));
    // Create a unit step in maximum norm.
    const T1 r = da[i_max];
    assert(r > 0);
    const T1 d1[3] = {d[0] / r, d[1] / r, d[2] / r};
    VF vf = v0;
    V v = v0;
    for (; vf[i_max] != (v1[i_max] + d1[i_max]); vf[0] += d1[0], vf[1] += d1[1], vf[2] += d1[2])
    {
        v = V(std::floor(vf[0] + 0.5), std::floor(vf[1] + 0.5), std::floor(vf[2] + 0.5));
        if (voxels)
            voxels->push_back(v);
        if (!filter(v))
            break;
    }
    return v;
}

/// Trace ray from v1 in the specified direction until the filter returns false, return the last voxel checked.
/// \brief ray
/// \param v0
/// \param d
/// \param filter
/// \param max_range
/// \param voxels
/// \return The last voxel checked.
///
template<typename V>
V ray(const V &v0, const double *d,
      const typename V::Filter &filter = &alwaysTrue<V>,
      const double max_range = std::numeric_limits<double>::infinity(),
      std::vector<V> *voxels = NULL)
{
    return line(v0, endPoint(v0, d, max_range), filter, voxels);
}

} // namespace

#endif
