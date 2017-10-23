
#include <omp.h>
#include <voxel_map/voxel_map.h>

namespace voxel_map {

/// Get values for query voxels.
/// @parma map
/// @param x Query x start iterator
/// @param x_end Query x end iterator
/// @param y
/// @param z
/// @param level Query level iterator
/// @param v Input value iterator
template<typename V, typename T, typename It1, typename It2, typename It3>
void getVoxels(const VoxelMap<V, T> &map, It1 x, It1 x_end, It1 y, It1 z, It2 level, It3 v)
{
    for (; x != x_end; ++x, ++y, ++z, ++level, ++v)
    {
        *v = map.voxelKnown(V(*x, *y, *z, *level))
                ? map.value(V(*x, *y, *z, *level))
                : std::numeric_limits<T>::quiet_NaN();
    }
}

/// Get values for query voxels.
/// @param map
/// @param x Query voxel start iterator
/// @param x_end Query voxel end iterator
/// @param v Input value iterator
template<typename V, typename T, typename It1, typename It2>
void getVoxels(const VoxelMap<V, T> &map, It1 x, It1 x_end, It2 v)
{
    for (; x != x_end; ++x, ++v)
        *v = map.voxelKnown(*x) ? map.value(*x) : std::numeric_limits<T>::quiet_NaN();
}

/// Get all defined voxels.
/// @parma map
/// @param x Input x iterator
/// @param y Input y iterator
/// @param z Input z iterator
/// @param level Input level iterator
/// @parma v Input v iterator
template<typename V, typename T, typename It1, typename It2, typename It3>
void getVoxels(const VoxelMap<V, T> &map, It1 x, It1 y, It1 z, It2 level, It3 v)
{
    for (typename VoxelMap<V, T>::Map::const_iterator it = map.map.begin(); it != map.map.end(); ++it)
    {
        *x++ = it->first[0];
        *y++ = it->first[1];
        *z++ = it->first[2];
        *level++ = it->first[3];
        *v++ = it->second;
    }
}

/// Set voxel values.
/// @param map
/// @param x Coord. X iterator
/// @param y Coord. Y iterator
/// @param z Coord. Z iterator
/// @param level Voxel level iterator
/// @parma v Value iterator
template<typename V, typename T, typename It1, typename It2, typename It3>
void setVoxels(VoxelMap<V, T> &map, It1 x, It1 x_end, It1 y, It1 z, It2 level, It3 v)
{
    map.map.reserve(map.size() + std::distance(x, x_end));
    for (; x != x_end; ++x, ++y, ++z, ++level, ++v)
    {
        map[V(*x, *y, *z, *level)] = *v;
    }
}

template<typename V, typename T, typename It1, typename It2, typename It3>
void updateVoxels(VoxelMap<V, T> &map, It1 x, It1 x_end, It1 y, It1 z, It2 level, It3 v)
{
    map.map.reserve(map.size() + std::distance(x, x_end));
    for (; x != x_end; ++x, ++y, ++z, ++level, ++v)
    {
        map[V(*x, *y, *z, *level)] += *v;
    }
}

template<typename V, typename T, typename It1, typename It2, typename It3, typename It4>
void updateVoxels(VoxelMap<V, T> &map, It1 x, It1 x_end, It1 y, It1 z, It2 level, It3 v, It4 vo)
{
    map.map.reserve(map.size() + std::distance(x, x_end));
    for (; x != x_end; ++x, ++y, ++z, ++level, ++v)
    {
        map[V(*x, *y, *z, *level)] += *v;
        *vo++ = map[V(*x, *y, *z, *level)];
    }
}

template<typename V, typename T, typename It1, typename It2, typename It3>
void traceLines(const VoxelMap<V, T> &map,
                It1 fx, It1 fx_end, It1 fy, It1 fz,
                It1 tx, It1 tx_end, It1 ty, It1 tz,
                typename V::Filter &filter,
                It2 hx, It2 hy, It2 hz, It3 v)
{
    assert(std::distance(fx, fx_end) == std::distance(tx, tx_end));
    size_t n = std::distance(fx, fx_end);
#pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < n; ++i)
    {
        const V v0(fx[i], fy[i], fz[i]);
        const V v1(tx[i], ty[i], tz[i]);
        const V vh = voxel_map::line(v0, v1, filter);
        hx[i] = vh[0];
        hy[i] = vh[1];
        hz[i] = vh[2];
        v[i] = map.voxelKnown(vh) ? map(vh) : std::numeric_limits<T>::quiet_NaN();
    }
}

template<typename V, typename T, typename It1, typename It2, typename It3, typename It4>
void traceRays(const VoxelMap<V, T> &map,
               It1 fx, It1 fx_end, It1 fy, It1 fz,
               It2 dx, It2 dx_end, It2 dy, It2 dz,
               typename V::Filter &filter,
               const double max_range,
               It3 hx, It3 hy, It3 hz, It4 v,
               std::vector<std::vector<V> > *rays = NULL)
{
    size_t n = std::max(std::distance(fx, fx_end), std::distance(dx, dx_end));
    if (rays != NULL) rays->resize(n);
#pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < n; ++i)
    {
        const V v0(fx[i], fy[i], fz[i]);
        double d[3] = {dx[i], dy[i], dz[i]};
        std::vector<V> *voxels = (rays != NULL) ? &(*rays)[i] : NULL;
        const V vh = voxel_map::ray(v0, d, filter, max_range, voxels);
        hx[i] = vh[0];
        hy[i] = vh[1];
        hz[i] = vh[2];
        v[i] = map.voxelKnown(vh) ? map(vh) : std::numeric_limits<T>::quiet_NaN();
    }
}

template<typename V, typename T, typename It1, typename It2>
void updateLines(VoxelMap<V, T> &map,
                 It1 fx, It1 fx_end, It1 fy, It1 fz,
                 It2 tx, It2 tx_end, It2 ty, It2 tz,
                 const T &free_update, const T &hit_update)
{
    assert(std::distance(fx, fx_end) == std::distance(tx, tx_end));
    // First, update the hit voxels and create a set thereof.
    typename V::Set occupied;
    for (It2 hx = tx, hy = ty, hz = tz; hx != tx_end; ++hx, ++hy, ++hz)
    {
        V v1(*hx, *hy, *hz);
        occupied.insert(v1);
        map[v1] += hit_update;
    }
    for (; fx != fx_end; ++fx, ++fy, ++fz, ++tx, ++ty, ++tz)
    {
        V v0(*fx, *fy, *fz);
        V v1(*tx, *ty, *tz);
        std::vector<V> voxels;
        voxel_map::line<V>(v0, v1, &voxel_map::alwaysTrue<V>, &voxels);
        for (size_t j = 0; j < voxels.size() - 1; ++j)  // The hit voxel is already updated.
        {
            // Update the visited voxel as free if it is not included in the occupied set.
            if (occupied.find(voxels[j]) == occupied.end())
            {
                map[voxels[j]] += free_update;
            }
        }
    }
}

} // namespace
