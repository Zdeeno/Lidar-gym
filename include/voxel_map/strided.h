#ifndef VOXEL_MAP_STRIDED_H
#define VOXEL_MAP_STRIDED_H

namespace voxel_map {

template<typename It>
class Strided {
public:
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef typename std::iterator_traits<It>::iterator_category iterator_category;
    typedef typename std::iterator_traits<It>::pointer pointer;
    typedef typename std::iterator_traits<It>::reference reference;
    typedef typename std::iterator_traits<It>::value_type value_type;

    Strided(const It &iter, const difference_type stride):
        iter_(iter), stride_(stride) {
    }
    Strided(const Strided &other): iter_(other.iter_), stride_(other.stride_) {
    }
    reference operator*() const {
        return *iter_;
    }
    Strided<It> & operator++() {
        iter_ += stride_;
        return *this;
    }
    Strided<It> operator++(int)
    {
         Strided<It> copy(*this);
         operator++();
         return copy;
    }
    Strided<It> & operator--() {
        iter_ -= stride_;
        return *this;
    }
    Strided<It> operator--(int)
    {
         Strided<It> copy(*this);
         operator--();
         return copy;
    }
    bool operator==(const Strided &other) {
        return (iter_ == other.iter_);
    }
    bool operator!=(const Strided &other) {
        return !this->operator==(other);
    }
    bool operator<(const Strided &other) {
        return (iter_ < other.iter_);
    }
    bool operator<=(const Strided &other) {
        return (iter_ <= other.iter_);
    }
    bool operator>(const Strided &other) {
        return (iter_ > other.iter_);
    }
    bool operator>=(const Strided &other) {
        return (iter_ >= other.iter_);
    }
    Strided<It> & operator+=(const difference_type diff) {
        iter_ += stride_ * diff;
        return *this;
    }
    Strided<It> & operator-=(const difference_type diff) {
        iter_ -= stride_ * diff;
        return *this;
    }
    reference operator[](const difference_type diff) {
        return *(iter_ + stride_ * diff);
    }
    friend difference_type operator-(const Strided<It> it1, const Strided<It> it0) {
        assert(it0.stride_ == it1.stride_);
        return (it1.iter_ - it0.iter_) / it1.stride_;
    }
private:
    It iter_;
    difference_type stride_;
};

//template<typename It>
//Strided<It> strided(It it, typename Strided<It>::difference_type stride)
//{
//    return Strided<It>(it, stride);
//}

template<typename It>
Strided<It> operator+(Strided<It> iter, const typename Strided<It>::difference_type diff) {
    return (iter += diff);
}
template<typename It>
Strided<It> operator-(Strided<It> iter, const typename Strided<It>::difference_type diff) {
    return (iter -= diff);
}

class Void
{
public:
    template<typename T>
    Void(const T &) {}
    Void() {}
    Void & operator*() { return *this; }
    Void & operator++(int) { return *this; }
    const Void & operator*() const { return *this; }
    const Void & operator++(int) const { return *this; }
    size_t size() const { return 0; }
};

template<typename T>
class FixedValueIterator
{
public:
    FixedValueIterator(const T &value, const long i = 0):
        i(i), value(value) {}
    const T & operator*() const { return value; }
    FixedValueIterator<T> & operator++()    { ++i; return *this; }
    FixedValueIterator<T> & operator--()    { --i; return *this; }
    FixedValueIterator<T> & operator++(int)
    {
        FixedValueIterator<T> ret(*this);
        ++i;
        return ret;
    }
    FixedValueIterator<T> & operator--(int)
    {
        FixedValueIterator<T> ret(*this);
        --i;
        return ret;
    }
    bool operator==(const FixedValueIterator<T> &it)
    {
        return (i == it.i);
    }
    bool operator!=(const FixedValueIterator<T> &it)
    {
        return !(*this == it);
    }
private:
    long i;
    const T value;
};

} // namespace voxel_map

#endif // VOXEL_MAP_STRIDED_H
