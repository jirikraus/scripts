#ifndef MANAGED_ALLOCATOR_CUH
#define MANAGED_ALLOCATOR_CUH
#include <memory>
#include <cuda_runtime_api.h>

template <class T>
class managed_allocator : public std::allocator<T>
{
public:
    typedef size_t size_type;
    typedef std::ptrdiff_t difference_type;

    typedef T* pointer;
    typedef const T* const_pointer;

    typedef T& reference;
    typedef const T& const_reference;

    typedef T value_type;

    template <class U>
    struct rebind { typedef managed_allocator<U> other; };

    managed_allocator() throw() { }
    managed_allocator(const managed_allocator& a) throw()
        : std::allocator<T>(a) { }
    template <class U>
    managed_allocator(const managed_allocator<U>&) throw() { }
    ~managed_allocator() throw() { }
    pointer allocate(size_type n, managed_allocator<T>::const_pointer /*hint*/ = 0) // space for n Ts
    {
        void * p;
        cudaMallocManaged(&p,n*sizeof(T));
        return pointer(p);
    }
    void deallocate(pointer p, size_type /*n*/)   // deallocate n Ts, don't destroy
    {
        cudaFree(p);
    }

    void construct(pointer p, const T& val) { new(p) T(val); }  // initialize *p by val
    void destroy(pointer p) { p->~T(); }                        // destroy *p but don't deallocate
};

template<class T1, class T2>
bool operator==(const managed_allocator<T1>&, const managed_allocator<T2>&)
{
    return true;
}

template<class T1, class T2>
bool operator!=(const managed_allocator<T1>&, const managed_allocator<T2>&)
{
    return false;
}

#endif //MANAGED_ALLOCATOR_CUH
