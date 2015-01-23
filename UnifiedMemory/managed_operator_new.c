#include <exception> // for std::bad_alloc
#include <new>
#include <cstdio>
#include <cuda_runtime_api.h>

void * operator new(std::size_t n) throw(std::bad_alloc)
{
	void *ptr;
	::cudaMallocManaged(&ptr, n, cudaMemAttachGlobal);

	::cudaError err = cudaGetLastError();
	if (cudaSuccess != err){
		throw std::bad_alloc();
	}
	return ptr;
}
void operator delete(void * ptr) throw()
{
	::cudaFree(ptr);
}

void *operator new[](std::size_t len) throw(std::bad_alloc)
{
	return ::operator new(len);
}
void operator delete[](void *ptr) throw()
{
	::operator delete(ptr);
}
