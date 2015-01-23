#include <vector>
#include <iostream>
#include "managed_allocator.cuh"

__global__
void vecadd( double* __restrict__ const c, const double* __restrict__ const a, const double* __restrict__ const b, const int n)
{
	for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i+=gridDim.x*blockDim.x)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	typedef std::vector<double, managed_allocator<double> > double_vector;

	int n = 1 << 16;
	
	double_vector a(n, 0.5);
	double_vector b(n, 0.5);
	double* c = new double[n];
	for ( int i = 0; i < n; ++i )
		c[i] = 0.0;
	
	vecadd<<<64,256>>>(&(c[0]),&(a[0]),&(b[0]),n);
	cudaDeviceSynchronize();
	
	for ( int i = 0; i < n; ++i )
	{
		if ( abs( c[i] - 1.0 ) > 1E-16 )
		{
			std::cerr<<"ERROR: c["<<i<<"] = "<<c[i]<<" != 1.0"<<std::endl;
			return 1;
		}
	}
	
	std::cout<<"SUCCESS"<<std::endl;
	
	delete[] c;
	
	return 0;
}
