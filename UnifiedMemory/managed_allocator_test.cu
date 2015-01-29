#include <vector>
#include <iostream>
#include "managed_allocator.cuh"

//Use a reference to pass values from arrays in unified memory to a kernel. This avoids a CPU page fault on kernel launch.
__global__
void vecadd( double& alpha, double* __restrict__ const c, const double* __restrict__ const a, const double* __restrict__ const b, const int n)
{
	for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i+=gridDim.x*blockDim.x)
	{
		c[i] = alpha*a[i] + b[i];
	}
}

int main()
{
	int n = 1 << 16;
	
	typedef std::vector<double, managed_allocator<double> > double_vector;

	cudaSetDevice(0);
	bool success = true;
	{
		double_vector alpha(1, 1.0);
		double_vector a(n, 0.5);
		double_vector b(n, 0.5);
		double* c = new double[n];
		for ( int i = 0; i < n; ++i )
			c[i] = 0.0;
	
		vecadd<<<64,256>>>(alpha[0],&(c[0]),&(a[0]),&(b[0]),n);
		cudaDeviceSynchronize();
	
		for ( int i = 0; i < n; ++i )
		{
			if ( abs( c[i] - 1.0 ) > 1E-16 )
			{
				std::cerr<<"ERROR: c["<<i<<"] = "<<c[i]<<" != 1.0"<<std::endl;
				success = false;
				break;
			}
		}
		
		delete[] c;
	}
	
	if (success)
	{
		std::cout<<"SUCCESS"<<std::endl;
	}
	
	cudaDeviceReset();
	return success ? 0 : 1;
}
