all: run

managed_allocator_test: Makefile managed_allocator_test.cu
	nvcc -arch=sm_35 -o managed_allocator_test managed_allocator_test.cu

run: managed_allocator_test
	./managed_allocator_test

clean:
	rm -f managed_allocator_test

