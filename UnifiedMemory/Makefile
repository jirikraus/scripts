all: run

managed_operator_new.o: Makefile managed_operator_new.cpp
	g++ -I$(CUDA_INC_PATH) -c -o managed_operator_new.o managed_operator_new.cpp

managed_allocator_test: Makefile managed_allocator_test.cu managed_operator_new.o
	nvcc -arch=sm_35 -o managed_allocator_test managed_allocator_test.cu managed_operator_new.o

run: managed_allocator_test
	./managed_allocator_test

clean:
	rm -f managed_allocator_test managed_operator_new.o

