#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

//## KERNEL FOR VECTOR ADDITION IN N STREAMING MULTIPROCESSOR ##//

extern double mysecond();
void init_array(float *a, const int N);
void init_mat(float *a, const int N, const int M);

int main (void) {

    float *a, *b, *c, *d;
    float *dev_a, *dev_b, *dev_c;
    
    double t;
    int N = 32768;
    int M = N;
    
    // Allocate host memory 
    a=(float*)malloc(sizeof(float)*N);
    b=(float*)malloc(sizeof(float)*N*M);
    c=(float*)malloc(sizeof(float)*M);
    d=(float*)malloc(sizeof(float)*M);

    // Initialize matrices    
    init_array(a, N);
    init_mat(b, N, M);
    init_array(c, M);

    // Allocate device memory
    cudaMalloc((void**)&dev_a, sizeof(float)*N);
    cudaMalloc((void**)&dev_b, sizeof(float)*N*M);
    cudaMalloc((void**)&dev_c, sizeof(float)*M);

    int block_size = 256; // value usually chosen by tuning and hardware constraints
    int nblocks = N / block_size;

    t = mysecond();
    cudaMemcpy(dev_a, a, sizeof(float)*N,   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    t = (mysecond() - t);
    printf ("\nElapsed time for copy from host to device   = %g\n", t);

    t = mysecond();
    // matrix vector product    
    matvec<<<nblocks, block_size>>>(dev_a, dev_b, dev_c, N, M);
    cudaDeviceSynchronize();
    t = (mysecond() - t);
    printf ("\nElapsed time for matrix vector product in n block = %g\n", t);

    t = mysecond();
    // Transfer data from device to host memory 
    cudaMemcpy(c, dev_c, sizeof(float)*M, cudaMemcpyDeviceToHost);
    t = (mysecond() - t);
    printf ("\nElapsed time for copy from device to host   = %g\n", t);
        
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // verify the kernel implementation      
    float sum = 0;
    for(int row = 0; row < N; row++)
	    {
		sum = 0;
		for(int col = 0; col < N; col++)
		{
		      sum = sum + b[row*N + col]*a[col];  
		    
		}
	      d[row] = sum;
	    } 
	    
    float error = 0;
    for(int i = 0; i < N; i++)
        error += d[i] - c[i];
  
    printf ("\nError   = %g\n", error ); 

    // Deallocate host memory
    free(a); 
    free(b); 
    free(c);
    free(d);

    printf ("\nBLock size (number of threads): %d \n", block_size);
    printf ("\nNumber of blocks              : %d \n", nblocks);
                
    return 0;
};

void init_array(float *a, const int N) {
        int i;
        for(i=0; i<N; i++)
                a[i] = 1.0;
}
void init_mat(float *a, const int N, const int M) {
        int i, j;
        for(i=0; i<N; i++)
            for(j=0; j<M; j++)
                    a[i*M+j] = 2.0;
}

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec  * 1.e-6);
}
