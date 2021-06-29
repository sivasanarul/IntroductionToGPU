#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define array_size 268435456

//## KERNEL FOR VECTOR ADDITION IN 1 STREAMING MULTIPROCESSOR ##//

extern double mysecond();

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;
    double t;
    
    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * array_size); 
    b   = (float*)malloc(sizeof(float) * array_size);
    out = (float*)malloc(sizeof(float) * array_size);

    // Initialize array
    for(int i = 0; i < array_size; i++){
        a[i] = 1.0f; 
        b[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_a,sizeof(float)*array_size);
    cudaMalloc((void**)&d_b,sizeof(float)*array_size);
    cudaMalloc((void**)&d_out,sizeof(float)*array_size);
 
    t = mysecond();
    // Transfer data from host to device memory
    cudaMemcpy(d_a,a, sizeof(float)*array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b, sizeof(float)*array_size, cudaMemcpyHostToDevice);
    t = (mysecond() - t);
    printf ("\nElapsed time for copy from host to device   = %g\n", t);
    
    int block_size = 256;
    t = mysecond();
    // Vector addition
    vector_add<<<1,block_size>>>(d_out, d_a, d_b, array_size);
    cudaDeviceSynchronize();
    t = (mysecond() - t);
    printf ("\nElapsed time for vector addition in 1 block = %g\n", t);  

    t = mysecond();
    // Transfer data from device to host memory    
    cudaMemcpy(out, d_out, sizeof(float)*array_size, cudaMemcpyDeviceToHost);
    t = (mysecond() - t);
    printf ("\nElapsed time for copy from device to host   = %g\n", t);
    
    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
    
    printf ("\nBLock size (number of threads): %d \n", block_size);
    printf ("\nNumber of blocks              : 1 \n");

}

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec  * 1.e-6);
}
