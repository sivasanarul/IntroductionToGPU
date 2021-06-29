#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define array_size 268435456

//## KERNEL FOR VECTOR ADDITION IN 1 THREAD ##//

extern double mysecond();

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 
    double t;   
    
    //## ALLOCATE MEMORY FOR VARIABLES AND INITIALIZE THE VARIABLES IN HOST ##//
    
    //## ALLOCATE MEMORY FOR VARIABLES IN DEVICE ##//
    
    t = mysecond();
    //## TRANSFER DATA FROM HOST TO DEVICE ##//
    t = (mysecond() - t);
    printf ("\nElapsed time for copy from host to device   = %g\n", t);
    
    t = mysecond();
    // Vector addition
    vector_add<<<1, 1>>>(d_out, d_a, d_b, array_size);
    cudaDeviceSynchronize();
    t = (mysecond() - t);
    printf ("\nElapsed time for vector addition in 1 thread = %g\n", t);    
        
    t = mysecond();
    //## TRANSFER DATA FROM DEVICE TO HOST ##//
    t = (mysecond() - t);
    printf ("\nElapsed time for copy from device to host   = %g\n", t);

    //## DEALLOCATE HOST AND DEVICE MEMORY ##//

    printf ("\nBLock size (number of thread) : 1 \n");
    printf ("\nNumber of blocks              : 1 \n");

}

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec  * 1.e-6);
}
