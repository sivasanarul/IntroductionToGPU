#include <stdlib.h>   //malloc and free
#include <stdio.h>    //printf
#include <omp.h>      //OpenMP
#include <time.h>
// Very small values for this simple illustrative example
#define N 640000000     //Size of arrays whose elements will be added together.

/*
 *  Classic vector addition using openMP default data decomposition.
 *
 *  Compile using gcc like this:
 *  	gcc -o va-omp-simple VA-OMP-simple.c -fopenmp
 *
 *  Execute:
 *  	./va-omp-simple
 */
int main (int argc, char *argv[]) 
{
	// elements of arrays a and b will be added
	// and placed in array c
	float * a;
	float * b; 
	float * c;
        
	int n = N;                       // number of array elements
	int n_per_thread;                         // elements per thread
	int total_threads;
	# pragma omp parallel
	total_threads = omp_get_num_threads();    // number of threads to use  
	printf("Hello World... from thread = %d\n", 
			total_threads);
	int i;       // loop index
        
        // allocate spce for the arrays
	a = (float*)malloc(sizeof(float)*n);
	b = (float*)malloc(sizeof(float)*n);
	c = (float*)malloc(sizeof(float)*n);

        // initialize arrays a and b with consecutive integer values
	// as a simple example
        for(i=0; i<n; i++) {
            a[i] = 1.0;
        }
        for(i=0; i<n; i++) {
            b[i] = 2.0;
        }   
        
        
        
	n_per_thread = n/total_threads;
	clock_t t; 
        t = clock();
	// determine how many elements each process will work on
	 
        // We compute the vector addition
	#pragma omp parallel for shared(a, b, c) private(i) schedule(static, n_per_thread)
        for(i=0; i<n; i++) {
		c[i] = a[i]+b[i];
        }
	t = clock() - t; 
       double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  
       printf("fun() took %f seconds to execute \n", time_taken); 
       
       
       
       
        // clean up memory
        free(a);  free(b); free(c);
	
	return 0;
}
