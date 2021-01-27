#include <stdlib.h>   //malloc and free
#include <stdio.h>    //printf
#include <omp.h>      //OpenMP


int main (int argc, char *argv[]) 
{
 
	printf("Hello World... from thread = %d\n", 
			omp_get_num_threads());
	# pragma omp parallel
	printf("Hello World... from thread = %d\n", 
			omp_get_num_threads());
	{
	  printf("Thread rank: %d\n", omp_get_thread_num());
	}
	
	return 0;
}
