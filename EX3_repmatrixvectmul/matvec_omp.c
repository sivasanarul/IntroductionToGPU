# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
#include <sys/time.h>

int main(void);

extern double mysecond();

int main ( void ){
  long int Nrow; Nrow = 32768;
  long int mat_size = Nrow*Nrow;
  float * A;
  float * b; 
  float * x;
  A   = (float*)malloc(sizeof(float) * mat_size); 
  x   = (float*)malloc(sizeof(float) * Nrow);
  b   = (float*)malloc(sizeof(float) * Nrow);

  long int i;
  long int j;
  long int k;
  long int tid;
  float sum;
  int thread_num;
  double wtime;
  long int n_per_thread; 
  int total_threads;
  


  printf ( "\n" );
  printf ( "MXV_OPENMP:\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  Compute matrix Vector product b = A * x.\n" );
  
  # pragma omp parallel
  total_threads = omp_get_num_threads();
  n_per_thread  = mat_size/total_threads;
  
  printf ( "\n" );
  printf ( "  The number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  The number of threads available    = %d\n", total_threads );
  printf ( "  The number of element per thread   = %d\n", n_per_thread );
  printf ( "  The matrix size N,N                = %d, %d\n", Nrow, Nrow );

  for ( i = 0; i < mat_size; i++ )
	A[i] = 1.0;   

  for ( i = 0; i < Nrow; i++ )
	x[i] = 1.0;    

  wtime = omp_get_wtime ( );
  double t;
  t = mysecond();
    int iter; int niter; niter = 100;
    for (iter = 0; iter<niter;iter++)
            { 
		#pragma omp parallel for 
		for (i=0; i<Nrow; i++)
		{
			sum = 0.0;
			#pragma omp simd reduction(+:sum)
			for (j=0; j<Nrow; j++)
			{
				sum+= A[i*Nrow  + j]*x[j];
			}
			b[i] =sum;
		}
	     } 
  
  t = (mysecond() - t);
  printf ("\nElapsed seconds = %g\n", t );  
  wtime = omp_get_wtime ( ) - wtime;
  printf ( "  Elapsed seconds ompwtime = %g\n", wtime );
 
  free(A);  free(b); free(x);
  printf ( "\n" );
  printf ( "MXM_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );
  
  printf ( "\n" );


  return 0;
}

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec  * 1.e-6);
}

