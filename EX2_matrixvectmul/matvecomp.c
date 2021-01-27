# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

int main ( void );
void timestamp ( void );

/******************************************************************************/

int main ( void )

/******************************************************************************/
/*
  Purpose:


*/
{
  long int Nrow; Nrow = 90000;
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
  
  timestamp ( );

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

  wtime = omp_get_wtime ( );

  for ( i = 0; i < mat_size; i++ )
  {  
      A[i] = 1.0;   
  }

  for ( i = 0; i < Nrow; i++ )
  {   
      x[i] = 1.0;    
  }
  

  
 
  wtime = omp_get_wtime ( );
  
/*	#pragma omp parallel shared(A, b, x, Nrow) private(i, j, sum, tid)
	{
		tid = omp_get_thread_num();
		printf("Hello World... from thread = %d\n", 
			tid);*/
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

	//} 
  
    
  wtime = omp_get_wtime ( ) - wtime;
  printf ( "  Elapsed seconds = %g\n", wtime );
  printf ( "  x(100)  = %g\n", b[100-1] );
/*
  Terminate.
*/
  // clean up memory
  free(A);  free(b); free(x);
  printf ( "\n" );
  printf ( "MXM_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );
  
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

