#include <stdio.h>
#include<iostream>
#include <cuda.h>
#include <time.h>
# include <omp.h>
using namespace std;
__global__ void matvec(float *vec, float *mat, float *out, const int N, const int M){
    int index  = threadIdx.x;
    int stride = blockDim.x;

        float sum=0;
        for(int row=index; row<N; row+=stride){       
            sum = 0;
            for(int col = 0; col<N;col++){
                sum += vec[col]*mat[(row*N)+col];
                }
            out[row]=sum;
            }
        

}

extern double mysecond();
void init_array(float *a, const int N);
void init_mat(float *a, const int N, const int M);
void print_array(float *a, const int N, char *d);
void print_mat(float *a, const int N, const int M, char *d);

int main (void) {

    float *a, *b, *c, *d;
    float *dev_a, *dev_b, *dev_c;
    double t;

    
    int N= 32768;
    int M=N;
    a=(float*)malloc(sizeof(float)*N);
    b=(float*)malloc(sizeof(float)*N*M);
    c=(float*)malloc(sizeof(float)*M);
    d=(float*)malloc(sizeof(float)*M);
    
    init_array(a, N);
    init_mat(b, N, M);
    init_array(c, M);

    cudaMalloc((void**)&dev_a, sizeof(float)*N);
    cudaMalloc((void**)&dev_b, sizeof(float)*N*M);
    cudaMalloc((void**)&dev_c, sizeof(float)*M);
    
    int blocksize = 64; // value usually chosen by tuning and hardware constraints
    int nblocks   = N / blocksize;    
    cout<<"\nblocksize:  "<<((double)blocksize);
    cout<<"\nnblocks  :  "<<((double)nblocks);    
    
    printf("\n\nRunning Kernel...\n\n");
    t = mysecond();
    cudaMemcpy(dev_b, b, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    int iter; int niter; niter = 100;
    for (iter = 0; iter<niter;iter++)
	    {
	    cudaMemcpy(dev_a, a, sizeof(float)*N,   cudaMemcpyHostToDevice);
	    matvec<<<1, blocksize>>>(dev_a, dev_b, dev_c, N, M);
	    cudaMemcpy(c, dev_c, sizeof(float)*M, cudaMemcpyDeviceToHost);
	    }
    t = (mysecond() - t);
    printf ("\nElapsed seconds = %g\n", t );
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

       
    float sum=0;
    for(int row=0;row<N;row++)
	    {
		sum=0;
		for(int col=0;col<N;col++)
		{
		      sum=sum+b[row*N+col]*a[col];  
		    
		}
	      d[row]=sum;
	     } 
	    
    float error=0;
    for(int i=0;i<N;i++)
        error+=d[i]-c[i];
    
    cout<<"Error: "<<error;       
    cout<<"\n\n"; 
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
void print_array(float *a, const int N, char *d) {
        int i;
        for(i=0; i<N; i++)
                printf("\n%s[%d]: %f",d, i, a[i]);
    printf("\n");
}
void print_mat(float *a, const int N, const int M, char *d) {
        int i, j;
        for(i=0; i<N; i++){
        printf("\n%s[%d]:", d, i);
        for (j=0; j<M; j++)
                    printf("\t%6.4f", a[i*M+j]);
    }
    printf("\n");
}

#include <sys/time.h>
double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i;
    
    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec  * 1.e-6);
}
