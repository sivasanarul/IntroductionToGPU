#include<iostream>
#include<cstdlib>
#include<cmath>
#include<time.h>
using namespace std;


__global__ void matrixVectorMultiplication(float *a, float *mat, float *c, int n)
{
    int row=threadIdx.x+blockDim.x*blockIdx.x;
    float sum=0;
   
    if(row<n){
    for(int j=0;j<n;j++)
    {
        sum=sum+mat[row*n+j]*a[j];
    }
    }
    c[row]=sum;
}
int main()
{
    float *a, *b, *c, *d;
    float *dev_a, *dev_b, *dev_c;
    int n=32*1024;
    
    a=(float*)malloc(sizeof(float)*n);
    b=(float*)malloc(sizeof(float)*n*n);
    c=(float*)malloc(sizeof(float)*n);
    d=(float*)malloc(sizeof(float)*n);
    
    int i, j;
    for(i=0; i<n; i++)
          a[i] = 1.0;          
          c[i] = 1.0; 
    for(i=0; i<n; i++)
       for(j=0; j<n; j++)
            b[i*n+j] = 2.0;      

    printf("<<<<<<<<<< initial data:\n");
    
    cudaMalloc((void**)&dev_a, sizeof(float)*n);
    cudaMalloc((void**)&dev_b, sizeof(float)*n*n);
    cudaMalloc((void**)&dev_c, sizeof(float)*n);
    
    cudaMemcpy(dev_a, a, sizeof(float)*n,   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    
    cudaEvent_t start,end;
    
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    
    int threadsPerBlock;
    threadsPerBlock = 32;
    int blocksPerGrid;
    blocksPerGrid   = n/threadsPerBlock;
    cudaEventRecord(start);
    matrixVectorMultiplication<<<blocksPerGrid,threadsPerBlock>>>(dev_a,dev_b,dev_c,n);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float time=0.0;
    cudaEventElapsedTime(&time,start,end);
    
    cudaMemcpy(c,dev_c,sizeof(float)*n,cudaMemcpyDeviceToHost);
    cout<<"\nGPU Time Elapsed:  "<<time;
    
    int sum=0;
    for(int row=0;row<n;row++)
	    {
		sum=0;
		for(int col=0;col<n;col++)
		{
		      sum=sum+a[row*n+col]*b[col];  
		    
		}
	      d[row]=sum;
	    }
    //t=clock()-t;
    //cout<<"\nCPU Time Elapsed:  "<<((double)t);      //((double)t)/CLOCKS_PER_SEC;

    
    int error=0;
    for(int i=0;i<n;i++){
        error+=d[i]-c[i];
       // cout<<" gpu "<<c[i]<<" CPU "<<d[i]<<endl;
    }
    
    cout<<"Error : "<<error;
    
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
