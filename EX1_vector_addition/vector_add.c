#include <stdio.h> 
#include <stdlib.h> 
#define N 100000000

int main(){
    float *a, *b, *c;

    a   = (float*)malloc(sizeof(float) * N); 
    b   = (float*)malloc(sizeof(float) * N);
    c   = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    
    
    
    clock_t t; 
    t = clock();
    // Main function
    for(int i=0;i<N;i++){
        c[i] = a[i] + b[i];}
    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("fun() took %f seconds to execute \n", time_taken);     
        
    free(a);  free(b); free(c);    

}
