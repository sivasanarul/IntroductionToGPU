#include <stdio.h>
#include <time.h>        
#include <stdlib.h> 
#define array_size 100000000

int main(){

    float *a, *b, *c;

    a   = (float*)malloc(sizeof(float) * array_size); 
    b   = (float*)malloc(sizeof(float) * array_size);
    c   = (float*)malloc(sizeof(float) * array_size);

    // Initialize array
    for(int i = 0; i < array_size; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }    
    
    clock_t t; 
    t = clock();
    
    // vector addition
    for(int i = 0;i < array_size; i++){
        c[i] = a[i] + b[i];}
        
    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("fun() took %f seconds to execute \n", time_taken);     
        
    free(a);  free(b); free(c); 
      
}
