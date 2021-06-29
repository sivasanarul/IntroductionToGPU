#include <stdio.h>
#include <time.h>        
#include <stdlib.h> 
#define array_size 100000000

int main(){

    //## ALLOCATE MEMORY FOR VARIABLES ##//
    

    for(int i = 0; i < array_size; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }    
    
    clock_t t; 
    t = clock();
    
    //## ADDITION OF VECTORS ##//
        
    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("fun() took %f seconds to execute \n", time_taken);     
        
    //## DEALLOCATE MEMORY##// 
      
}
