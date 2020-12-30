#define N 100000000

void vector_add(float *out, float *a, float *b, int n){
     for(int i=0;i<n;i++){
        out[i] = a[i] + b[i];}
}

int main(){
    float *a, *b, *out;

    a   = (float*)malloc(sizeof(float) * N); 
    a   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);

}
