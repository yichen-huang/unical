#include <iostream>
#include <math.h>

__global__ void vectorAdd(int n, float *a, float *b, float *c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) 
      c[i] = a[i] + b[i];
}

int main(void)
{
  int N = pow(2, 24);
  float *a, *b, *c, *d_a, *d_b, *d_c;
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));
  c = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_b, N*sizeof(float));
  cudaMalloc(&d_c, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 0.0f;
  }

  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, N*sizeof(float), cudaMemcpyHostToDevice);

  vectorAdd<<<1, 32>>>(N, a, b, c);

  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(c[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  
  return 0;
}