#include <iostream>
#include <math.h>

__global__ void vectorInit(int n, float *x, float value)
{
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i += stride) {
      x[i] = value;
  }
}

__global__ void vectorAdd(int n, float *a, float *b, float *c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) 
      c[i] = a[i] + b[i];
}

int main(void)
{
  int N = pow(2, 24);
  float *a, *b, *c;

  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&b, N*sizeof(float));
  cudaMallocManaged(&c, N*sizeof(float));

  vectorInit<<<1, 32>>>(N, a, 1.0f);
  vectorInit<<<1, 32>>>(N, b, 2.0f);
  vectorInit<<<1, 32>>>(N, c, 0.0f);

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