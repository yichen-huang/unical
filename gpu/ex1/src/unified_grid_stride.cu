#include <iostream>
#include <math.h>

__global__ void vectorInit(int n, float *a, float value)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
      a[i] = value;
  }
}

__global__ void vectorAdd(int n, float *a, float *b, float *c)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      c[i] = a[i] + b[i];
}

int main(void)
{
  int N = pow(2, 24);
  float *a, *b, *c;

  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&b, N*sizeof(float));
  cudaMallocManaged(&c, N*sizeof(float));

  int blockSize = 32;
  int numBlocks = (N + blockSize - 1) / blockSize;

  vectorInit<<<numBlocks, blockSize>>>(N, a, 1.0f);
  vectorInit<<<numBlocks, blockSize>>>(N, b, 2.0f);
  vectorInit<<<numBlocks, blockSize>>>(N, c, 0.0f);

  vectorAdd<<<numBlocks, blockSize>>>(N, a, b, c);

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