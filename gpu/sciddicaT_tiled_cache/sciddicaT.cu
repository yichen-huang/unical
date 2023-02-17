#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// CUDA functions
// ----------------------------------------------------------------------------

#define TILE_WIDTH 32
#define CUDA_GET(M, i, j) (M[(i)][(j)])

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;

  if ( (f = fopen(path,"r") ) == 0){
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Reading the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }
  }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double* cuda_addLayer2D(int rows, int columns)
{
  double *tmp;
  checkCuda(cudaMallocManaged(&tmp, sizeof(double) * rows * columns));

  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// cuda init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__global__ void cuda_sciddicaTSimulationInit(int r, int c, double* Sz, double* Sh)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row_step = blockDim.y * gridDim.y;
  int col_step = blockDim.x * gridDim.x;

  double z, h;

  for (int i = row+1; i < r-1; i += row_step)
    for (int j = col+1; j < c-1; j += col_step){
      h = GET(Sh, c, i, j);

      if (h > 0.0) {
        z = GET(Sz, c, i, j);
        SET(Sz, c, i, j, z - h);
      }
    }
}

// ----------------------------------------------------------------------------
// cuda computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

__global__ void cuda_sciddicaTResetFlows(int r, int c, double nodata, double* Sf)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row_step = blockDim.y * gridDim.y;
  int col_step = blockDim.x * gridDim.x;

  for (int i = row+1; i < r-1; i += row_step)
    for (int j = col+1; j < c-1; j += col_step)
      for (int step = 0; step < 4; step++)
        BUF_SET(Sf, r, c, step, i, j, 0.0);

}

__global__ void cuda_sciddicaTFlowsComputation(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  int start_i = blockIdx.y * blockDim.y;
  int start_j = blockIdx.x * blockDim.x;

  int next_i = (blockIdx.y + 1) * blockDim.y;
  int next_j = (blockIdx.x + 1) * blockDim.x;

  __shared__ double Sz_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Sh_tile[TILE_WIDTH][TILE_WIDTH];

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  if (row <= 0 || row >= r-1 || col <= 0 || col >= c-1) return;

  Sz_tile[threadIdx.y][threadIdx.x] = GET(Sz, c, row, col);
  Sh_tile[threadIdx.y][threadIdx.x] = GET(Sh, c, row, col);
  __syncthreads();

  m = CUDA_GET(Sh_tile, threadIdx.y, threadIdx.x) - p_epsilon;
  u[0] = CUDA_GET(Sz_tile, threadIdx.y, threadIdx.x) + p_epsilon;

  for (int step = 1; step <= 4; step++) {
    int i = row + Xi[step];
    int j = col + Xj[step];

    if (i < r && j < c) {
      if (i >= start_i && i < next_i && j >= start_j && j < next_j) {
        z = CUDA_GET(Sz_tile, threadIdx.y+Xi[step], threadIdx.x+Xj[step]);
        h = CUDA_GET(Sh_tile, threadIdx.y+Xi[step], threadIdx.x+Xj[step]);
      }
      else {
        z = GET(Sz, c, i, j);
        h = GET(Sh, c, i, j);
      }
      u[step] = z + h;
    }
  }

  do {
    again = false;
    average = m;
    cells_count = 0;

    for (n = 0; n < 5; n++)
      if (!eliminated_cells[n]) {
        average += u[n];
        cells_count++;
      }

    if (cells_count != 0)
      average /= cells_count;

    for (n = 0; n < 5; n++)
      if ((average <= u[n]) && (!eliminated_cells[n])) {
        eliminated_cells[n] = true;
        again = true;
      }
  } while (again);

  for (int step = 1; step <= 4; step++)
    if (!eliminated_cells[step]) BUF_SET(Sf, r, c, step-1, row, col, (average - u[step]) * p_r);
}

 __global__ void cuda_sciddicaTWidthUpdate(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  int start_j = blockIdx.x * blockDim.x;
  int start_i = blockIdx.y * blockDim.y;

  int next_j = (blockIdx.x + 1) * blockDim.x;
  int next_i = (blockIdx.y + 1) * blockDim.y;

  __shared__ double Sf_tile[TILE_WIDTH * ADJACENT_CELLS][TILE_WIDTH];

  double h_next;
  int buf_count = 4;

  if (row <= 0 || row >= r-1 || col <= 0 || col >= c-1) return;

  for(int step = 0; step < 4; step++)
    Sf_tile[threadIdx.y + step * TILE_WIDTH][threadIdx.x] = BUF_GET(Sf, r, c, step, row, col);
  __syncthreads();

  h_next = GET(Sh, c, row, col);

  for (int step = 1; step <= 4; step++) {
    int i = row + Xi[step];
    int j = col + Xj[step];

    if (i < r && j < c)
      if (i >= start_i && i < next_i && j >= start_j && j < next_j)
        h_next += CUDA_GET(Sf_tile, threadIdx.y+Xi[step]+(buf_count-step)*TILE_WIDTH, threadIdx.x+Xj[step]) - CUDA_GET(Sf_tile, threadIdx.y + (step-1) * TILE_WIDTH, threadIdx.x) ;
      else
        h_next += BUF_GET(Sf, r, c, buf_count-step, i, j) - BUF_GET(Sf, r, c, step-1, row, col);
  }
  SET(Sh, c, row, col, h_next);
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;                  // r: grid rows
  int c = cols;                  // c: grid columns
  //int i_start = 1, i_end = r-1;  // [i_start,i_end[: kernels application range along the rows
  //int j_start = 1, j_end = c-1;  // [i_start,i_end[: kernels application range along the rows
  //double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
  //double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
  //double *Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
  int *Xi;                       // Xj: von Neuman neighborhood row coordinates (see below)
  int *Xj;                       // Xj: von Neuman neighborhood col coordinates (see below)
  double p_r = P_R;              // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;  // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps

  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
  //   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //               |0:1:(-1, 0)|
  //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
  //               |3:4:( 1, 0)|
  //
  //

  int n = rows * cols;

  dim3 dimGrid(ceil(n / (TILE_WIDTH * TILE_WIDTH)), ceil(n / (TILE_WIDTH * TILE_WIDTH)), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  double *cuda_Sz;
  double *cuda_Sh;
  double *cuda_Sf;

  printf("Initialising variables...\n");

  cudaMallocManaged(&Xi, sizeof(int) * 5);
  cudaMallocManaged(&Xj, sizeof(int) * 5);

  Xi[0] = 0; Xi[1] = -1; Xi[2] = 0;  Xi[3] = 0; Xi[4] = 1;
  Xj[0] = 0; Xj[1] = 0;  Xj[2] = -1; Xj[3] = 1; Xj[4] = 0;

  printf("Initialising memory...\n");
  cuda_Sz = cuda_addLayer2D(r, c);                 // Allocates the Sz substate grid
  cuda_Sh = cuda_addLayer2D(r, c);                 // Allocates the Sh substate grid
  cuda_Sf = cuda_addLayer2D(ADJACENT_CELLS* r, c); // Allocates the Sf substates grid,
                                         //   having one layer for each adjacent cell

  printf("Loading memory...\n");
  loadGrid2D(cuda_Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
  loadGrid2D(cuda_Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
  cuda_sciddicaTSimulationInit<<<dimGrid, dimBlock>>>(r, c, cuda_Sz, cuda_Sh);
  checkCuda(cudaGetLastError());

  printf("Loop start ...\n");
  util::Timer cl_timer;
  // simulation loop
  for (int s = 0; s < steps; ++s)
  {
    // Apply the resetFlow kernel to the whole domain
    cuda_sciddicaTResetFlows<<<dimGrid, dimBlock>>>(r, c, nodata, cuda_Sf);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());

    // Apply the FlowComputation kernel to the whole domain
    cuda_sciddicaTFlowsComputation<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, cuda_Sz, cuda_Sh, cuda_Sf, p_r, p_epsilon);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());

    // Apply the WidthUpdate mass balance kernel to the whole domain
    cuda_sciddicaTWidthUpdate<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, cuda_Sz, cuda_Sh, cuda_Sf);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(cuda_Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file

  printf("Releasing memory...\n");

  cudaFree(cuda_Sz);
  cudaFree(cuda_Sh);
  cudaFree(cuda_Sf);

  cudaFree(Xi);
  cudaFree(Xj);

  return 0;
}
