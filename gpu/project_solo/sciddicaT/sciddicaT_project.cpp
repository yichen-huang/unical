#include "sciddicaT_project.hpp"
#include "sciddicaT_io.hpp"
#include "sciddicaT_compute.hpp"

// ----------------------------------------------------------------------------
// Function init()
// Local init
// ----------------------------------------------------------------------------
static sciddicat::variable init_struct(int argc, char** argv) {
  sciddicat::variable var;

  readHeaderInfo(argv[HEADER_PATH_ID], var.rows, var.cols, var.nodata);
  var.c = var.rows;
  var.c = var.cols;

  var.i_start = 1;
  var.i_end = var.r-1;
  var.j_start = 1;
  var.j_end = var.c-1;

  var.Xi = {0, -1,  0,  0,  1};
  var.Xj = {0,  0, -1,  1,  0};

  var.p_r = P_R;
  var.p_epsilon = P_EPSILON;
  var.steps = atoi(argv[STEPS_ID]);

  var.Sz = addLayer2D(var.r, var.c);
  var.Sh = addLayer2D(var.r, var.c);
  var.Sf = addLayer2D(ADJACENT_CELLS* var.r, var.c);

  loadGrid2D(var.Sz, var.r, var.c, argv[DEM_PATH_ID]);
  loadGrid2D(var.Sh, var.r, var.c, argv[SOURCE_PATH_ID]);

  return var;
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  /*
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;
  int c = cols;
  int i_start = 1, i_end = r-1;
  int j_start = 1, j_end = c-1;
  double *Sz;
  double *Sh;
  double *Sf;
  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};
  double p_r = P_R;
  double p_epsilon = P_EPSILON;
  int steps = atoi(argv[STEPS_ID]);

  Sz = addLayer2D(r, c);
  Sh = addLayer2D(r, c);
  Sf = addLayer2D(ADJACENT_CELLS* r, c);

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);
  */
 sciddicat::variable var = init_struct(argc, argv);


  #pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        sciddicaTSimulationInit(i, j, r, c, Sz, Sh);

  util::Timer cl_timer;

  for (int s = 0; s < steps; ++s) {
    #pragma omp parallel for
      for (int i = i_start; i < i_end; i++)
        for (int j = j_start; j < j_end; j++)
          sciddicaTResetFlows(i, j, r, c, nodata, Sf);

    #pragma omp parallel for
      for (int i = i_start; i < i_end; i++)
        for (int j = j_start; j < j_end; j++)
          sciddicaTFlowsComputation(i, j, r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);

    #pragma omp parallel for
      for (int i = i_start; i < i_end; i++)
        for (int j = j_start; j < j_end; j++)
          sciddicaTWidthUpdate(i, j, r, c, nodata, Xi, Xj, Sz, Sh, Sf);
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file

  printf("Releasing memory...\n");
  delete[] Sz;
  delete[] Sh;
  delete[] Sf;

  return 0;
}
