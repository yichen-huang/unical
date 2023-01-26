#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "sciddicaT_project.hpp"

void sciddicaTSimulationInit(int, int, int, int, double*, double*);
void sciddicaTResetFlows(int, int, int, int, double, double*);
void sciddicaTFlowsComputation(int, int, int, int, double, int*, int*, double*, double*, double*, double, double);
void sciddicaTWidthUpdate(int, int, int, int, double, int *, int *, double *, double *, double *);

#endif