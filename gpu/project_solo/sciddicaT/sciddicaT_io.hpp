#ifndef __LOADER_H__
#define __LOADER_H__

#include "sciddicaT_project.hpp"

void readHeaderInfo(char *, int &, int &, double &);
bool loadGrid2D(double *, int, int, char *);
bool saveGrid2Dr(double *, int, int, char *);
double* addLayer2D(int, int);

#endif