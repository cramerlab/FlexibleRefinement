#pragma once


#include "liblionImports.h"
#include "Types.h"

using namespace relion;

void drawOneGaussian(MultidimArray<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, idxtype gaussFactor = 1000);

void drawOneGaussian(Matrix1D<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, idxtype gaussFactor = 1000);


