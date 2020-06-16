#pragma once
#ifndef FUNCS
#define FUNCS



#include "liblionImports.h"
#include "Types.h"
#include "macros.h"

using namespace relion;

void drawOneGaussian(MultidimArray<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, DOUBLE gaussFactor = 1000);

void drawOneGaussian(Matrix1D<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, DOUBLE gaussFactor = 1000);

void Uproject_to_plane(const Matrix1D<DOUBLE> &r,
	const Matrix2D<DOUBLE> &euler, Matrix1D<DOUBLE> &result);

void Uproject_to_plane(const Matrix1D<DOUBLE> &point,
	const Matrix1D<DOUBLE> &direction, DOUBLE distance,
	Matrix1D<DOUBLE> &result);

void writeFSC(MultidimArray<DOUBLE> &V1, MultidimArray<DOUBLE> &V2, FileName outpath);

DOUBLE Lerp(DOUBLE a, DOUBLE b, DOUBLE x);

//DOUBLE gaussian1D(DOUBLE x, DOUBLE sigma, DOUBLE mu = 0);
#endif // !FUNCS