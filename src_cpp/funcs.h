#pragma once
#ifndef FUNCS
#define FUNCS



#include "liblionImports.h"
#include "Types.h"
#include "macros.h"

using namespace relion;

void drawOneGaussian(MultidimArray<RDOUBLE> &gaussianTable, RDOUBLE boundary, RDOUBLE k, RDOUBLE i, RDOUBLE j, MultidimArray<RDOUBLE> &V, RDOUBLE intensity, RDOUBLE gaussFactor = 1000);

void drawOneGaussian(Matrix1D<RDOUBLE> &gaussianTable, RDOUBLE boundary, RDOUBLE k, RDOUBLE i, RDOUBLE j, MultidimArray<RDOUBLE> &V, RDOUBLE intensity, RDOUBLE gaussFactor = 1000);

void Uproject_to_plane(const Matrix1D<RDOUBLE> &r,
	const Matrix2D<RDOUBLE> &euler, Matrix1D<RDOUBLE> &result);

void Uproject_to_plane(const Matrix1D<RDOUBLE> &point,
	const Matrix1D<RDOUBLE> &direction, RDOUBLE distance,
	Matrix1D<RDOUBLE> &result);

void writeFSC(MultidimArray<RDOUBLE> &V1, MultidimArray<RDOUBLE> &V2, FileName outpath);

RDOUBLE Lerp(RDOUBLE a, RDOUBLE b, RDOUBLE x);

//RDOUBLE gaussian1D(RDOUBLE x, RDOUBLE sigma, RDOUBLE mu = 0);
#endif // !FUNCS