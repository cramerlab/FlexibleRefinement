#pragma once
#ifndef PSEUDOATOMS
#define PSEUDOATOMS

#include "liblionImports.h"
#include "Types.h"
#include "funcs.h"
#include "Warp_GPU.h"
enum PseudoAtomMode { ATOM_GAUSSIAN=0, ATOM_INTERPOLATE=1 };


class Pseudoatoms {

public:
	std::vector<Matrix1D<DOUBLE>> AtomPositions;
	PseudoAtomMode Mode;
	Matrix1D<DOUBLE> GaussianTable;
	void RasterizeToVolume(MultidimArray<DOUBLE> &vol, int3 Dims, DOUBLE super);
	void IntensityFromVolume(MultidimArray<DOUBLE> &vol, DOUBLE super);
	std::vector< DOUBLE > AtomWeights;
	DOUBLE TableLength;
	DOUBLE Sigma;

	DOUBLE GaussFactor;

	/*
	Pseudoatoms(std::vector<Matrix1D<DOUBLE>> atomPositions, std::vector< DOUBLE > atomWeight, PseudoAtomMode = ATOM_INTERPOLATE, DOUBLE sigma=1.0) {
	
	
	}*/

	Pseudoatoms(PseudoAtomMode mode = ATOM_INTERPOLATE, DOUBLE sigma = 1.0, DOUBLE gaussFactor = 1.0):Mode(mode), Sigma(sigma), GaussFactor(gaussFactor){};

	Pseudoatoms(DOUBLE *atomPositionCArr, DOUBLE *atomWeights, idxtype nAtoms, PseudoAtomMode mode = ATOM_INTERPOLATE, DOUBLE sigma = 1.0, DOUBLE gaussFactor=1.0):Mode(mode), Sigma(sigma), GaussFactor(gaussFactor) {
		AtomPositions = std::vector<Matrix1D<DOUBLE>>();
		AtomPositions.reserve(nAtoms);

		AtomWeights.reserve(nAtoms);
		for (size_t i = 0; i < nAtoms; i++)
		{
			Matrix1D<DOUBLE> tmp = Matrix1D<DOUBLE>(3);
			XX(tmp) = atomPositionCArr[i * 3];
			YY(tmp) = atomPositionCArr[i * 3 + 1];
			ZZ(tmp) = atomPositionCArr[i * 3 + 2];
			AtomPositions.push_back(tmp);
			AtomWeights.push_back(atomWeights[i]);
		}



		DOUBLE sigma4 = 4 * sigma;
		TableLength = sigma4;
		GaussianTable = Matrix1D<DOUBLE>(CEIL(sigma4*sqrt(3) * GaussFactor + 1));

		FOR_ALL_ELEMENTS_IN_MATRIX1D(GaussianTable)
			GaussianTable(i) = gaussian1D(i / ((DOUBLE)GaussFactor), sigma);
		GaussianTable *= gaussian1D(0, sigma);
	}

};
#endif // !PSEUDOATOMS