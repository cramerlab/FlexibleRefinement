#pragma once
#ifndef  PSEUDO_PROJECTOR
#define PSEUDO_PROJECTOR
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include "liblionImports.h"
#include "Types.h"
#include "readMRC.h"
#include "macros.h"
#include "funcs.h"
#include "pseudoatoms.h"
#include "Prerequisites.cuh"

constexpr auto GAUSS_FACTOR = 30;

using namespace relion;

constexpr auto PSEUDO_FORWARD = 1;
constexpr auto PSEUDO_BACKWARD = -1;


struct projecction {
	std::vector< Matrix1D<RDOUBLE> > *atomPositons;
	MultidimArray<RDOUBLE> *image;
	Matrix1D<RDOUBLE> angles;
	Matrix2D<RDOUBLE> Euler;
};

class PseudoProjector {

public:
	// Gaussian projection table
	Matrix1D<RDOUBLE> gaussianProjectionTable;

	// Gaussian projection2 table
	Matrix1D<RDOUBLE> gaussianProjectionTable2;

	PseudoAtomMode mode;

	Pseudoatoms *atoms;
	Pseudoatoms *ctfAtoms;

	//std::vector< Matrix1D<RDOUBLE> > atomPositions;
	
	// Atomic weights
	//std::vector< RDOUBLE > atomWeight;
	double sigma;
	double super;
	double tableLength;
	RDOUBLE lambdaART;
	int3 Dims;
	bool gauss2D;


	//PseudoProjector with Gaussian mode
	PseudoProjector(int3 dims, RDOUBLE *atomPositionCArr, RDOUBLE *atomWeights, RDOUBLE sigma, RDOUBLE super, unsigned int nAtoms):Dims(dims), sigma(sigma), super(super),mode(ATOM_GAUSSIAN), lambdaART(0.1){	
		atoms = new Pseudoatoms(atomPositionCArr, atomWeights, nAtoms, ATOM_GAUSSIAN, sigma, GAUSS_FACTOR);
		RDOUBLE sigma4 = 4 * sigma;
		tableLength = sigma4;
		gaussianProjectionTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		gaussianProjectionTable2 = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
			gaussianProjectionTable(i) = gaussian1D(i / ((RDOUBLE)GAUSS_FACTOR), sigma);
		gaussianProjectionTable *= gaussian1D(0, sigma);
		gaussianProjectionTable2 = gaussianProjectionTable;
		gaussianProjectionTable2 *= gaussianProjectionTable;
	};

	PseudoProjector(int3 dims, RDOUBLE *atomPositionCArr, RDOUBLE *atomWeights, RDOUBLE super, unsigned int nAtoms) :Dims(dims), sigma(0), super(super), mode(ATOM_INTERPOLATE), lambdaART(0.1) {
		atoms = new Pseudoatoms(atomPositionCArr, atomWeights, nAtoms, ATOM_INTERPOLATE);
		ctfAtoms = new Pseudoatoms(atomPositionCArr, 0.0f, nAtoms, ATOM_INTERPOLATE);
	};

	PseudoProjector(int3 dims, Pseudoatoms *p_atoms, RDOUBLE super) :Dims(dims), sigma(0), super(super), mode(ATOM_INTERPOLATE), lambdaART(0.1) {
		atoms = new Pseudoatoms(p_atoms);
		ctfAtoms = new Pseudoatoms(p_atoms);
	};

	RDOUBLE ART_single_image(const MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &Itheo, MultidimArray<RDOUBLE> &Icorr, MultidimArray<RDOUBLE> &Idiff, RDOUBLE rot, RDOUBLE tilt, RDOUBLE psi, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE ART_single_image(const MultidimArray<RDOUBLE> &Iexp, RDOUBLE rot, RDOUBLE tilt, RDOUBLE psi, RDOUBLE shiftX, RDOUBLE shiftY);


	RDOUBLE ART_batched(const MultidimArray<RDOUBLE> &Iexp, idxtype batchSize, float3 *angles, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE ART_batched(const MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &Itheo, MultidimArray<RDOUBLE> &Icorr, MultidimArray<RDOUBLE> &Idiff, idxtype batchSize, float3 *angles, RDOUBLE shiftX, RDOUBLE shiftY);


	RDOUBLE ART_multi_Image_step(RDOUBLE * Iexp, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
	RDOUBLE ART_multi_Image_step(RDOUBLE * Iexp, float3 * angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
	RDOUBLE ART_multi_Image_step(std::vector< MultidimArray<RDOUBLE> > Iexp, std::vector<float3> angles, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE ART_multi_Image_step_DB(RDOUBLE * Iexp, RDOUBLE * Itheo, RDOUBLE * Icorr, RDOUBLE * Idiff, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE tableLength, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);

	MRCImage<RDOUBLE> *create3DImage(RDOUBLE oversampling = 1.0);

	void createMask(float3 *angles, int* positionMatching, MultidimArray<RDOUBLE>* CTFs, MultidimArray<RDOUBLE>& projections, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE CTFSIRT(MultidimArray<RDOUBLE> &CTFs, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE CTFSIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, int *positionMatching, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY);

	void projectForward(float3 *angles, int* positionMatching, MultidimArray<RDOUBLE>* ctfs, MultidimArray<RDOUBLE>& projections, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY);
	
	void project_Pseudo(RDOUBLE * out,
		RDOUBLE * out_nrm,
		float3 angles,
		RDOUBLE shiftX,
		RDOUBLE shiftY,
		int direction);


	void project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj,
		Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY,
		int direction);

	void project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj, std::vector<Matrix1D<RDOUBLE>> * atomPositions,
		Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY,
		int direction);


	void setAtomPositions(std::vector<float3> newPositions) {
		atoms->AtomPositions = newPositions;
	}

	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* Inorm, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, int *positionMapping, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* Inorm, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &CTFs, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &CTFs, float3 *angles, int *positionMatching, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE VolumeUpdate(MultidimArray<RDOUBLE> &Volume, RDOUBLE shiftX, RDOUBLE shiftY);

};

#endif // ! PSEUDO_PROJECTOR

