#pragma once
#ifndef PSEUDOATOMS
#define PSEUDOATOMS
#include "AtomFitting.h"
#include "liblionImports.h"
#include "Types.h"
#include "funcs.h"
#include "Warp_GPU.h"
#include "Eigen\Core"

enum PseudoAtomMode { ATOM_GAUSSIAN=0, ATOM_INTERPOLATE=1 };


class Pseudoatoms {

public:
	std::vector<float3> AtomPositions;
	PseudoAtomMode Mode;
	Matrix1D<RDOUBLE> GaussianTable;
	void RasterizeToVolume(MultidimArray<RDOUBLE> &vol, int3 Dims, RDOUBLE super, bool resize=true);
	void IntensityFromVolume(MultidimArray<RDOUBLE> &vol, RDOUBLE super);
	std::vector< RDOUBLE > AtomWeights;
	RDOUBLE TableLength;
	RDOUBLE Sigma;
	idxtype NAtoms;

	RDOUBLE GaussFactor;

	/*
	Pseudoatoms(std::vector<Matrix1D<RDOUBLE>> atomPositions, std::vector< RDOUBLE > atomWeight, PseudoAtomMode = ATOM_INTERPOLATE, RDOUBLE sigma=1.0) {
	
	
	}*/

	Pseudoatoms(PseudoAtomMode mode = ATOM_INTERPOLATE, RDOUBLE sigma = 1.0, RDOUBLE gaussFactor = 1.0):Mode(mode), Sigma(sigma), GaussFactor(gaussFactor), NAtoms(0) {};

	Pseudoatoms(RDOUBLE *atomPositionCArr, RDOUBLE atomWeight, idxtype nAtoms, PseudoAtomMode mode = ATOM_INTERPOLATE, RDOUBLE sigma = 1.0, RDOUBLE gaussFactor = 1.0) :Mode(mode), Sigma(sigma), GaussFactor(gaussFactor), NAtoms(nAtoms) {
		AtomPositions = std::vector<float3>();
		AtomPositions.reserve(nAtoms);

		AtomWeights.reserve(nAtoms);
		for (size_t i = 0; i < nAtoms; i++)
		{
			float3 tmp = { atomPositionCArr[i * 3], atomPositionCArr[i * 3 + 1], atomPositionCArr[i * 3 + 2] };
			AtomPositions.push_back(tmp);
			AtomWeights.push_back(atomWeight);
		}


		if (mode == ATOM_GAUSSIAN) {
			RDOUBLE sigma4 = 4 * sigma;
			TableLength = sigma4;
			GaussianTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GaussFactor + 1));

			FOR_ALL_ELEMENTS_IN_MATRIX1D(GaussianTable)
				GaussianTable(i) = gaussian1D(i / ((RDOUBLE)GaussFactor), sigma);
			GaussianTable *= gaussian1D(0, sigma);
		}
	}

	Pseudoatoms(RDOUBLE *atomPositionCArr, RDOUBLE *atomWeights, idxtype nAtoms, PseudoAtomMode mode = ATOM_INTERPOLATE, RDOUBLE sigma = 1.0, RDOUBLE gaussFactor=1.0):Mode(mode), Sigma(sigma), GaussFactor(gaussFactor), NAtoms(nAtoms) {
		AtomPositions = std::vector<float3>();
		AtomPositions.reserve(nAtoms);

		AtomWeights.reserve(nAtoms);
		for (size_t i = 0; i < nAtoms; i++)
		{
			float3 tmp = { atomPositionCArr[i * 3], atomPositionCArr[i * 3 + 1], atomPositionCArr[i * 3 + 2] };
			AtomPositions.push_back(tmp);
			AtomWeights.push_back(atomWeights[i]);
		}


		if (mode == ATOM_GAUSSIAN) {
			RDOUBLE sigma4 = 4 * sigma;
			TableLength = sigma4;
			GaussianTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GaussFactor + 1));

			FOR_ALL_ELEMENTS_IN_MATRIX1D(GaussianTable)
				GaussianTable(i) = gaussian1D(i / ((RDOUBLE)GaussFactor), sigma);
			GaussianTable *= gaussian1D(0, sigma);
		}
	}

	void MoveAtoms(MultidimArray<RDOUBLE>& superRefVol, int3 Dims, RDOUBLE super, bool resize, float limit);
	void MoveAtoms(MultidimArray<RDOUBLE>& superRefVol, int3 Dims, RDOUBLE super, bool resize, double limit, ADAMParams * adamparams);

	static idxtype readAtomsFromFile(FileName pdbFile, std::vector<float3> &AtomPositions, std::vector<RDOUBLE> &AtomIntensities, idxtype N=100000) {
		std::ifstream ifs(pdbFile);
		if (ifs.fail()) {
			std::cerr << "Failed to open " << pdbFile << std::endl;
			exit(EXIT_FAILURE);
		}
		std::string line;
		double sigma = 0.0;

		AtomPositions.clear();
		AtomPositions.reserve(N);

		AtomIntensities.clear();
		AtomIntensities.reserve(N);

		idxtype NAtoms = 0;
		while (std::getline(ifs, line)) {
			if (line[0] == 'R' && line.rfind("REMARK fixedGaussian") != std::string::npos) {
				sscanf(line.c_str(), "REMARK fixedGaussian %lf\n", &sigma);
			}
			if (line.rfind("ATOM") != std::string::npos) {
				float3 atom;
				Matrix1D<RDOUBLE> position(3);
				double intensity;
				sscanf(line.c_str(), "ATOM\t%*d\tDENS\tDENS\t%*d\t%f\t%f\t%f\t%*d\t%lf\tDENS", &(atom.x), &(atom.y), &(atom.z), &intensity);
				AtomIntensities.emplace_back((RDOUBLE)intensity);
				AtomPositions.emplace_back(atom);
				NAtoms++;
			}
		}
		return NAtoms;
	}

	double operator() (Eigen::VectorXd positions, Eigen::VectorXd& grad);

};
#endif // !PSEUDOATOMS