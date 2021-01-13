#pragma once
#ifndef PSEUDOATOMS
#define PSEUDOATOMS
#include "AtomFitting.h"
#include <assert.h>
#include "liblionImports.h"
#include "Types.h"
#include "funcs.h"
#include "Warp_GPU.h"
#include <queue>
#include <list>

enum PseudoAtomMode { ATOM_GAUSSIAN=0, ATOM_INTERPOLATE=1 };


class Pseudoatoms {

public:
	std::vector<float3> AtomPositions;
	std::vector<float *> alternativePositions;
	PseudoAtomMode Mode;
	Matrix1D<RDOUBLE> GaussianTable;
	void RasterizeToVolume(MultidimArray<RDOUBLE> &vol, int3 Dims, RDOUBLE super, bool resize=true, bool weighting=false);
	void IntensityFromVolume(MultidimArray<RDOUBLE> &vol, RDOUBLE super);
	std::vector< RDOUBLE > AtomWeights;

	RDOUBLE TableLength;
	RDOUBLE Sigma;
	idxtype NAtoms;

	RDOUBLE GaussFactor;
	std::list<int> *grid;
	std::vector<int> *neighbours;
	std::vector<float> *neighbour_dists;
	int3 gridDims;

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

		addAlternativeOrientation(atomPositionCArr, NAtoms);

		if (mode == ATOM_GAUSSIAN) {
			RDOUBLE sigma4 = 4 * sigma;
			TableLength = sigma4;
			GaussianTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GaussFactor + 1));

			FOR_ALL_ELEMENTS_IN_MATRIX1D(GaussianTable)
				GaussianTable(i) = gaussian1D(i / ((RDOUBLE)GaussFactor), sigma);
			GaussianTable *= gaussian1D(0, sigma);
		}
	}

	Pseudoatoms(Pseudoatoms *other, RDOUBLE atomWeight, PseudoAtomMode mode = ATOM_INTERPOLATE, RDOUBLE sigma = 1.0, RDOUBLE gaussFactor = 1.0) :Mode(mode), Sigma(sigma), GaussFactor(gaussFactor){
		NAtoms = other->NAtoms;
		AtomPositions = other->AtomPositions;
		AtomPositions.reserve(NAtoms);

		AtomWeights.reserve(NAtoms);
		for (size_t i = 0; i < NAtoms; i++)
		{
			AtomWeights.push_back(atomWeight);
		}
		for (size_t i = 0; i < other->alternativePositions.size(); i++)
		{
			addAlternativeOrientation(other->alternativePositions[i], NAtoms);
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

	Pseudoatoms(Pseudoatoms *other, PseudoAtomMode mode = ATOM_INTERPOLATE, RDOUBLE sigma = 1.0, RDOUBLE gaussFactor = 1.0) :Mode(mode), Sigma(sigma), GaussFactor(gaussFactor) {
		NAtoms = other->NAtoms;
		AtomPositions = other->AtomPositions;
		AtomWeights = other->AtomWeights;

		for (size_t i = 0; i < other->alternativePositions.size(); i++)
		{
			addAlternativeOrientation(other->alternativePositions[i], NAtoms);
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

		addAlternativeOrientation(atomPositionCArr, NAtoms);

		if (mode == ATOM_GAUSSIAN) {
			RDOUBLE sigma4 = 4 * sigma;
			TableLength = sigma4;
			GaussianTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GaussFactor + 1));

			FOR_ALL_ELEMENTS_IN_MATRIX1D(GaussianTable)
				GaussianTable(i) = gaussian1D(i / ((RDOUBLE)GaussFactor), sigma);
			GaussianTable *= gaussian1D(0, sigma);
		}
	}

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

	static Pseudoatoms* ReadTsvFile(FileName tsvFileName) {
		FILE* ifs = fopen(tsvFileName.c_str(), "r");
		if (ifs == NULL) {
			std::cerr << "Failed to open " << tsvFileName << std::endl;
			exit(EXIT_FAILURE);
		}
		char buffer[1000];
		int N;
		fscanf(ifs, "%d\n", &N);
		float *AtomPositions = (float*)malloc(sizeof(*AtomPositions) * N * 3);
		float *AtomWeights = (float*)malloc(sizeof(*AtomWeights) * N);
		fscanf(ifs, "%s\n", buffer);
		if (!(buffer[0] == '#')) {
			std::cerr << "Invalid line " << std::string(buffer) << std::endl;
		}
		bool stop = false;
		while (!stop)
		{
			float x, y, z, weight;
			int idx;
			int n = fscanf(ifs, "%d\t%f\t%f\t%f\t%f\n", &idx, &x, &y, &z, &weight);
			//sscanf(buffer, "%d\t%f\t%f\t%f\t%f\n", &idx, &x, &y, &z, &weight);
			if (n == 5) {
				assert(idx < N);
				AtomPositions[3 * idx + 0] = x;
				AtomPositions[3 * idx + 1] = y;
				AtomPositions[3 * idx + 2] = z;
				AtomWeights[idx] = weight;
			}
			else
				stop = true;
		}

		return new Pseudoatoms(AtomPositions, AtomWeights, N, ATOM_INTERPOLATE);
	}

	void writeTsvFile(FileName tsvFileName) {

		FILE* ofs = fopen(tsvFileName.c_str(), "w");
		if (ofs == NULL) {
			std::cerr << "Failed to open " << tsvFileName << std::endl;
			exit(EXIT_FAILURE);
		}

		fprintf(ofs, "%d\n", NAtoms);
		fprintf(ofs, "##################################\n");
		for (size_t i = 0; i < NAtoms; i++)
		{
			fprintf(ofs, "%d\t%.10f\t%.10f\t%.10f\t%.10f\n", i, AtomPositions[i].x, AtomPositions[i].y, AtomPositions[i].z, AtomWeights[i]);
		}
		fprintf(ofs, "##################################\n");

		fclose(ofs);
	}

	void initGrid(int3 Dims, float cutoff) {
		gridDims = Dims;
		grid = new std::list<int>[Elements(Dims)];
		for (size_t i = 0; i < NAtoms; i++) {

			int x = (int)AtomPositions[i].x;
			int y = (int)AtomPositions[i].y;
			int z = (int)AtomPositions[i].z;
			grid[z*(Dims.x*Dims.y) + y * Dims.x + x].emplace_back(i);
		}
		neighbours = new std::vector<int>[NAtoms];
		neighbour_dists = new std::vector<float>[NAtoms];
		for (size_t i = 0; i < NAtoms; i++) {

			float x = AtomPositions[i].x;
			float y = AtomPositions[i].y;
			float z = AtomPositions[i].z;
			//search adjacent grid cells for close atoms
			for (int zz = (int)std::floor(z - cutoff); zz <= (int)std::ceil(z + cutoff); zz++){
				for (int yy = (int)std::floor(y - cutoff); yy <= (int)std::ceil(y + cutoff); yy++) {
					for (int xx = (int)std::floor(x - cutoff); xx <= (int)std::ceil(x + cutoff); xx++) {
						for (int otherAtomIdx : grid[zz*(Dims.x*Dims.y) + yy * Dims.x + xx]) {
							float dist = getLength(AtomPositions[i] - AtomPositions[otherAtomIdx]);
							if (dist < cutoff && otherAtomIdx != i) {
								neighbours[i].emplace_back(otherAtomIdx);
								neighbour_dists[i].emplace_back(dist);
							}
						}
					}
				}
			}
		}
	}

	void addAlternativeOrientation(float * positions, int n) {
		assert(n == NAtoms);
		float * copy = (float *)malloc(sizeof(*copy)*NAtoms * 3);
		memcpy(copy, positions, sizeof(*copy)*NAtoms * 3);
		alternativePositions.emplace_back(copy);
	}



};
#endif // !PSEUDOATOMS