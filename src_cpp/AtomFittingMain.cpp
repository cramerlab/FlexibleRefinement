#include "AtomFitting.h"
#include <cstdlib>
#include "readMRC.h"
#include "pseudoatoms.h"
#include "Solver.h"
#include <random>
#include <Eigen/Core>
#include <LBFGS.h>

int main(char** argv, int argc) {

	idxtype N = 800000; //Estimate of number of Atoms
	FileName pixsize = "2.0";
	float super = 4.0f;
	FileName starFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".projections_uniform_combined.distorted_10.star";


	FileName refFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".mrc";
	FileName refMaskFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_mask.mrc";
	FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_" + std::to_string(N / 1000) + "k.pdb";		//PDB File containing pseudo atom coordinates
	FileName fnOut = "D:\\EMD\\9233\\AtomFitting\\mock_";
	
	MRCImage<float> referenceVolume(refFileName);
	//int3 Dims = { referenceVolume().xdim, referenceVolume().ydim, referenceVolume().zdim };
	int3 Dims = { 20, 20, 20};

	MRCImage<float> referenceMaskIm(refMaskFileName);

	int3 superDims = { (int)(Dims.z * super + 0.5), (int)(Dims.y * super + 0.5), (int)(Dims.x * super + 0.5) };
	MultidimArray<float> superRefVolume = referenceVolume();
	if(super != 1.0f)
		ResizeMapGPU(superRefVolume, superDims);
	std::vector<float3> AtomPositions;
	std::vector<RDOUBLE> AtomIntensities;
	idxtype NAtoms = 0;
	//NAtoms = Pseudoatoms::readAtomsFromFile(pdbFileName, AtomPositions, AtomIntensities, N); //actually read number of atoms
	/*
	float3 pos = { 1.25, 1.1, 1.1 };
	AtomPositions.emplace_back(pos);
	AtomIntensities.emplace_back(1.0);
	NAtoms++;

	pos = { 1.75, 1.1, 1.1};
	AtomPositions.emplace_back(pos);
	AtomIntensities.emplace_back(2.0);
	NAtoms++;
	*/

	static std::default_random_engine e;
	e.seed(42); //Reproducible results for testing
	static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1

	for (size_t x = 0; x < super*Dims.x-1; x++)
	{
		for (size_t y = 0; y < super*Dims.y-1; y++)
		{
			for (size_t z = 0; z < super*Dims.z-1; z++)
			{
				float3 pos = { (x+0.5)/super, (y+0.5)/super, (z+0.5)/super };
				AtomPositions.emplace_back(pos);
				
				NAtoms++;
				AtomIntensities.emplace_back(dis(e));
			}
		}
	}

	//Pseudoatoms atoms(pseudoPositions, pseudoweights, 2, ATOM_INTERPOLATE);
	Pseudoatoms atoms(((float*)AtomPositions.data()), ((float*)AtomIntensities.data()), NAtoms, ATOM_INTERPOLATE);
	std::string weightString = "WithWeighting";
	MultidimArray<float> SuperRefVolume;
	atoms.RasterizeToVolume(SuperRefVolume, Dims, super, false);
	{
		MRCImage<float> refIm(SuperRefVolume);
		refIm.writeAs<float>(fnOut + "SuperRefVol" + weightString + ".mrc", true);
	}

	MultidimArray<float> RefVolume;
	atoms.RasterizeToVolume(RefVolume, Dims, super);
	{
		MRCImage<float> refIm(RefVolume);
		refIm.writeAs<float>(fnOut + "RefVol" + weightString + ".mrc", true);
	}
	//writeFSC(RefVolume*referenceMaskIm(), referenceVolume()*referenceMaskIm(), fnOut + "RefVol" + weightString + ".fsc");



	for (size_t i = 0; i < NAtoms/2; i++)
	{
		float3 diff = { (dis(e) * 2 - 1)*0.4/super,(dis(e) * 2 - 1)*0.4/super, (dis(e) * 2 - 1)*0.4/super };
		atoms.AtomPositions[i] = atoms.AtomPositions[i] + diff;
	}



	MultidimArray<float> SuperMovedVolume;
	atoms.RasterizeToVolume(SuperMovedVolume, Dims, super, false);
	{
		MRCImage<float> refIm(SuperMovedVolume);
		refIm.writeAs<float>(fnOut + "SuperMovedVolume" + weightString + ".mrc", true);
	}

	MultidimArray<float> MovedVolume;
	atoms.RasterizeToVolume(MovedVolume, Dims, super);
	{
		MRCImage<float> refIm(MovedVolume);
		refIm.writeAs<float>(fnOut + "MovedVolume" + weightString + ".mrc", true);
	}
	int numIt = 50;
	float* Error = (float*)malloc(sizeof(float)*(numIt+1));
	float err = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(MovedVolume) {
		err += pow(DIRECT_A3D_ELEM(MovedVolume, k, i, j) - DIRECT_A3D_ELEM(RefVolume, k, i, j), 2);
	}
	Error[0] = sqrt(err) / Elements(Dims);
	std::cout << Error[0] << std::endl;

	//writeFSC(MovedVolume*referenceMaskIm(), referenceVolume()*referenceMaskIm(), fnOut + "MovedVolume" + weightString + ".fsc");
	ADAMParams *adampars = new ADAMParams();
	adampars->alpha = 0.001;
	adampars->beta1 = 0.9;
	adampars->beta2 = 0.999;
	adampars->t = 0;
	adampars->epsilon = 1e-8;
	adampars->m = new float3[atoms.NAtoms];
	adampars->v = new float3[atoms.NAtoms];
	memset(adampars->m, 0, sizeof(*(adampars->m))*atoms.NAtoms);
	memset(adampars->v, 0, sizeof(*(adampars->v))*atoms.NAtoms);
	Solver solver(&atoms, SuperRefVolume, Dims, super, false);
	solver.run();
	
	/*
	for (size_t i = 0; i < numIt; i++)
	{
		adampars->t++;
		atoms.MoveAtoms(SuperRefVolume, Dims, super, false, 0.1,adampars);
		atoms.RasterizeToVolume(MovedVolume, Dims, super);
		{
			MRCImage<float> refIm(MovedVolume);
			//refIm.writeAs<float>(fnOut + "MovedVolume" + weightString + "_it" + std::to_string(i) + ".mrc", true);
		}
		err = 0.0;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(MovedVolume) {
			err += pow(DIRECT_A3D_ELEM(MovedVolume, k, i, j) - DIRECT_A3D_ELEM(RefVolume, k, i, j), 2);
		}
		Error[i+1] = sqrt(err)/Elements(Dims);
		std::cout << Error[i+1] <<  "\t" << 0.05*sqrt(3) << "\t" << sqrt(3)*0.1 / sqrt(i + 1) << std::endl;
	}
	std::ofstream outfile("D:\\EMD\\9233\\AtomFitting\\iterationErr.txt", std::ios_base::out);
	for (size_t i = 0; i < numIt; i++)
	{
		outfile << std::to_string(Error[i]) << std::endl;
	}
	*/


	return EXIT_SUCCESS;
}