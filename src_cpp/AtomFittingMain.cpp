#include "AtomFitting.h"
#include <cstdlib>
#include "readMRC.h"
#include "pseudoatoms.h"
#include "AtomMover.h"
#include <random>
#include <Eigen/Core>
#include <LBFGSpp/include/LBFGSB.h>
#include "ADAM_Solver.h"
#include "LBFGS_Solver.h"
int main(char** argv, int argc) {

	idxtype N = 800000; //Estimate of number of Atoms
	FileName pixsize = "2.0";
	float super = 1.0f;
	
	bool weighting = false;
	std::string weightString;
	if(weighting)
		weightString = "WithWeighting";
	else
		weightString = "NoWeighting";
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
	
	/*float3 pos = { 1.25, 1.5, 1.5 };
	AtomPositions.emplace_back(pos);
	AtomIntensities.emplace_back(1.0);
	NAtoms++;

	pos = { 1.75, 1.5, 1.5};
	AtomPositions.emplace_back(pos);
	AtomIntensities.emplace_back(2.0);
	NAtoms++;
	*/
	static std::default_random_engine e;
	e.seed(42); //Reproducible results for testing
	static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1

	for (size_t x = (super*Dims.x)/4; x < 3*(super*Dims.x-1)/4; x++)
	{
		for (size_t y = (super*Dims.y) / 4; y < 3 * (super*Dims.y - 1) / 4; y++)
		{
			for (size_t z = (super*Dims.z) / 4; z < 3 * (super*Dims.z - 1) / 4; z++)
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
	atoms.initGrid(Dims, 1.1/super);
	MultidimArray<float> SuperRefVolume;
	atoms.RasterizeToVolume(SuperRefVolume, Dims, super, false, weighting);
	{
		MRCImage<float> refIm(SuperRefVolume);
		refIm.writeAs<float>(fnOut + "SuperRefVol" + weightString + ".mrc", true);
	}

	MultidimArray<float> RefVolume;
	atoms.RasterizeToVolume(RefVolume, Dims, super, true, weighting);
	{
		MRCImage<float> refIm(RefVolume);
		refIm.writeAs<float>(fnOut + "RefVol" + weightString + ".mrc", true);
	}
	//writeFSC(RefVolume*referenceMaskIm(), referenceVolume()*referenceMaskIm(), fnOut + "RefVol" + weightString + ".fsc");



	for (size_t i = 0; i < NAtoms; i++)
	{
		float3 diff = { (dis(e) * 2 - 1)*0.4/super,(dis(e) * 2 - 1)*0.4/super, (dis(e) * 2 - 1)*0.4/super };
		atoms.AtomPositions[i] = atoms.AtomPositions[i] + diff;
	}

	//atoms.AtomPositions[0].x = 1.1;

	MultidimArray<float> SuperMovedVolume;
	atoms.RasterizeToVolume(SuperMovedVolume, Dims, super, false, weighting);
	{
		MRCImage<float> refIm(SuperMovedVolume);
		refIm.writeAs<float>(fnOut + "SuperMovedVolume" + weightString + ".mrc", true);
	}

	MultidimArray<float> MovedVolume;
	atoms.RasterizeToVolume(MovedVolume, Dims, super, true, weighting);
	{
		MRCImage<float> refIm(MovedVolume);
		refIm.writeAs<float>(fnOut + "MovedVolume" + weightString + ".mrc", true);
	}
	int numIt = 500;
	float* Error = (float*)malloc(sizeof(float)*(numIt+1));
	float err = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(MovedVolume) {
		err += pow(DIRECT_A3D_ELEM(MovedVolume, k, i, j) - DIRECT_A3D_ELEM(RefVolume, k, i, j), 2);
	}
	Error[0] = sqrt(err) / Elements(Dims);
	std::cout << Error[0] << std::endl;

	//writeFSC(MovedVolume*referenceMaskIm(), referenceVolume()*referenceMaskIm(), fnOut + "MovedVolume" + weightString + ".fsc");

	
	AtomMover mover(&atoms, SuperRefVolume, Dims, super, false, weighting, 0.1);
	
	//solver.run();
	ADAM_Solver adam_solver;
	LBFGS_Solver lbfgs_solver;

	//lbfgs_solver.run(mover, 500);
	adam_solver.run(mover, numIt);
	
	
	std::ofstream outfile("D:\\EMD\\9233\\AtomFitting\\iterationErr.txt", std::ios_base::out);
	for (size_t i = 0; i < numIt; i++)
	{
		outfile << std::to_string(Error[i]) << std::endl;
	}
	
	

 	return EXIT_SUCCESS;
}