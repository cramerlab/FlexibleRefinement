#include "ADAM_Solver.h"
#include <fstream>



void ADAM_Solver::run(AtomMover &mover, int numIt, FileName outfile)
{
	/*
	alpha = 0.01;
	beta1 = 0.9;
	beta2 = 0.999;
	t = 0;
	epsilon = 1e-8;
	*/
	m = Eigen::VectorXd::Zero(3*mover.Atoms->NAtoms);
	v = Eigen::VectorXd::Zero(3*mover.Atoms->NAtoms);
	std::ofstream ofs;
	if(outfile != "")
		ofs = std::ofstream(outfile);

	double fx;
	Eigen::VectorXd positions(mover.Atoms->NAtoms * 3);
	Eigen::VectorXd grad(mover.Atoms->NAtoms * 3);
	for (size_t i = 0; i < mover.Atoms->NAtoms; i++)
	{
		positions[i * 3 + 0] = mover.Atoms->AtomPositions[i].x;
		positions[i * 3 + 1] = mover.Atoms->AtomPositions[i].y;
		positions[i * 3 + 2] = mover.Atoms->AtomPositions[i].z;
	}
	fx = mover(positions, grad);
	for (size_t i = 0; i < numIt; i++)
	{
		t++;

		
		m = beta1*m + (1 - beta1)*grad;
		v = beta2*v + (1 - beta2)*(grad.cwiseProduct(grad));
		Eigen::VectorXd mt = m * 1.0f / (1 - pow(beta1, t));
		Eigen::VectorXd vt = v / (1 - pow(beta2, t));
		Eigen::VectorXd update = alpha * mt.array() / (vt.array().sqrt() + epsilon);
		double pos0 = positions[0];
		double pos1 = positions[1];
		double pos2 = positions[2];

		double upd0 = update[0];
		double upd1 = update[1];
		double upd2 = update[2];

		positions = positions - update;
		double updateNorm = update.norm();
		double pos0_u = positions[0];
		double pos1_u = positions[1];
		double pos2_u = positions[2];
		fx = mover(positions, grad);
		if (outfile != "")
			ofs << i << ": " << fx << std::endl;
		
		std::cout << i << ": " << fx << std::endl;
		/*{ (float)(AtomPositions[p].x - adamparams->alpha*mt.x / (sqrt(vt.x) + adamparams->epsilon)),
								(float)(AtomPositions[p].y - adamparams->alpha*mt.y / (sqrt(vt.y) + adamparams->epsilon)),
								(float)(AtomPositions[p].z - adamparams->alpha*mt.z / (sqrt(vt.z) + adamparams->epsilon)) };
		atoms->RasterizeToVolume(MovedVolume, Dims, super, true, weighting);
		{
			MRCImage<float> refIm(MovedVolume);
			//refIm.writeAs<float>(fnOut + "MovedVolume" + weightString + "_it" + std::to_string(i) + ".mrc", true);
		}
		err = 0.0;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(MovedVolume) {
			err += pow(DIRECT_A3D_ELEM(MovedVolume, k, i, j) - DIRECT_A3D_ELEM(RefVolume, k, i, j), 2);
		}
		Error[i + 1] = sqrt(err) / Elements(Dims);
		std::cout << Error[i + 1] << "\t" << 0.05*sqrt(3) << "\t" << sqrt(3)*0.1 / sqrt(i + 1) << std::endl;*/
	}
	for (size_t i = 0; i < mover.Atoms->NAtoms; i++)
	{
		mover.Atoms->AtomPositions[i].x = positions[i * 3 + 0];
		mover.Atoms->AtomPositions[i].y = positions[i * 3 + 1];
		mover.Atoms->AtomPositions[i].z = positions[i * 3 + 2];
	}
}
