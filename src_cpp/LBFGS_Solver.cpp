#include "LBFGS_Solver.h"
#include "AtomMover.h"
#include <LBFGSpp\include\LBFGS.h>
#include <LBFGSpp\include\LBFGSB.h>

void LBFGS_Solver::run(AtomMover &mover, int numIt)
{
	// Set up parameters
	LBFGSpp::LBFGSBParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = numIt;
	param.max_linesearch = 100;

	// Create solver and function object
	LBFGSpp::LBFGSBSolver<double> solver(param);

	// Initial guess & bounds
	Eigen::VectorXd lb = Eigen::VectorXd::Constant(mover.Atoms->NAtoms * 3, 0.0);
	Eigen::VectorXd ub = Eigen::VectorXd::Constant(mover.Atoms->NAtoms * 3, 0.0);
	Eigen::VectorXd positions = Eigen::VectorXd::Zero(mover.Atoms->NAtoms * 3);
	for (size_t i = 0; i < mover.Atoms->NAtoms; i++)
	{
		positions[i * 3] = mover.Atoms->AtomPositions[i].x;
		positions[i * 3 + 1] = mover.Atoms->AtomPositions[i].y;
		positions[i * 3 + 2] = mover.Atoms->AtomPositions[i].z;
		ub[i * 3] = mover.Dims.x - 1;
		ub[i * 3 + 1] = mover.Dims.y - 1;
		ub[i * 3 + 2] = mover.Dims.z - 1;
	}

	Eigen::VectorXd grad = Eigen::VectorXd(mover.Atoms->NAtoms * 3);
	// x will be overwritten to be the best point found
	double fx;
	fx = mover(positions, grad);
	std::cout << "iter 0 f(x) = " << fx << std::endl;
	int niter = solver.minimize(mover, positions, fx, lb, ub);

	std::cout << niter << " iterations" << std::endl;
	std::cout << "f(x) = " << fx << std::endl;
}
