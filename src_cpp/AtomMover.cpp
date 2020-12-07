#include "AtomMover.h"

//#include <lbfgs-cpp\include\lbfgs\lbfgs.hpp>

double AtomMover::operator()(Eigen::VectorXd positions, Eigen::VectorXd & grad)
{
	
	MultidimArray<float> SuperRastered((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
	MultidimArray<float> u((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
	//RasterizeToVolume(SuperRastered,Dims,super, false);

	MultidimArray<RDOUBLE> v((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));

#pragma omp parallel for
	for (int p = 0; p < this->Atoms->NAtoms; p++)
	{
		float3 pos = { positions[p * 3], positions[p * 3 + 1], positions[p * 3 + 2] };
		float3 superPos = pos * Super;

		int X0 = (int)(superPos.x);
		RDOUBLE ix = (superPos.x) - X0;
		int X1 = X0 + 1;

		int Y0 = (int)(superPos.y);
		RDOUBLE iy = (superPos.y) - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)(superPos.z);
		RDOUBLE iz = (superPos.z) - Z0;
		int Z1 = Z0 + 1;

		RDOUBLE v0 = 1.0f - ix;
		RDOUBLE v1 = ix;

		RDOUBLE v00 = (1.0f - iy) * v0;
		RDOUBLE v10 = iy * v0;
		RDOUBLE v01 = (1.0f - iy) * v1;
		RDOUBLE v11 = iy * v1;

		RDOUBLE v000 = (1.0f - iz) * v00;
		RDOUBLE v100 = iz * v00;
		RDOUBLE v010 = (1.0f - iz) * v10;
		RDOUBLE v110 = iz * v10;
		RDOUBLE v001 = (1.0f - iz) * v01;
		RDOUBLE v101 = iz * v01;
		RDOUBLE v011 = (1.0f - iz) * v11;
		RDOUBLE v111 = iz * v11;

		A3D_ELEM(u, Z0, Y0, X0) += Atoms->AtomWeights[p] * v000;
		A3D_ELEM(u, Z0, Y0, X1) += Atoms->AtomWeights[p] * v001;
		A3D_ELEM(u, Z0, Y1, X0) += Atoms->AtomWeights[p] * v010;
		A3D_ELEM(u, Z0, Y1, X1) += Atoms->AtomWeights[p] * v011;

		A3D_ELEM(u, Z1, Y0, X0) += Atoms->AtomWeights[p] * v100;
		A3D_ELEM(u, Z1, Y0, X1) += Atoms->AtomWeights[p] * v101;
		A3D_ELEM(u, Z1, Y1, X0) += Atoms->AtomWeights[p] * v110;
		A3D_ELEM(u, Z1, Y1, X1) += Atoms->AtomWeights[p] * v111;
		if (Weighting) {
			A3D_ELEM(v, Z0, Y0, X0) += v000;
			A3D_ELEM(v, Z0, Y0, X1) += v001;
			A3D_ELEM(v, Z0, Y1, X0) += v010;
			A3D_ELEM(v, Z0, Y1, X1) += v011;

			A3D_ELEM(v, Z1, Y0, X0) += v100;
			A3D_ELEM(v, Z1, Y0, X1) += v101;
			A3D_ELEM(v, Z1, Y1, X0) += v110;
			A3D_ELEM(v, Z1, Y1, X1) += v111;
		}
	}
	if (Weighting) {
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(SuperRastered) {
			A3D_ELEM(v, k, i, j) = std::max((RDOUBLE)(1e-10), A3D_ELEM(v, k, i, j));
			A3D_ELEM(SuperRastered, k, i, j) = A3D_ELEM(u, k, i, j) / std::max((RDOUBLE)(1e-10), A3D_ELEM(v, k, i, j));
		}
	}
	else {
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(v) {
			A3D_ELEM(v, k, i, j) = 1.0;
			A3D_ELEM(SuperRastered, k, i, j) = A3D_ELEM(u, k, i, j);
			
		}
	}
	MultidimArray<float> SuperDiffVol = SuperRastered;

	Substract_GPU(SuperDiffVol, SuperRefVolume);
	double err = SquaredSum(SuperDiffVol);

#pragma omp parallel for
	for (int p = 0; p < Atoms->NAtoms; p++)
	{
		float3 pos = { positions[p * 3], positions[p * 3 + 1], positions[p * 3 + 2] };
		float3 superPos = pos * Super;
		RDOUBLE weight = Atoms->AtomWeights[p];

		int X0 = (int)(superPos.x);
		RDOUBLE ix = (superPos.x) - X0;
		int X1 = X0 + 1;

		int Y0 = (int)(superPos.y);
		RDOUBLE iy = (superPos.y) - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)(superPos.z);
		RDOUBLE iz = (superPos.z) - Z0;
		int Z1 = Z0 + 1;
		float3 gt;
		if (Weighting)
		{
			RDOUBLE v0 = 1.0f - ix;
			RDOUBLE vd0 = -1;
			RDOUBLE v1 = ix;
			RDOUBLE vd1 = 1;

			RDOUBLE v00 = (1.0f - iy) * v0;
			RDOUBLE v0d0 = (1.0f - iy) * vd0;
			RDOUBLE vd00 = -1 * v0;

			RDOUBLE v10 = iy * v0;
			RDOUBLE v1d0 = iy * vd0;
			RDOUBLE vd10 = 1 * v0;


			RDOUBLE v01 = (1.0f - iy) * v1;
			RDOUBLE v0d1 = (1.0f - iy) * vd1;
			RDOUBLE vd01 = (-1) * v1;

			RDOUBLE v11 = iy * v1;
			RDOUBLE v1d1 = iy * vd1;
			RDOUBLE vd11 = 1 * v1;




			RDOUBLE v00d0 = (1.0f - iz) * v0d0;
			RDOUBLE g00d0 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) / DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X0))*((v00d0*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(u, Z0, Y0, X0)*v00d0) / (DIRECT_A3D_ELEM(v, Z0, Y0, X0)*DIRECT_A3D_ELEM(v, Z0, Y0, X0)));
			RDOUBLE v0d00 = (1.0f - iz) * vd00;
			RDOUBLE g0d00 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) / DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X0))*((v0d00*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(u, Z0, Y0, X0)*v0d00) / (DIRECT_A3D_ELEM(v, Z0, Y0, X0)*DIRECT_A3D_ELEM(v, Z0, Y0, X0)));
			RDOUBLE vd000 = (-1) * v00;
			RDOUBLE gd000 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) / DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X0))*((vd000*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(u, Z0, Y0, X0)*vd000) / (DIRECT_A3D_ELEM(v, Z0, Y0, X0)*DIRECT_A3D_ELEM(v, Z0, Y0, X0)));


			RDOUBLE v10d0 = iz * v0d0;
			RDOUBLE g10d0 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) / DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X0))*((v10d0*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(u, Z1, Y0, X0)*v10d0) / (DIRECT_A3D_ELEM(v, Z1, Y0, X0)*DIRECT_A3D_ELEM(v, Z1, Y0, X0)));
			RDOUBLE v1d00 = iz * vd00;
			RDOUBLE g1d00 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) / DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X0))*((v1d00*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(u, Z1, Y0, X0)*v1d00) / (DIRECT_A3D_ELEM(v, Z1, Y0, X0)*DIRECT_A3D_ELEM(v, Z1, Y0, X0)));
			RDOUBLE vd100 = 1 * v00;
			RDOUBLE gd100 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) / DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X0))*((vd100*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(u, Z1, Y0, X0)*vd100) / (DIRECT_A3D_ELEM(v, Z1, Y0, X0)*DIRECT_A3D_ELEM(v, Z1, Y0, X0)));


			RDOUBLE v01d0 = (1.0f - iz) * v1d0;
			RDOUBLE g01d0 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) / DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X0))*((v01d0*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(u, Z0, Y1, X0)*v01d0) / (DIRECT_A3D_ELEM(v, Z0, Y1, X0)*DIRECT_A3D_ELEM(v, Z0, Y1, X0)));
			RDOUBLE v0d10 = (1.0f - iz) * vd10;
			RDOUBLE g0d10 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) / DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X0))*((v0d10*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(u, Z0, Y1, X0)*v0d10) / (DIRECT_A3D_ELEM(v, Z0, Y1, X0)*DIRECT_A3D_ELEM(v, Z0, Y1, X0)));
			RDOUBLE vd010 = (-1) * v10;
			RDOUBLE gd010 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) / DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X0))*((vd010*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(u, Z0, Y1, X0)*vd010) / (DIRECT_A3D_ELEM(v, Z0, Y1, X0)*DIRECT_A3D_ELEM(v, Z0, Y1, X0)));


			RDOUBLE v11d0 = iz * v1d0;
			RDOUBLE g11d0 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) / DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X0))*((v11d0*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(u, Z1, Y1, X0)*v11d0) / (DIRECT_A3D_ELEM(v, Z1, Y1, X0)*DIRECT_A3D_ELEM(v, Z1, Y1, X0)));
			RDOUBLE v1d10 = iz * vd10;
			RDOUBLE g1d10 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) / DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X0))*((v1d10*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(u, Z1, Y1, X0)*v1d10) / (DIRECT_A3D_ELEM(v, Z1, Y1, X0)*DIRECT_A3D_ELEM(v, Z1, Y1, X0)));
			RDOUBLE vd110 = 1 * v10;
			RDOUBLE gd110 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) / DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X0))*((vd110*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(u, Z1, Y1, X0)*vd110) / (DIRECT_A3D_ELEM(v, Z1, Y1, X0)*DIRECT_A3D_ELEM(v, Z1, Y1, X0)));


			RDOUBLE v00d1 = (1.0f - iz) * v0d1;
			RDOUBLE g00d1 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) / DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X1))*((v00d1*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(u, Z0, Y0, X1)*v00d1) / (DIRECT_A3D_ELEM(v, Z0, Y0, X1)*DIRECT_A3D_ELEM(v, Z0, Y0, X1)));
			RDOUBLE v0d01 = (1.0f - iz) * vd01;
			RDOUBLE g0d01 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) / DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X1))*((v0d01*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(u, Z0, Y0, X1)*v0d01) / (DIRECT_A3D_ELEM(v, Z0, Y0, X1)*DIRECT_A3D_ELEM(v, Z0, Y0, X1)));
			RDOUBLE vd001 = (-1.0f) * v01;
			RDOUBLE gd001 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) / DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X1))*((vd001*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(u, Z0, Y0, X1)*vd001) / (DIRECT_A3D_ELEM(v, Z0, Y0, X1)*DIRECT_A3D_ELEM(v, Z0, Y0, X1)));


			RDOUBLE v10d1 = iz * v0d1;
			RDOUBLE g10d1 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) / DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X1))*((v10d1*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(u, Z1, Y0, X1)*v10d1) / (DIRECT_A3D_ELEM(v, Z1, Y0, X1)*DIRECT_A3D_ELEM(v, Z1, Y0, X1)));
			RDOUBLE v1d01 = iz * vd01;
			RDOUBLE g1d01 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) / DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X1))*((v1d01*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(u, Z1, Y0, X1)*v1d01) / (DIRECT_A3D_ELEM(v, Z1, Y0, X1)*DIRECT_A3D_ELEM(v, Z1, Y0, X1)));
			RDOUBLE vd101 = 1 * v01;
			RDOUBLE gd101 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) / DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X1))*((vd101*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(u, Z1, Y0, X1)*vd101) / (DIRECT_A3D_ELEM(v, Z1, Y0, X1)*DIRECT_A3D_ELEM(v, Z1, Y0, X1)));


			RDOUBLE v01d1 = (1.0f - iz) * v1d1;
			RDOUBLE g01d1 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) / DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X1))*((v01d1*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(u, Z0, Y1, X1)*v01d1) / (DIRECT_A3D_ELEM(v, Z0, Y1, X1)*DIRECT_A3D_ELEM(v, Z0, Y1, X1)));
			RDOUBLE v0d11 = (1.0f - iz) * vd11;
			RDOUBLE g0d11 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) / DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X1))*((v0d11*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(u, Z0, Y1, X1)*v0d11) / (DIRECT_A3D_ELEM(v, Z0, Y1, X1)*DIRECT_A3D_ELEM(v, Z0, Y1, X1)));
			RDOUBLE vd011 = (-1.0f) * v11;
			RDOUBLE gd011 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) / DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X1))*((vd011*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(u, Z0, Y1, X1)*vd011) / (DIRECT_A3D_ELEM(v, Z0, Y1, X1)*DIRECT_A3D_ELEM(v, Z0, Y1, X1)));


			RDOUBLE v11d1 = iz * v1d1;
			RDOUBLE g11d1 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) / DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X1))*((v11d1*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(u, Z1, Y1, X1)*v11d1) / (DIRECT_A3D_ELEM(v, Z1, Y1, X1)*DIRECT_A3D_ELEM(v, Z1, Y1, X1)));
			RDOUBLE v1d11 = iz * vd11;
			RDOUBLE g1d11 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) / DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X1))*((v1d11*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(u, Z1, Y1, X1)*v1d11) / (DIRECT_A3D_ELEM(v, Z1, Y1, X1)*DIRECT_A3D_ELEM(v, Z1, Y1, X1)));
			RDOUBLE vd111 = 1 * v11;
			RDOUBLE gd111 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) / DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X1))*((vd111*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(u, Z1, Y1, X1)*vd111) / (DIRECT_A3D_ELEM(v, Z1, Y1, X1)*DIRECT_A3D_ELEM(v, Z1, Y1, X1)));
			gt = { g00d0 + g00d1 + g01d0 + g01d1 + g10d0 + g10d1 + g11d0 + g11d1,
					g0d00 + g0d01 + g0d10 + g0d11 + g1d00 + g1d01 + g1d10 + g1d11,
					gd000 + gd001 + gd010 + gd011 + gd100 + gd101 + gd110 + gd111 };
		}
		else {
			RDOUBLE v0 = 1.0f - ix;
			RDOUBLE vd0 = -1;
			RDOUBLE v1 = ix;
			RDOUBLE vd1 = 1;

			RDOUBLE v00 = (1.0f - iy) * v0;
			RDOUBLE v0d0 = (1.0f - iy) * vd0;
			RDOUBLE vd00 = -1 * v0;

			RDOUBLE v10 = iy * v0;
			RDOUBLE v1d0 = iy * vd0;
			RDOUBLE vd10 = 1 * v0;


			RDOUBLE v01 = (1.0f - iy) * v1;
			RDOUBLE v0d1 = (1.0f - iy) * vd1;
			RDOUBLE vd01 = (-1) * v1;

			RDOUBLE v11 = iy * v1;
			RDOUBLE v1d1 = iy * vd1;
			RDOUBLE vd11 = 1 * v1;

			RDOUBLE v00d0 = (1.0f - iz) * v0d0;
			RDOUBLE g00d0 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X0))*(v00d0*weight);
			RDOUBLE v0d00 = (1.0f - iz) * vd00;
			RDOUBLE g0d00 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X0))*(v0d00*weight);
			RDOUBLE vd000 = (-1) * v00;
			RDOUBLE gd000 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X0))*(vd000*weight);

			RDOUBLE v10d0 = iz * v0d0;
			RDOUBLE g10d0 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X0))*(v10d0*weight);
			RDOUBLE v1d00 = iz * vd00;
			RDOUBLE g1d00 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X0))*(v1d00*weight);
			RDOUBLE vd100 = 1 * v00;
			RDOUBLE gd100 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X0))*(vd100*weight);


			RDOUBLE v01d0 = (1.0f - iz) * v1d0;
			RDOUBLE g01d0 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X0))*(v01d0*weight);
			RDOUBLE v0d10 = (1.0f - iz) * vd10;
			RDOUBLE g0d10 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X0))*(v0d10*weight);
			RDOUBLE vd010 = (-1) * v10;
			RDOUBLE gd010 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X0))*(vd010*weight);


			RDOUBLE v11d0 = iz * v1d0;
			RDOUBLE g11d0 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X0))*(v11d0*weight);
			RDOUBLE v1d10 = iz * vd10;
			RDOUBLE g1d10 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X0))*(v1d10*weight);
			RDOUBLE vd110 = 1 * v10;
			RDOUBLE gd110 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X0))*(vd110*weight);


			RDOUBLE v00d1 = (1.0f - iz) * v0d1;
			RDOUBLE g00d1 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X1))*(v00d1*weight);
			RDOUBLE v0d01 = (1.0f - iz) * vd01;
			RDOUBLE g0d01 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X1))*(v0d01*weight);
			RDOUBLE vd001 = (-1.0f) * v01;
			RDOUBLE gd001 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y0, X1))*(vd001*weight);


			RDOUBLE v10d1 = iz * v0d1;
			RDOUBLE g10d1 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X1))*(v10d1*weight);
			RDOUBLE v1d01 = iz * vd01;
			RDOUBLE g1d01 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X1))*(v1d01*weight);
			RDOUBLE vd101 = 1 * v01;
			RDOUBLE gd101 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y0, X1))*(vd101*weight);


			RDOUBLE v01d1 = (1.0f - iz) * v1d1;
			RDOUBLE g01d1 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X1))*(v01d1*weight);
			RDOUBLE v0d11 = (1.0f - iz) * vd11;
			RDOUBLE g0d11 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X1))*(v0d11*weight);
			RDOUBLE vd011 = (-1.0f) * v11;
			RDOUBLE gd011 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z0, Y1, X1))*(vd011*weight);


			RDOUBLE v11d1 = iz * v1d1;
			RDOUBLE g11d1 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X1))*(v11d1*weight);
			RDOUBLE v1d11 = iz * vd11;
			RDOUBLE g1d11 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X1))*(v1d11*weight);
			RDOUBLE vd111 = 1 * v11;
			RDOUBLE gd111 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVolume, Z1, Y1, X1))*(vd111*weight);



			gt = { g00d0 + g00d1 + g01d0 + g01d1 + g10d0 + g10d1 + g11d0 + g11d1,
					g0d00 + g0d01 + g0d10 + g0d11 + g1d00 + g1d01 + g1d10 + g1d11,
					gd000 + gd001 + gd010 + gd011 + gd100 + gd101 + gd110 + gd111 };
		}
		float3 elasticGrad = { 0,0,0 };
		/* Distance based component*/
		for (size_t i = 0; i < Atoms->neighbours[p].size(); i++)
		{
			int otherAtomIdx = Atoms->neighbours[p][i];
			float3 otherPos = { positions[otherAtomIdx * 3], positions[otherAtomIdx * 3 + 1], positions[otherAtomIdx * 3 + 2] };
			float3 diffvec = pos - otherPos;
			float diff = getLength(diffvec);
			
			elasticGrad = elasticGrad + this->K*(diff - Atoms->neighbour_dists[p][i])*1.0 / diff * diffvec;

		}

		float length = sqrt(gt.x*gt.x + gt.y*gt.y + gt.z*gt.z);
		/*if (length > 1) {
			gt = 1.0 / length * gt;
		}*/
		grad[3 * p] = gt.x+elasticGrad.x;
		grad[3 * p + 1] = gt.y + elasticGrad.y;
		grad[3 * p + 2] = gt.z + elasticGrad.z;
	}
	return err;
}
/*
void AtomMover::run()
{
}*/
