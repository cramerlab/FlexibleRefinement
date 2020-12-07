#include "pseudoatoms.h"
#include "readMRC.h"

using namespace relion;




void Pseudoatoms::RasterizeToVolume(MultidimArray<RDOUBLE>& vol, int3 Dims, RDOUBLE Super, bool resize, bool weighting)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		vol.resizeNoCopy((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
		vol.initZeros();

		//resizeMap(vol, (long)(Dims.z * Super + 0.5));
		//ResizeMapGPU(vol, make_int3((int)(volSuper.xdim*Super + 0.5), (int)(volSuper.ydim*Super + 0.5), (int)(volSuper.zdim*Super + 0.5)));


		MultidimArray<RDOUBLE> Weights((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 SuperPos = AtomPositions[p] * Super;

			int X0 = (int)(SuperPos.x);
			RDOUBLE ix = (SuperPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(SuperPos.y);
			RDOUBLE iy = (SuperPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(SuperPos.z);
			RDOUBLE iz = (SuperPos.z) - Z0;
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

			A3D_ELEM(vol, Z0, Y0, X0) += AtomWeights[p] * v000;
			A3D_ELEM(vol, Z0, Y0, X1) += AtomWeights[p] * v001;
			A3D_ELEM(vol, Z0, Y1, X0) += AtomWeights[p] * v010;
			A3D_ELEM(vol, Z0, Y1, X1) += AtomWeights[p] * v011;

			A3D_ELEM(vol, Z1, Y0, X0) += AtomWeights[p] * v100;
			A3D_ELEM(vol, Z1, Y0, X1) += AtomWeights[p] * v101;
			A3D_ELEM(vol, Z1, Y1, X0) += AtomWeights[p] * v110;
			A3D_ELEM(vol, Z1, Y1, X1) += AtomWeights[p] * v111;
			if (weighting) {
				A3D_ELEM(Weights, Z0, Y0, X0) += v000;
				A3D_ELEM(Weights, Z0, Y0, X1) += v001;
				A3D_ELEM(Weights, Z0, Y1, X0) += v010;
				A3D_ELEM(Weights, Z0, Y1, X1) += v011;

				A3D_ELEM(Weights, Z1, Y0, X0) += v100;
				A3D_ELEM(Weights, Z1, Y0, X1) += v101;
				A3D_ELEM(Weights, Z1, Y1, X0) += v110;
				A3D_ELEM(Weights, Z1, Y1, X1) += v111;
			}
		}
		if (weighting)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(vol) {
				A3D_ELEM(vol, k, i, j) /= std::max((RDOUBLE)(1e-10), A3D_ELEM(Weights, k, i, j));
			}
		}
		Weights.coreDeallocate();
		if(resize)
			ResizeMapGPU(vol, Dims);
		return;
	}
	else if (Mode == ATOM_GAUSSIAN) {
		resizeMap(vol, (long)(Dims.z * Super + 0.5));
		auto itPos = AtomPositions.begin();
		auto itWeight = AtomWeights.begin();

		for (int n = 0; n < AtomPositions.size(); n++) {
			drawOneGaussian(GaussianTable, 4 * Sigma*Super, (AtomPositions[n].z)*Super, (AtomPositions[n].y)*Super, (AtomPositions[n].x)*Super, vol, AtomWeights[n], GaussFactor / Super);
		}

		if (Super > 1.0) {
			resizeMap(vol, Dims.x);
		}
	}
}

void Pseudoatoms::IntensityFromVolume(MultidimArray<RDOUBLE>& vol, RDOUBLE Super)
{
	AtomWeights.clear();
	AtomWeights.resize(AtomPositions.size());
	MultidimArray<RDOUBLE> volSuper = vol;
	ResizeMapGPU(volSuper, make_int3((int)(volSuper.xdim*Super + 0.5), (int)(volSuper.ydim*Super + 0.5), (int)(volSuper.zdim*Super + 0.5)));

#pragma omp parallel for
	for (int p = 0; p < AtomPositions.size(); p++)
	{
		float3 SuperPos = AtomPositions[p] * Super;

		int X0 = (int)(SuperPos.x);
		RDOUBLE ix = (SuperPos.x) - X0;
		int X1 = X0 + 1;

		int Y0 = (int)(SuperPos.y);
		RDOUBLE iy = (SuperPos.y) - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)(SuperPos.z);
		RDOUBLE iz = (SuperPos.z) - Z0;
		int Z1 = Z0 + 1;

		RDOUBLE v000 = A3D_ELEM(volSuper, Z0, Y0, X0);
		RDOUBLE v001 = A3D_ELEM(volSuper, Z0, Y0, X1);
		RDOUBLE v010 = A3D_ELEM(volSuper, Z0, Y1, X0);
		RDOUBLE v011 = A3D_ELEM(volSuper, Z0, Y1, X1);
		RDOUBLE v100 = A3D_ELEM(volSuper, Z1, Y0, X0);
		RDOUBLE v101 = A3D_ELEM(volSuper, Z1, Y0, X1);
		RDOUBLE v110 = A3D_ELEM(volSuper, Z1, Y1, X0);
		RDOUBLE v111 = A3D_ELEM(volSuper, Z1, Y1, X1);

		RDOUBLE v00 = Lerp(v000, v001, ix);
		RDOUBLE v01 = Lerp(v010, v011, ix);
		RDOUBLE v10 = Lerp(v100, v101, ix);
		RDOUBLE v11 = Lerp(v110, v111, ix);

		RDOUBLE v0 = Lerp(v00, v01, iy);
		RDOUBLE v1 = Lerp(v10, v11, iy);

		RDOUBLE v = Lerp(v0, v1, iz);

		AtomWeights[p] = v;
	}

	volSuper.coreDeallocate();
}


double dIRECT_A3D_Elem(MultidimArray<RDOUBLE> u, int k, int i, int j) {
	return DIRECT_A3D_ELEM(u, k, i, j);
}
/*
void Pseudoatoms::MoveAtoms(MultidimArray<RDOUBLE>& SuperRefVolume, int3 Dims, RDOUBLE Super, bool resize, double limit, bool Weighting, ADAMParams * adamparams)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		MultidimArray<float> SuperRastered((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
		MultidimArray<float> u((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
		//RasterizeToVolume(SuperRastered,Dims,Super, false);

		MultidimArray<RDOUBLE> v((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
		std::vector<float3> newAtomPositions(AtomPositions);
#pragma omp parallel for
		for (int p = 0; p < this->NAtoms; p++)
		{
			float3 pos = AtomPositions[p];
			float3 SuperPos = pos * Super;

			int X0 = (int)(SuperPos.x);
			RDOUBLE ix = (SuperPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(SuperPos.y);
			RDOUBLE iy = (SuperPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(SuperPos.z);
			RDOUBLE iz = (SuperPos.z) - Z0;
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

			A3D_ELEM(u, Z0, Y0, X0) += AtomWeights[p] * v000;
			A3D_ELEM(u, Z0, Y0, X1) += AtomWeights[p] * v001;
			A3D_ELEM(u, Z0, Y1, X0) += AtomWeights[p] * v010;
			A3D_ELEM(u, Z0, Y1, X1) += AtomWeights[p] * v011;

			A3D_ELEM(u, Z1, Y0, X0) += AtomWeights[p] * v100;
			A3D_ELEM(u, Z1, Y0, X1) += AtomWeights[p] * v101;
			A3D_ELEM(u, Z1, Y1, X0) += AtomWeights[p] * v110;
			A3D_ELEM(u, Z1, Y1, X1) += AtomWeights[p] * v111;
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
			}
		}
		MultidimArray<float> SuperDiffVol = SuperRastered;

		Substract_GPU(SuperDiffVol, SuperRefVolume);
		double err = SquaredSum(SuperDiffVol);


		for (int p = 0; p < NAtoms; p++)
		{
			float3 pos = AtomPositions[p];
			float3 SuperPos = pos * Super;
			RDOUBLE weight = AtomWeights[p];

			int X0 = (int)(SuperPos.x);
			RDOUBLE ix = (SuperPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(SuperPos.y);
			RDOUBLE iy = (SuperPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(SuperPos.z);
			RDOUBLE iz = (SuperPos.z) - Z0;
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
				gt = gt;
			}
			//float movementLength = sqrt(movement.x*movement.x + movement.y*movement.y + movement.z*movement.z);
			//if (movementLength > limit) {
			//	movement = movement * (limit / movementLength);
			//}

			adamparams->m[p] = adamparams->beta1*adamparams->m[p] + (1 - adamparams->beta1)*gt;
			adamparams->v[p] = adamparams->beta2*adamparams->v[p] + (1 - adamparams->beta2)*(gt*gt);
			float3 mt = adamparams->m[p] * 1.0f / (1 - pow(adamparams->beta1, adamparams->t));
			float3 vt = adamparams->v[p] / (1 - pow(adamparams->beta2, adamparams->t));
			
			newAtomPositions[p] = {(float) (AtomPositions[p].x - adamparams->alpha*mt.x / (sqrt(vt.x) + adamparams->epsilon)),
									(float)(AtomPositions[p].y - adamparams->alpha*mt.y / (sqrt(vt.y) + adamparams->epsilon) ),
									(float)(AtomPositions[p].z - adamparams->alpha*mt.z / (sqrt(vt.z) + adamparams->epsilon) )};
			//if (1)
				//std::cout << "Atom " << p << " move by " << -adamparams->alpha*mt.x / (sqrt(vt.x) + adamparams->epsilon) << ", " << -adamparams->alpha*mt.y / (sqrt(vt.y) + adamparams->epsilon) << ", " << -adamparams->alpha*mt.z / (sqrt(vt.z) + adamparams->epsilon) << std::endl;
			if (newAtomPositions[p].x*Super < (int)AtomPositions[p].x*Super)
				newAtomPositions[p].x = ((int)(AtomPositions[p].x*Super) + 1e-5) / Super;
			if (newAtomPositions[p].x*Super > (int)AtomPositions[p].x*Super + 1)
				newAtomPositions[p].x = ((int)(AtomPositions[p].x*Super + 1) - 1e-5) / Super;

			if (newAtomPositions[p].z*Super < (int)AtomPositions[p].z*Super)
				newAtomPositions[p].z = ((int)(AtomPositions[p].z*Super) + 1e-5) / Super;
			if (newAtomPositions[p].z*Super > (int)AtomPositions[p].z*Super + 1.0)
				newAtomPositions[p].z = (int)(AtomPositions[p].z + 1.0) / Super - 1e-5;

			if (newAtomPositions[p].y*Super < (int)AtomPositions[p].y*Super)
				newAtomPositions[p].y = ((int)(AtomPositions[p].y*Super) + 1e-5) / Super;
			if (newAtomPositions[p].y*Super > (int)AtomPositions[p].y*Super + 1)
				newAtomPositions[p].y = (int)(AtomPositions[p].y + 1.0) / Super - 1e-5;
		}
		AtomPositions = newAtomPositions;



		return;
	}

}

static RDOUBLE dIRECT_A3D_ELEM(MultidimArray<RDOUBLE> u, int k, int i, int j)
{
	return DIRECT_A3D_ELEM(u, k, i, j);
}

void Pseudoatoms::MoveAtoms(MultidimArray<RDOUBLE>& SuperRefVol, int3 Dims, RDOUBLE Super, bool resize, float limit)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		MultidimArray<float> SuperRastered((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
		MultidimArray<float> u((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
		//RasterizeToVolume(SuperRastered,Dims,Super, false);

		MultidimArray<RDOUBLE> v((long)(Dims.z * Super + 0.5), (long)(Dims.y * Super + 0.5), (long)(Dims.x * Super + 0.5));
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 SuperPos = AtomPositions[p] * Super;

			int X0 = (int)(SuperPos.x);
			RDOUBLE ix = (SuperPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(SuperPos.y);
			RDOUBLE iy = (SuperPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(SuperPos.z);
			RDOUBLE iz = (SuperPos.z) - Z0;
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

			A3D_ELEM(u, Z0, Y0, X0) += AtomWeights[p] * v000;
			A3D_ELEM(u, Z0, Y0, X1) += AtomWeights[p] * v001;
			A3D_ELEM(u, Z0, Y1, X0) += AtomWeights[p] * v010;
			A3D_ELEM(u, Z0, Y1, X1) += AtomWeights[p] * v011;

			A3D_ELEM(u, Z1, Y0, X0) += AtomWeights[p] * v100;
			A3D_ELEM(u, Z1, Y0, X1) += AtomWeights[p] * v101;
			A3D_ELEM(u, Z1, Y1, X0) += AtomWeights[p] * v110;
			A3D_ELEM(u, Z1, Y1, X1) += AtomWeights[p] * v111;

			A3D_ELEM(v, Z0, Y0, X0) += v000;
			A3D_ELEM(v, Z0, Y0, X1) += v001;
			A3D_ELEM(v, Z0, Y1, X0) += v010;
			A3D_ELEM(v, Z0, Y1, X1) += v011;

			A3D_ELEM(v, Z1, Y0, X0) += v100;
			A3D_ELEM(v, Z1, Y0, X1) += v101;
			A3D_ELEM(v, Z1, Y1, X0) += v110;
			A3D_ELEM(v, Z1, Y1, X1) += v111;
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(SuperRastered) {
			A3D_ELEM(v, k, i, j) = std::max((RDOUBLE)(1e-10), A3D_ELEM(v, k, i, j));
			A3D_ELEM(SuperRastered, k, i, j) = A3D_ELEM(u, k, i, j) / std::max((RDOUBLE)(1e-10), A3D_ELEM(v, k, i, j));
		}

		MultidimArray<float> SuperDiffVol = SuperRastered;

		Substract_GPU(SuperDiffVol, SuperRefVol);
		{
			MRCImage<float> DiffIm(SuperDiffVol);
			DiffIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_SuperDiffIm.mrc", true);
		}

		{
			MRCImage<float> SuperRefIm(SuperRefVol);
			SuperRefIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_SuperRefVol.mrc", true);
			MRCImage<float> SuperRasteredIm(SuperRastered);
			SuperRasteredIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_rasteredVol.mrc", true);

		}
		std::vector<float3> newAtomPositions;
		newAtomPositions = AtomPositions;
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 SuperPos = AtomPositions[p] * Super;
			RDOUBLE weight = AtomWeights[p];

			int X0 = (int)(SuperPos.x);
			RDOUBLE ix = (SuperPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(SuperPos.y);
			RDOUBLE iy = (SuperPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(SuperPos.z);
			RDOUBLE iz = (SuperPos.z) - Z0;
			int Z1 = Z0 + 1;

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
			RDOUBLE g00d0 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) / DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y0, X0))*((v00d0*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(u, Z0, Y0, X0)*v00d0) / (DIRECT_A3D_ELEM(v, Z0, Y0, X0)*DIRECT_A3D_ELEM(v, Z0, Y0, X0)));
			RDOUBLE v0d00 = (1.0f - iz) * vd00;
			RDOUBLE g0d00 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) / DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y0, X0))*((v0d00*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(u, Z0, Y0, X0)*v0d00) / (DIRECT_A3D_ELEM(v, Z0, Y0, X0)*DIRECT_A3D_ELEM(v, Z0, Y0, X0)));
			RDOUBLE vd000 = (-1) * v00;
			RDOUBLE gd000 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X0) / DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y0, X0))*((vd000*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X0) - DIRECT_A3D_ELEM(u, Z0, Y0, X0)*vd000) / (DIRECT_A3D_ELEM(v, Z0, Y0, X0)*DIRECT_A3D_ELEM(v, Z0, Y0, X0)));


			RDOUBLE v10d0 = iz * v0d0;
			RDOUBLE g10d0 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) / DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y0, X0))*((v10d0*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(u, Z1, Y0, X0)*v10d0) / (DIRECT_A3D_ELEM(v, Z1, Y0, X0)*DIRECT_A3D_ELEM(v, Z1, Y0, X0)));
			RDOUBLE v1d00 = iz * vd00;
			RDOUBLE g1d00 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) / DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y0, X0))*((v1d00*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(u, Z1, Y0, X0)*v1d00) / (DIRECT_A3D_ELEM(v, Z1, Y0, X0)*DIRECT_A3D_ELEM(v, Z1, Y0, X0)));
			RDOUBLE vd100 = 1 * v00;
			RDOUBLE gd100 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X0) / DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y0, X0))*((vd100*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X0) - DIRECT_A3D_ELEM(u, Z1, Y0, X0)*vd100) / (DIRECT_A3D_ELEM(v, Z1, Y0, X0)*DIRECT_A3D_ELEM(v, Z1, Y0, X0)));


			RDOUBLE v01d0 = (1.0f - iz) * v1d0;
			RDOUBLE g01d0 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) / DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y1, X0))*((v01d0*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(u, Z0, Y1, X0)*v01d0) / (DIRECT_A3D_ELEM(v, Z0, Y1, X0)*DIRECT_A3D_ELEM(v, Z0, Y1, X0)));
			RDOUBLE v0d10 = (1.0f - iz) * vd10;
			RDOUBLE g0d10 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) / DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y1, X0))*((v0d10*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(u, Z0, Y1, X0)*v0d10) / (DIRECT_A3D_ELEM(v, Z0, Y1, X0)*DIRECT_A3D_ELEM(v, Z0, Y1, X0)));
			RDOUBLE vd010 = (-1) * v10;
			RDOUBLE gd010 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X0) / DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y1, X0))*((vd010*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X0) - DIRECT_A3D_ELEM(u, Z0, Y1, X0)*vd010) / (DIRECT_A3D_ELEM(v, Z0, Y1, X0)*DIRECT_A3D_ELEM(v, Z0, Y1, X0)));


			RDOUBLE v11d0 = iz * v1d0;
			RDOUBLE g11d0 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) / DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y1, X0))*((v11d0*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(u, Z1, Y1, X0)*v11d0) / (DIRECT_A3D_ELEM(v, Z1, Y1, X0)*DIRECT_A3D_ELEM(v, Z1, Y1, X0)));
			RDOUBLE v1d10 = iz * vd10;
			RDOUBLE g1d10 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) / DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y1, X0))*((v1d10*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(u, Z1, Y1, X0)*v1d10) / (DIRECT_A3D_ELEM(v, Z1, Y1, X0)*DIRECT_A3D_ELEM(v, Z1, Y1, X0)));
			RDOUBLE vd110 = 1 * v10;
			RDOUBLE gd110 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X0) / DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y1, X0))*((vd110*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X0) - DIRECT_A3D_ELEM(u, Z1, Y1, X0)*vd110) / (DIRECT_A3D_ELEM(v, Z1, Y1, X0)*DIRECT_A3D_ELEM(v, Z1, Y1, X0)));


			RDOUBLE v00d1 = (1.0f - iz) * v0d1;
			RDOUBLE g00d1 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) / DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y0, X1))*((v00d1*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(u, Z0, Y0, X1)*v00d1) / (DIRECT_A3D_ELEM(v, Z0, Y0, X1)*DIRECT_A3D_ELEM(v, Z0, Y0, X1)));
			RDOUBLE v0d01 = (1.0f - iz) * vd01;
			RDOUBLE g0d01 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) / DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y0, X1))*((v0d01*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(u, Z0, Y0, X1)*v0d01) / (DIRECT_A3D_ELEM(v, Z0, Y0, X1)*DIRECT_A3D_ELEM(v, Z0, Y0, X1)));
			RDOUBLE vd001 = (-1.0f) * v01;
			RDOUBLE gd001 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y0, X1) / DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y0, X1))*((vd001*weight*DIRECT_A3D_ELEM(v, Z0, Y0, X1) - DIRECT_A3D_ELEM(u, Z0, Y0, X1)*vd001) / (DIRECT_A3D_ELEM(v, Z0, Y0, X1)*DIRECT_A3D_ELEM(v, Z0, Y0, X1)));


			RDOUBLE v10d1 = iz * v0d1;
			RDOUBLE g10d1 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) / DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y0, X1))*((v10d1*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(u, Z1, Y0, X1)*v10d1) / (DIRECT_A3D_ELEM(v, Z1, Y0, X1)*DIRECT_A3D_ELEM(v, Z1, Y0, X1)));
			RDOUBLE v1d01 = iz * vd01;
			RDOUBLE g1d01 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) / DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y0, X1))*((v1d01*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(u, Z1, Y0, X1)*v1d01) / (DIRECT_A3D_ELEM(v, Z1, Y0, X1)*DIRECT_A3D_ELEM(v, Z1, Y0, X1)));
			RDOUBLE vd101 = 1 * v01;
			RDOUBLE gd101 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y0, X1) / DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y0, X1))*((vd101*weight*DIRECT_A3D_ELEM(v, Z1, Y0, X1) - DIRECT_A3D_ELEM(u, Z1, Y0, X1)*vd101) / (DIRECT_A3D_ELEM(v, Z1, Y0, X1)*DIRECT_A3D_ELEM(v, Z1, Y0, X1)));


			RDOUBLE v01d1 = (1.0f - iz) * v1d1;
			RDOUBLE g01d1 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) / DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y1, X1))*((v01d1*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(u, Z0, Y1, X1)*v01d1) / (DIRECT_A3D_ELEM(v, Z0, Y1, X1)*DIRECT_A3D_ELEM(v, Z0, Y1, X1)));
			RDOUBLE v0d11 = (1.0f - iz) * vd11;
			RDOUBLE g0d11 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) / DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y1, X1))*((v0d11*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(u, Z0, Y1, X1)*v0d11) / (DIRECT_A3D_ELEM(v, Z0, Y1, X1)*DIRECT_A3D_ELEM(v, Z0, Y1, X1)));
			RDOUBLE vd011 = (-1.0f) * v11;
			RDOUBLE gd011 = 2 * (DIRECT_A3D_ELEM(u, Z0, Y1, X1) / DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z0, Y1, X1))*((vd011*weight*DIRECT_A3D_ELEM(v, Z0, Y1, X1) - DIRECT_A3D_ELEM(u, Z0, Y1, X1)*vd011) / (DIRECT_A3D_ELEM(v, Z0, Y1, X1)*DIRECT_A3D_ELEM(v, Z0, Y1, X1)));


			RDOUBLE v11d1 = iz * v1d1;
			RDOUBLE g11d1 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) / DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y1, X1))*((v11d1*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(u, Z1, Y1, X1)*v11d1) / (DIRECT_A3D_ELEM(v, Z1, Y1, X1)*DIRECT_A3D_ELEM(v, Z1, Y1, X1)));
			RDOUBLE v1d11 = iz * vd11;
			RDOUBLE g1d11 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) / DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y1, X1))*((v1d11*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(u, Z1, Y1, X1)*v1d11) / (DIRECT_A3D_ELEM(v, Z1, Y1, X1)*DIRECT_A3D_ELEM(v, Z1, Y1, X1)));
			RDOUBLE vd111 = 1 * v11;
			RDOUBLE gd111 = 2 * (DIRECT_A3D_ELEM(u, Z1, Y1, X1) / DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(SuperRefVol, Z1, Y1, X1))*((vd111*weight*DIRECT_A3D_ELEM(v, Z1, Y1, X1) - DIRECT_A3D_ELEM(u, Z1, Y1, X1)*vd111) / (DIRECT_A3D_ELEM(v, Z1, Y1, X1)*DIRECT_A3D_ELEM(v, Z1, Y1, X1)));

			//
			//float3 movement = {
			//	 (v00d1 * v001 + v01d1 * v011 + v10d1 * v101 + v11d1 * v111) - (v00d0 * v000 + v01d0 * v010 + v10d0 * v100 + v11d0 * v110),
			//	 (v0d10 * v010 + v1d10 * v110 + v0d11 * v011 + v1d11 * v111) - (v1d00 * v100 + v0d00 * v000 + v0d01 * v001 + v1d01 * v101),
			//	 (vd100 * v100 + vd110 * v110 + vd101 * v101 + vd111 * v111) - (vd000 * v000 + vd010 * v010 + vd001 * v001 + vd011 * v011) };
			//
			float3 movement = { g00d0 + g00d1 + g01d0 + g01d1 + g10d0 + g10d1 + g11d0 + g11d1,
								g0d00 + g0d01 + g0d10 + g0d11 + g1d00 + g1d01 + g1d10 + g1d11,
								gd000 + gd001 + gd010 + gd011 + gd100 + gd101 + gd110 + gd111 };
			float movementLength = sqrt(movement.x*movement.x + movement.y*movement.y + movement.z*movement.z);
			if (movementLength > limit) {
				movement = movement * (limit / movementLength);
			}
			if (true)
				;
			newAtomPositions[p] = AtomPositions[p] - movement / Super;
			if (newAtomPositions[p].x*Super < (int)AtomPositions[p].x*Super)
				newAtomPositions[p].x = ((int)(AtomPositions[p].x*Super) + 1e-5) / Super;
			if (newAtomPositions[p].x*Super > (int)AtomPositions[p].x*Super + 1)
				newAtomPositions[p].x = ((int)(AtomPositions[p].x*Super + 1) - 1e-5) / Super;

			if (newAtomPositions[p].z*Super < (int)AtomPositions[p].z*Super)
				newAtomPositions[p].z = ((int)(AtomPositions[p].z*Super) + 1e-5) / Super;
			if (newAtomPositions[p].z*Super > (int)AtomPositions[p].z*Super + 1.0)
				newAtomPositions[p].z = (int)(AtomPositions[p].z + 1.0) / Super - 1e-5;

			if (newAtomPositions[p].y*Super < (int)AtomPositions[p].y*Super)
				newAtomPositions[p].y = ((int)(AtomPositions[p].y*Super) + 1e-5) / Super;
			if (newAtomPositions[p].y*Super > (int)AtomPositions[p].y*Super + 1)
				newAtomPositions[p].y = (int)(AtomPositions[p].y + 1.0) / Super - 1e-5;
		}
		AtomPositions = newAtomPositions;



		return;
	}
}
*/