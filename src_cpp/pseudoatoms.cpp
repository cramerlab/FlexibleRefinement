#include "pseudoatoms.h"
#include "readMRC.h"
using namespace relion;

void Pseudoatoms::RasterizeToVolume(MultidimArray<RDOUBLE>& vol, int3 Dims, RDOUBLE super, bool resize)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		vol.resizeNoCopy((long)(Dims.z * super + 0.5), (long)(Dims.y * super + 0.5), (long)(Dims.x * super + 0.5));
		vol.initZeros();
		//resizeMap(vol, (long)(Dims.z * super + 0.5));
		//ResizeMapGPU(vol, make_int3((int)(volSuper.xdim*super + 0.5), (int)(volSuper.ydim*super + 0.5), (int)(volSuper.zdim*super + 0.5)));


		MultidimArray<RDOUBLE> Weights((long)(Dims.z * super + 0.5), (long)(Dims.y * super + 0.5), (long)(Dims.x * super + 0.5));
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 superPos = AtomPositions[p] * super;

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

			A3D_ELEM(vol, Z0, Y0, X0) += AtomWeights[p] * v000;
			A3D_ELEM(vol, Z0, Y0, X1) += AtomWeights[p] * v001;
			A3D_ELEM(vol, Z0, Y1, X0) += AtomWeights[p] * v010;
			A3D_ELEM(vol, Z0, Y1, X1) += AtomWeights[p] * v011;

			A3D_ELEM(vol, Z1, Y0, X0) += AtomWeights[p] * v100;
			A3D_ELEM(vol, Z1, Y0, X1) += AtomWeights[p] * v101;
			A3D_ELEM(vol, Z1, Y1, X0) += AtomWeights[p] * v110;
			A3D_ELEM(vol, Z1, Y1, X1) += AtomWeights[p] * v111;

			A3D_ELEM(Weights, Z0, Y0, X0) += v000;
			A3D_ELEM(Weights, Z0, Y0, X1) += v001;
			A3D_ELEM(Weights, Z0, Y1, X0) += v010;
			A3D_ELEM(Weights, Z0, Y1, X1) += v011;

			A3D_ELEM(Weights, Z1, Y0, X0) += v100;
			A3D_ELEM(Weights, Z1, Y0, X1) += v101;
			A3D_ELEM(Weights, Z1, Y1, X0) += v110;
			A3D_ELEM(Weights, Z1, Y1, X1) += v111;
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(vol) {
			A3D_ELEM(vol, k, i, j) /= std::max((RDOUBLE)(1e-10), A3D_ELEM(Weights, k, i, j));
		}

		Weights.coreDeallocate();
		if(resize)
			ResizeMapGPU(vol, Dims);
		return;
	}
	else if (Mode == ATOM_GAUSSIAN) {
		resizeMap(vol, (long)(Dims.z * super + 0.5));
		auto itPos = AtomPositions.begin();
		auto itWeight = AtomWeights.begin();

		for (int n = 0; n < AtomPositions.size(); n++) {
			drawOneGaussian(GaussianTable, 4 * Sigma*super, (AtomPositions[n].z)*super, (AtomPositions[n].y)*super, (AtomPositions[n].x)*super, vol, AtomWeights[n], GaussFactor / super);
		}

		if (super > 1.0) {
			resizeMap(vol, Dims.x);
		}
	}
}


void Pseudoatoms::MoveAtoms(MultidimArray<RDOUBLE>& superRefVol, int3 Dims, RDOUBLE super, bool resize)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		MultidimArray<float> SuperRastered;
		RasterizeToVolume(SuperRastered,Dims,super, false);
		MultidimArray<float> SuperDiffVol = SuperRastered;

		Substract_GPU(SuperDiffVol, superRefVol);
		{
			MRCImage<float> DiffIm(SuperDiffVol);
			DiffIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_SuperDiffIm.mrc", true);
		}

		{
			MRCImage<float> SuperRefIm(superRefVol);
			SuperRefIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_SuperRefVol.mrc", true);
			MRCImage<float> SuperRasteredIm(SuperRastered);
			SuperRasteredIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_rasteredVol.mrc", true);

		}
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 superPos = AtomPositions[p] * super;

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
			RDOUBLE vd0 = -1;
			RDOUBLE v1 = ix;
			RDOUBLE vd1 = 1;

			RDOUBLE v00 = (1.0f - iy) * v0;
			RDOUBLE v0d0 = (1.0f - iy) * vd0;
			RDOUBLE vd00 = - 1 * v0;

			RDOUBLE v10 = iy * v0;
			RDOUBLE v1d0 = iy * vd0;
			RDOUBLE vd10 = 1 * v0;


			RDOUBLE v01 = (1.0f - iy) * v1;
			RDOUBLE v0d1 = (1.0f - iy) * vd1;
			RDOUBLE vd01 = ( - 1) * v1;

			RDOUBLE v11 = iy * v1;
			RDOUBLE v1d1 = iy * vd1;
			RDOUBLE vd11 = 1 * v1;

			RDOUBLE v000 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y0, X0);
			RDOUBLE v00d0 = (1.0f - iz) * v0d0;
			RDOUBLE v0d00 = (1.0f - iz) * vd00;
			RDOUBLE vd000 = (- 1) * v00;

			RDOUBLE v100 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y0, X0);
			RDOUBLE v10d0 = iz * v0d0;
			RDOUBLE v1d00 = iz * vd00;
			RDOUBLE vd100 = 1 * v00;

			RDOUBLE v010 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y1, X0);
			RDOUBLE v01d0 = (1.0f - iz) * v1d0;
			RDOUBLE v0d10 = (1.0f - iz) * vd10;
			RDOUBLE vd010 = (-1) * v10;

			RDOUBLE v110 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y1, X0);
			RDOUBLE v11d0 = iz * v1d0;
			RDOUBLE v1d10 = iz * vd10;
			RDOUBLE vd110 = 1 * v10;

			RDOUBLE v001 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y0, X1);
			RDOUBLE v00d1 = (1.0f - iz) * v0d1;
			RDOUBLE v0d01 = (1.0f - iz) * vd01;
			RDOUBLE vd001 = (-1.0f) * v01;

			RDOUBLE v101 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y0, X1);
			RDOUBLE v10d1 = iz * v0d1;
			RDOUBLE v1d01 = iz * vd01;
			RDOUBLE vd101 = 1 * v01;

			RDOUBLE v011 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y1, X1);
			RDOUBLE v01d1 = (1.0f - iz) * v1d1;
			RDOUBLE v0d11 = (1.0f - iz) * vd11;
			RDOUBLE vd011 = (-1.0f) * v11;

			RDOUBLE v111 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y1, X1);
			RDOUBLE v11d1 = iz * v1d1;
			RDOUBLE v1d11 = iz * vd11;
			RDOUBLE vd111 = 1 * v11;

			float3 movement = {
				 (v00d1 * v001 + v01d1 * v011 + v10d1 * v101 + v11d1 * v111) - (v00d0 * v000 + v01d0 * v010 + v10d0 * v100 + v11d0 * v110),
				 (v0d10 * v010 + v1d10 * v110 + v0d11 * v011 + v1d11 * v111) - (v1d00 * v100 + v0d00 * v000 + v0d01 * v001 + v1d01 * v101),
				 (vd100 * v100 + vd110 * v110 + vd101 * v101 + vd111 * v111) - (vd000 * v000 + vd010 * v010 + vd001 * v001 + vd011 * v011) };
			if (true)
				;

		}



		return;
	}

}

void Pseudoatoms::IntensityFromVolume(MultidimArray<RDOUBLE>& vol, RDOUBLE super)
{
	AtomWeights.clear();
	AtomWeights.resize(AtomPositions.size());
	MultidimArray<RDOUBLE> volSuper = vol;
	ResizeMapGPU(volSuper, make_int3((int)(volSuper.xdim*super + 0.5), (int)(volSuper.ydim*super + 0.5), (int)(volSuper.zdim*super + 0.5)));

#pragma omp parallel for
	for (int p = 0; p < AtomPositions.size(); p++)
	{
		float3 superPos = AtomPositions[p] * super;

		int X0 = (int)(superPos.x);
		RDOUBLE ix = (superPos.x) - X0;
		int X1 = X0 + 1;

		int Y0 = (int)(superPos.y);
		RDOUBLE iy = (superPos.y) - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)(superPos.z);
		RDOUBLE iz = (superPos.z) - Z0;
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
