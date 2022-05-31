#include "funcs.h"
#include "metadata_table.h"

RDOUBLE Lerp(RDOUBLE a, RDOUBLE b, RDOUBLE x)
{
	return a + (b - a) * x;
}


/*
RDOUBLE gaussian1D(RDOUBLE x, RDOUBLE sigma, RDOUBLE mu = 0)
{
	x -= mu;
	return exp(-0.5*((x / sigma)*(x / sigma)));
}*/


void Uproject_to_plane(const Matrix1D<RDOUBLE> &r,
	const Matrix2D<RDOUBLE> &euler, Matrix1D<RDOUBLE> &result)
{
	SPEED_UP_temps012;
	if (VEC_XSIZE(result) != 3)
		result.resize(3);
	M3x3_BY_V3x1(result, euler, r);
}

void Uproject_to_plane(const Matrix1D<RDOUBLE> &point,
	const Matrix1D<RDOUBLE> &direction, RDOUBLE distance,
	Matrix1D<RDOUBLE> &result)
{

	if (result.size() != 3)
		result.resize(3);
	RDOUBLE xx = distance - (XX(point) * XX(direction) + YY(point) * YY(direction) +
		ZZ(point) * ZZ(direction));
	XX(result) = XX(point) + xx * XX(direction);
	YY(result) = YY(point) + xx * YY(direction);
	ZZ(result) = ZZ(point) + xx * ZZ(direction);
}

void drawOneGaussian(MultidimArray<RDOUBLE> &gaussianTable, RDOUBLE boundary, RDOUBLE k, RDOUBLE i, RDOUBLE j, MultidimArray<RDOUBLE> &V, RDOUBLE intensity, RDOUBLE gaussFactor)
{

	int k0 = CEIL(XMIPP_MAX(STARTINGZ(V), k - boundary));
	int i0 = CEIL(XMIPP_MAX(STARTINGY(V), i - boundary));
	int j0 = CEIL(XMIPP_MAX(STARTINGX(V), j - boundary));
	int kF = FLOOR(XMIPP_MIN(FINISHINGZ(V), k + boundary));
	int iF = FLOOR(XMIPP_MIN(FINISHINGY(V), i + boundary));
	int jF = FLOOR(XMIPP_MIN(FINISHINGX(V), j + boundary));
	for (int kk = k0; kk <= kF; kk++)
	{
		RDOUBLE aux = kk - k;
		RDOUBLE diffkk2 = aux * aux;
		for (int ii = i0; ii <= iF; ii++)
		{
			aux = ii - i;
			RDOUBLE diffiikk2 = aux * aux + diffkk2;
			for (int jj = j0; jj <= jF; jj++)
			{
				aux = jj - j;
				RDOUBLE r = sqrt(diffiikk2 + aux * aux);
				aux = r * gaussFactor;
				long iaux = lround(aux);
				A3D_ELEM(V, kk, ii, jj) += intensity * DIRECT_A1D_ELEM(gaussianTable, iaux);
			}
		}
	}

}


void drawOneGaussian(Matrix1D<RDOUBLE> &gaussianTable, RDOUBLE boundary, RDOUBLE k, RDOUBLE i, RDOUBLE j, MultidimArray<RDOUBLE> &V, RDOUBLE intensity, RDOUBLE gaussFactor)
{

	int k0 = CEIL(XMIPP_MAX(STARTINGZ(V), k - boundary));
	int i0 = CEIL(XMIPP_MAX(STARTINGY(V), i - boundary));
	int j0 = CEIL(XMIPP_MAX(STARTINGX(V), j - boundary));
	int kF = FLOOR(XMIPP_MIN(FINISHINGZ(V), k + boundary));
	int iF = FLOOR(XMIPP_MIN(FINISHINGY(V), i + boundary));
	int jF = FLOOR(XMIPP_MIN(FINISHINGX(V), j + boundary));
	for (int kk = k0; kk <= kF; kk++)
	{
		RDOUBLE aux = kk - k;
		RDOUBLE diffkk2 = aux * aux;
		for (int ii = i0; ii <= iF; ii++)
		{
			aux = ii - i;
			RDOUBLE diffiikk2 = aux * aux + diffkk2;
			for (int jj = j0; jj <= jF; jj++)
			{
				aux = jj - j;
				RDOUBLE r = sqrt(diffiikk2 + aux * aux);
				aux = r * gaussFactor;
				long iaux = lround(aux);
				A3D_ELEM(V, kk, ii, jj) += intensity * VEC_ELEM(gaussianTable, iaux);
			}
		}
	}

}

void myGetFSC(MultidimArray< Complex > &FT1,
	MultidimArray< Complex > &FT2,
	MultidimArray< RDOUBLE > &fsc)
{
	if (!FT1.sameShape(FT2))
		REPORT_ERROR("fourierShellCorrelation ERROR: MultidimArrays have different shapes!");

	MultidimArray< int > radial_count(XSIZE(FT1));
	MultidimArray<double> num, den1, den2;
	Matrix1D<double> f(3);
	num.initZeros(radial_count);
	den1.initZeros(radial_count);
	den2.initZeros(radial_count);
	fsc.initZeros(radial_count);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(FT1)
	{
		int idx = ROUND(sqrt(kp*kp + ip * ip + jp * jp));
		if (idx >= XSIZE(FT1))
			continue;
		Complex z1 = DIRECT_A3D_ELEM(FT1, k, i, j);
		Complex z2 = DIRECT_A3D_ELEM(FT2, k, i, j);
		double absz1 = abs(z1);
		double absz2 = abs(z2);
		num(idx) += (conj(z1) * z2).real;
		den1(idx) += absz1 * absz1;
		den2(idx) += absz2 * absz2;
		radial_count(idx)++;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY1D(fsc)
	{
		auto den1Num = den1(i);
		auto den2Num = den2(i);
		auto numI = num(i);
		fsc(i) = (float)(num(i) / (sqrt(den1(i))*sqrt(den2(i))));
	}

}


void myGetFSC(MultidimArray< RDOUBLE > &m1,
	MultidimArray< RDOUBLE > &m2,
	MultidimArray< RDOUBLE > &fsc)
{
	MultidimArray< Complex > FT1, FT2;
	FourierTransformer transformer;
	transformer.FourierTransform(m1, FT1);
	transformer.FourierTransform(m2, FT2);
	myGetFSC(FT1, FT2, fsc);
}

MultidimArray<RDOUBLE> writeFSC(MultidimArray<RDOUBLE> &V1, MultidimArray<RDOUBLE> &V2, FileName outpath) {
	MultidimArray<RDOUBLE> fsc;
	MetaDataTable MDfsc;
	myGetFSC(V1, V2, fsc);
	MDfsc.setName("fsc");
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc)
	{
		MDfsc.addObject();
		MDfsc.setValue(EMDL_SPECTRAL_IDX, (int)i);
		MDfsc.setValue(EMDL_POSTPROCESS_FSC_GENERAL, DIRECT_A1D_ELEM(fsc, i));
	}
	MDfsc.write(outpath);
	return fsc;
}

float getLength(float3 f) {
	return sqrt(pow(f.x, 2) + pow(f.y, 2) + pow(f.z, 2));
}
