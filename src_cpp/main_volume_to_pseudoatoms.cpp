/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "volume_to_pseudoatoms.h"
//#include <data/pdb.h>
#include <algorithm>
#include <stdio.h>
#include <list>
#include <filesystem>
#include "macros.h"
#include "funcs.h"
#define DEBUGFJ
namespace fs = std::filesystem;
 /* Pseudo atoms ------------------------------------------------------------ */
PseudoAtom::PseudoAtom()
{
	location.initZeros(3);
	intensity = 0;
}

bool operator <(const PseudoAtom &a, const PseudoAtom &b)
{
	return a.intensity < b.intensity;
}

std::ostream& operator << (std::ostream &o, const PseudoAtom &a)
{
	o << a.location.transpose() << " " << a.intensity;
	return o;
}

/* I/O --------------------------------------------------------------------- */


void ProgVolumeToPseudoatoms::show() const
{
	//if (verbose == 0)
	//	return;
	//std::cout << "Input volume:   " << fnVol << std::endl
	//	<< "Output volume:  " << fnOut << std::endl
	//	<< "Sigma:          " << sigma << std::endl
	//	<< "Initial seeds:  " << initialSeeds << std::endl
	//	<< "Grow seeds:     " << growSeeds << std::endl
	//	<< "Target error:   " << targetError << std::endl
	//	<< "Stop:           " << stop << std::endl
	//	<< "AllowMovement:  " << allowMovement << std::endl
	//	<< "AllowIntensity: " << allowIntensity << std::endl
	//	<< "Intensity Frac: " << intensityFraction << std::endl
	//	<< "Intensity Col:  " << intensityColumn << std::endl
	//	<< "Nclosest:       " << Nclosest << std::endl
	//	<< "Min. Distance:  " << minDistance << std::endl
	//	<< "Penalty:        " << penalty << std::endl
	//	<< "Threads:        " << numThreads << std::endl
	//	<< "Sampling Rate:  " << sampling << std::endl
	//	<< "Don't scale:    " << dontScale << std::endl
	//	<< "Binarize:       " << binarize << std::endl
	//	<< "Threshold:      " << threshold << std::endl
	//	;
	//if (useMask)
	//	mask_prm.show();
	//else
	//	std::cout << "No mask\n";
}


std::vector<float3> equidistantPoints(MultidimArray<int> maskArr, idxtype n, DOUBLE *R) {
	MRCImage<int> mask = MRCImage<int>(maskArr);
	mask.setZeroOrigin();
	float3 MaskCenter = mask.getCenterOfMass();
	int3 Dims = toInt3(mask().xdim, mask().ydim, mask().zdim);

	MultidimArray<float3> BestSolution;

	float a = 0, b = Dims.x / 2;
	(*R) = (a + b) / 2;
	float3 Offset = make_float3(0, 0, 0);
	std::vector<float3> InsideMask;
	int outerLim = 2;
	for (int o = 0; o < outerLim; o++)
	{
		for (int i = 0; i < 10; i++)
		{
			(*R) = (a + b) / 2;

			float Root3 = (float)sqrt(3);
			float ZTerm = (float)(2 * sqrt(6) / 3);
			float SpacingX = (*R) * 2;
			float SpacingY = Root3 * (*R);
			float SpacingZ = ZTerm * (*R);
			int3 DimsSphere = toInt3(std::min(512, (int)std::ceil(Dims.x / SpacingX)),
				std::min(512, (int)std::ceil(Dims.y / SpacingX)),
				std::min(512, (int)std::ceil(Dims.z / SpacingX)));
			BestSolution.resize(std::min(512, (int)std::ceil(Dims.z / SpacingX)), std::min(512, (int)std::ceil(Dims.y / SpacingX)), std::min(512, (int)std::ceil(Dims.x / SpacingX)));

			DOUBLE maxX = 0.0;
			DOUBLE maxXY = 0.0;
			DOUBLE maxXZ = 0.0;

			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(BestSolution)
			{
				DIRECT_A3D_ELEM(BestSolution, k, i, j) = make_float3((2 * j + (i + k) % 2) * (*R) + Offset.x, Root3 * (i + 1.0 / 3.0 * (k % 2))* (*R) + Offset.y, ZTerm * k* (*R) + Offset.z);
				if ((2 * j + (i + k) % 2) * (*R) + Offset.x > maxX) {
					maxX = (2 * j + (i + k) % 2) * (*R) + Offset.x;
					maxXY = Root3 * (i + 1.0 / 3.0 * (k % 2))* (*R) + Offset.y;
					maxXZ = ZTerm * k* (*R) + Offset.z;

				}
			}

			InsideMask.clear();
			InsideMask.reserve(BestSolution.nzyxdim);
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(BestSolution)
			{
				float3 c = DIRECT_A3D_ELEM(BestSolution, k, i, j);
				int3 ip = toInt3(DIRECT_A3D_ELEM(BestSolution, k, i, j));
				if (ip.x >= 0 && ip.x < Dims.x && ip.y >= 0 && ip.y < Dims.y && ip.z >= 0 && ip.z < Dims.z && DIRECT_A3D_ELEM(mask(),ip.z, ip.y, ip.x) > 0)
						InsideMask.emplace_back(DIRECT_A3D_ELEM(BestSolution, k, i, j));
			}
			if (InsideMask.size() == n)
				break;
			else if (InsideMask.size() < n)
				b = (*R);
			else
				a = (*R);
		}

		float3 CenterOfPoints = mean(InsideMask);
		Offset = MaskCenter - CenterOfPoints;
		if (o != outerLim - 1) {
			a = 0.8f * (*R);
			b = 1.2f * (*R);
		}
	}


	for (auto p : InsideMask) {
		p = p + Offset;
	}

	return InsideMask;
}

void ProgVolumeToPseudoatoms::produceSideInfo()
{
	sigma /= sampling;
	minDistance /= sampling;

	if (intensityColumn != "occupancy" && intensityColumn != "Bfactor")
		REPORT_ERROR( (std::string)"Unknown column: " + intensityColumn);

	Vin = MRCImage<DOUBLE>::readAs(std::string(fnVol));
	Vin().setXmippOrigin();

	if (doInputFilter) {
		bandpassFilter(Vin(), inputFilterThresh, (DOUBLE)0, (DOUBLE)5);
		Vin.writeAs<float>(fnVol.withoutExtension() + "_lowpass.mrc");
	}

	

	if (binarize)
		Vin().binarize(threshold, 0);

	if (fnOut == "")
		fnOut = fnVol.withoutExtension();

	Vcurrent().initZeros(Vin());
	Vcurrent().setXmippOrigin();
	mask_prm.generate_mask(Vin());

	sigma3 = 3 * sigma;
	gaussianTable.resize(CEIL(sigma3*sqrt(3.0) * 1000));
	FOR_ALL_ELEMENTS_IN_ARRAY1D(gaussianTable)
		gaussianTable(i) = gaussian1D(i / 1000.0, sigma);

	energyOriginal = 0;
	DOUBLE N = 0;
	DOUBLE minval = 1e38, maxval = -1e38;
	const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(Vin())
	{
		if (useMask && iMask3D(k, i, j) == 0)
			continue;
		DOUBLE v = Vin(k, i, j);
		energyOriginal += v * v;
		minval = XMIPP_MIN(minval, v);
		maxval = XMIPP_MAX(maxval, v);
		N++;
	}
	energyOriginal /= N;

	Histogram1D hist;
	if (useMask)
		compute_hist_within_binary_mask(iMask3D, Vin(), hist,
			minval, maxval, 200);
	else
		compute_hist(Vin(), hist, minval, maxval, 200);
	percentil1 = hist.percentil(1);
	if (percentil1 <= 0)
		percentil1 = maxval / 500;
	range = hist.percentil(99) - percentil1;

	/*if (XMIPP_EQUAL_ZERO(range))
		REPORT_ERROR(ERR_VALUE_INCORRECT, "Range cannot be zero.");*/

	smallAtom = range * intensityFraction;

	// Create threads
	//pthread_barrierattr_t barrieratt;
	pthread_barrier_init(&barrier, NULL, numThreads + 1);
	size_t size = sizeof(pthread_t);
	threadIds = (pthread_t *)malloc(numThreads * size);
	threadArgs = (Prog_Convert_Vol2Pseudo_ThreadParams *)
		malloc(numThreads * sizeof(Prog_Convert_Vol2Pseudo_ThreadParams));
	for (int i = 0; i < numThreads; i++)
	{
		threadArgs[i].myThreadID = i;
		threadArgs[i].parent = this;
		pthread_create((threadIds + i), NULL, optimizeCurrentAtomsThread,
			(void *)(threadArgs + i));
	}

	// Filter for the difference volume
	Filter.FilterBand = LOWPASS;
	Filter.FilterShape = REALGAUSSIAN;
	Filter.w1 = sigma;
	Filter.generateMask(Vin());
	Filter.do_generate_3dmask = false;
}

#ifdef NEVER_DEFINED
//#define DEBUG
void ProgVolumeToPseudoatoms::placeSeeds(int Nseeds)
{
	// Convolve the difference with the Gaussian to know
	// where it would be better to put a Gaussian
	FourierFilter Filter;
	Filter.FilterBand = LOWPASS;
	Filter.FilterShape = REALGAUSSIAN;
	Filter.w1 = sigma;
	Filter.generateMask(Vin());
	Filter.do_generate_3dmask = false;

	MultidimArray<DOUBLE> Vdiff = Vin();
	Vdiff -= Vcurrent();
	Filter.applyMaskSpace(Vdiff);

	// Place all seeds
	const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	for (int n = 0; n < Nseeds; n++)
	{
		// Look for the maximum error
		bool first = true;
		int kmax = 0, imax = 0, jmax = 0;
		DOUBLE maxVal = 0.;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(Vdiff)
		{
			if (useMask && A3D_ELEM(iMask3D, k, i, j) == 0)
				continue;
			DOUBLE voxel = A3D_ELEM(Vdiff, k, i, j);
			if (first || voxel > maxVal)
			{
				kmax = k;
				imax = i;
				jmax = j;
				maxVal = voxel;
				first = false;
			}
		}

		// Keep this as an atom
		PseudoAtom a;
		VEC_ELEM(a.location, 0) = kmax;
		VEC_ELEM(a.location, 1) = imax;
		VEC_ELEM(a.location, 2) = jmax;
		if (allowIntensity)
			a.intensity = maxVal;
		else
		{
			if (maxVal < smallAtom)
				break;
			a.intensity = smallAtom;
		}
		atoms.push_back(a);

		// Remove this density from the difference
		drawGaussian(kmax, imax, jmax, Vdiff, -a.intensity);

#ifdef DEBUG

		std::cout << "New atom: " << a << std::endl;
		VolumeXmipp save;
		save() = Vdiff;
		save.write("PPPDiff.vol");
		std::cout << "Press any key\n";
		char c;
		std::cin >> c;
#endif

	}
}
#undef DEBUG
#endif

class SeedCandidate
{
public:
	int k, i, j;
	DOUBLE v;
	bool operator < (const SeedCandidate& c) const { return v > c.v; }
};

void ProgVolumeToPseudoatoms::placeSeeds(int Nseeds)
{
#ifdef DEBUGFJ
	std::cout << "placeSeeds" << std::endl;
#endif
	// Convolve the difference with the Gaussian to know
	// where it would be better to put a Gaussian
	MultidimArray<DOUBLE> Vdiff = Vin();
	Vdiff -= Vcurrent();
	Filter.applyMaskSpace(Vdiff);

	// Look for the Nseeds
	const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	size_t idx = 0;
	std::list<SeedCandidate> candidateList;
	size_t listSize = 0;
	DOUBLE lastValue = 0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(Vdiff)
	{
		if (!useMask || useMask && DIRECT_MULTIDIM_ELEM(iMask3D, idx))
		{
			DOUBLE v = A3D_ELEM(Vdiff, k, i, j);
			if (listSize == 0 || v > lastValue)
			{
				// Check if there is any other candidate around
				bool found = false;
				for (std::list<SeedCandidate>::iterator iter = candidateList.begin(); iter != candidateList.end(); iter++)
					if (std::abs(iter->k - k) < sigma && std::abs(iter->i - i) < sigma && std::abs(iter->j - j) < sigma)
					{
						found = true;
						break;
					}
				if (!found)
				{
					SeedCandidate aux;
					aux.v = v;
					aux.k = k;
					aux.i = i;
					aux.j = j;
					candidateList.push_back(aux);
					candidateList.sort();
					listSize++;
					if (listSize > Nseeds)
					{
						candidateList.pop_back();
						listSize--;
					}
					lastValue = candidateList.back().v;
				}
			}
		}
		idx++;
	}

	// Place atoms
	for (std::list<SeedCandidate>::iterator iter = candidateList.begin(); iter != candidateList.end(); iter++)
	{
		PseudoAtom a;
		VEC_ELEM(a.location, 0) = iter->k;
		VEC_ELEM(a.location, 1) = iter->i;
		VEC_ELEM(a.location, 2) = iter->j;
		if (allowIntensity)
			a.intensity = iter->v;
		else
		{
			if (iter->v < smallAtom)
				break;
			a.intensity = smallAtom;
		}
		atoms.push_back(a);

		// Remove this density from the difference
		drawGaussian(iter->k, iter->i, iter->j, Vdiff, -a.intensity);
	}
#ifdef DEBUGFJ
	std::cout << "placeSeeds done" << std::endl;
#endif
}

/* Remove seeds ------------------------------------------------------------ */
void ProgVolumeToPseudoatoms::removeSeeds(int Nseeds)
{
#ifdef DEBUGFJ
	std::cout << "removeSeeds" << std::endl;
#endif
	int fromNegative = ROUND(Nseeds*0.5);
	int fromSmall = Nseeds - fromNegative;

	if (allowIntensity)
	{
		// Remove too small atoms
		std::sort(atoms.begin(), atoms.end());
		atoms.erase(atoms.begin(), atoms.begin() + fromSmall);
	}
	else
	{
		fromNegative = Nseeds;
		fromSmall = 0;
	}

	// Remove atoms from regions in which the error is too negative
	MultidimArray<DOUBLE> Vdiff = Vin();
	Vdiff -= Vcurrent();
	int alreadyRemoved = 0;
	DOUBLE vmin = Vdiff.computeMin();
	if (vmin < 0)
	{
		for (DOUBLE v = vmin + vmin / 20; v < 0; v -= vmin / 20)
		{
			size_t oldListSize;
			do
			{
				oldListSize = atoms.size();

				// Search for a point within a negative region
				bool found = false;
				int kneg = 0, ineg = 0, jneg = 0;
				const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
				for (int k = STARTINGZ(Vdiff); k <= FINISHINGZ(Vdiff) && !found; k++)
					for (int i = STARTINGY(Vdiff); i <= FINISHINGY(Vdiff) && !found; i++)
						for (int j = STARTINGX(Vdiff); j <= FINISHINGX(Vdiff) && !found; j++)
						{
							if (useMask && iMask3D(k, i, j) == 0)
								continue;
							if (A3D_ELEM(Vdiff, k, i, j) < v)
							{
								kneg = k;
								ineg = i;
								jneg = j;
								A3D_ELEM(Vdiff, k, i, j) = 0;
								found = true;
							}
						}

				// If found such a point, search for a nearby atom
				if (found)
				{
					// Search for atom
					idxtype nmax = atoms.size();
					for (idxtype n = 0; n < nmax; n++)
					{
						DOUBLE r =
							(kneg - atoms[n].location(0))*(kneg - atoms[n].location(0)) +
							(ineg - atoms[n].location(1))*(ineg - atoms[n].location(1)) +
							(jneg - atoms[n].location(2))*(jneg - atoms[n].location(2));
						r = sqrt(r);
						if (r < sigma3)
						{
							drawGaussian(atoms[n].location(0),
								atoms[n].location(1),
								atoms[n].location(2),
								Vdiff,
								atoms[n].intensity);
							atoms.erase(atoms.begin() + n);
							alreadyRemoved++;
							break;
						}
					}
				}
			} while (oldListSize > atoms.size() && alreadyRemoved < fromNegative);
			if (alreadyRemoved == fromNegative)
				break;
		}
	}

	removeTooCloseSeeds();
}

void ProgVolumeToPseudoatoms::removeTooCloseSeeds()
{
#ifdef DEBUGFJ
	std::cout << "removeTooCloseSeeds" << std::endl;
#endif
	// Remove atoms that are too close to each other
	if (minDistance > 0 && allowIntensity)
	{
		std::vector<int> toRemove;
		idxtype nmax = atoms.size();
		DOUBLE minDistance2 = minDistance * minDistance;
		for (int n1 = 0; n1 < nmax; n1++)
		{
			bool found = false;
			idxtype nn = 0, nnmax = toRemove.size();
			while (nn < nnmax)
			{
				if (toRemove[nn] == n1)
				{
					found = true;
					break;
				}
				else if (toRemove[nn] > n1)
					break;
				nn++;
			}
			if (found)
				continue;
			for (int n2 = n1 + 1; n2 < nmax; n2++)
			{
				nn = 0;
				found = false;
				while (nn < nnmax)
				{
					if (toRemove[nn] == n2)
					{
						found = true;
						break;
					}
					else if (toRemove[nn] > n2)
						break;
					nn++;
				}
				if (found)
					continue;
				DOUBLE diffZ = atoms[n1].location(0) - atoms[n2].location(0);
				DOUBLE diffY = atoms[n1].location(1) - atoms[n2].location(1);
				DOUBLE diffX = atoms[n1].location(2) - atoms[n2].location(2);
				DOUBLE d2 = diffZ * diffZ + diffY * diffY + diffX * diffX;
				if (d2 < minDistance2)
				{
					if (atoms[n1].intensity < atoms[n2].intensity)
					{
						toRemove.push_back(n1);
						break;
					}
					else
						toRemove.push_back(n2);
					std::sort(toRemove.begin(), toRemove.end());
				}
			}
		}
		for (idxtype n = toRemove.size(); n > 0; n--)
			atoms.erase(atoms.begin() + toRemove[n-1]);
	}
#ifdef DEBUGFJ
	std::cout << "removeTooCloseSeeds done" << std::endl;
#endif
}

/* Draw approximation ------------------------------------------------------ */
void ProgVolumeToPseudoatoms::drawApproximation()
{
#ifdef DEBUGFJ
	std::cout << "drawApproximation" << std::endl;
#endif
	Vcurrent().initZeros(Vin());
	Vcurrent.setZeroOrigin();
	idxtype nmax = atoms.size();
#ifdef DEBUGFJ
	std::cout << "drawGaussian in drawApproximation" << std::endl;
#endif
	for (idxtype n = 0; n < nmax; n++)
		drawGaussian(atoms[n].location(0), atoms[n].location(1),
			atoms[n].location(2), Vcurrent(), atoms[n].intensity);
#ifdef DEBUGFJ
	std::cout << "drawGaussian drawApproximation done" << std::endl;
#endif
	energyDiff = 0;
	DOUBLE N = 0;
	percentageDiff = 0;
	const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	const MultidimArray<DOUBLE> &mVcurrent = Vcurrent();
	const MultidimArray<DOUBLE> &mVin = Vin();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mVcurrent)
	{
		if (useMask && DIRECT_MULTIDIM_ELEM(iMask3D, n) == 0)
			continue;
		DOUBLE Vinv = DIRECT_MULTIDIM_ELEM(mVin, n);
		if (Vinv <= 0)
			continue;
		DOUBLE vdiff = Vinv - DIRECT_MULTIDIM_ELEM(mVcurrent, n);
		DOUBLE vperc = fabs(vdiff);
		energyDiff += vdiff * vdiff;
		percentageDiff += vperc;
		N++;
	}
	energyDiff /= N;
	percentageDiff /= (N*range);
#ifdef DEBUGFJ
	std::cout << "drawApproximation done" << std::endl;
#endif
}

/* Gaussian operations ----------------------------------------------------- */
DOUBLE ProgVolumeToPseudoatoms::computeAverage(int k, int i, int j,
	MultidimArray<DOUBLE> &V)
{
#ifdef DEBUG
	std::cout << "computeAverage" << std::endl;
#endif
	int k0 = std::max(STARTINGZ(V), (long int)floor(k - sigma3));
	int i0 = std::max(STARTINGY(V), (long int)floor(i - sigma3));
	int j0 = std::max(STARTINGX(V), (long int)floor(j - sigma3));
	int kF = std::min(FINISHINGZ(V), (long int)ceil(k + sigma3));
	int iF = std::min(FINISHINGY(V), (long int)ceil(i + sigma3));
	int jF = std::min(FINISHINGX(V), (long int)ceil(j + sigma3));
	DOUBLE sum = 0;
	for (int kk = k0; kk <= kF; kk++)
		for (int ii = i0; ii <= iF; ii++)
			for (int jj = j0; jj <= jF; jj++)
				sum += V(kk, ii, jj);
	return sum / ((kF - k0 + 1)*(iF - i0 + 1)*(jF - j0 + 1));
}

void ProgVolumeToPseudoatoms::drawGaussian(DOUBLE k, DOUBLE i, DOUBLE j,
	MultidimArray<DOUBLE> &V, DOUBLE intensity)
{

	drawOneGaussian(gaussianTable, sigma3, k, i, j, V, intensity);
	/*

	int k0 = CEIL(XMIPP_MAX(STARTINGZ(V), k - sigma3));
	int i0 = CEIL(XMIPP_MAX(STARTINGY(V), i - sigma3));
	int j0 = CEIL(XMIPP_MAX(STARTINGX(V), j - sigma3));
	int kF = FLOOR(XMIPP_MIN(FINISHINGZ(V), k + sigma3));
	int iF = FLOOR(XMIPP_MIN(FINISHINGY(V), i + sigma3));
	int jF = FLOOR(XMIPP_MIN(FINISHINGX(V), j + sigma3));
	for (int kk = k0; kk <= kF; kk++)
	{
		DOUBLE aux = kk - k;
		DOUBLE diffkk2 = aux * aux;
		for (int ii = i0; ii <= iF; ii++)
		{
			aux = ii - i;
			DOUBLE diffiikk2 = aux * aux + diffkk2;
			for (int jj = j0; jj <= jF; jj++)
			{
				aux = jj - j;
				DOUBLE r = sqrt(diffiikk2 + aux * aux);
				aux = r * 1000;
				long iaux = lround(aux);
				A3D_ELEM(V, kk, ii, jj) += intensity * DIRECT_A1D_ELEM(gaussianTable, iaux);
			}
		}
	}
	*/
}

void ProgVolumeToPseudoatoms::extractRegion(int idxGaussian,
	MultidimArray<DOUBLE> &region, bool extended) const
{
	DOUBLE k = atoms[idxGaussian].location(0);
	DOUBLE i = atoms[idxGaussian].location(1);
	DOUBLE j = atoms[idxGaussian].location(2);

	DOUBLE sigma3ToUse = sigma3;
	if (extended)
		sigma3ToUse += 1.5;

	int k0 = CEIL(XMIPP_MAX(STARTINGZ(Vcurrent()), k - sigma3ToUse));
	int i0 = CEIL(XMIPP_MAX(STARTINGY(Vcurrent()), i - sigma3ToUse));
	int j0 = CEIL(XMIPP_MAX(STARTINGX(Vcurrent()), j - sigma3ToUse));
	int kF = FLOOR(XMIPP_MIN(FINISHINGZ(Vcurrent()), k + sigma3ToUse));
	int iF = FLOOR(XMIPP_MIN(FINISHINGY(Vcurrent()), i + sigma3ToUse));
	int jF = FLOOR(XMIPP_MIN(FINISHINGX(Vcurrent()), j + sigma3ToUse));

	if ((kF - k0 + 1)*(iF - i0 + 1)*(jF - j0 + 1) == 2028)
		bool tmp = true;
	else
		bool tmp = false;
	region.resizeNoCopy(kF - k0 + 1, iF - i0 + 1, jF - j0 + 1);
	STARTINGZ(region) = k0;
	STARTINGY(region) = i0;
	STARTINGX(region) = j0;
	const MultidimArray<DOUBLE> &mVcurrent = Vcurrent();
	for (int k = k0; k <= kF; k++)
		for (int i = i0; i <= iF; i++)
			for (int j = j0; j <= jF; j++)
				A3D_ELEM(region, k, i, j) = A3D_ELEM(mVcurrent, k, i, j);
}

//#define DEBUG
DOUBLE ProgVolumeToPseudoatoms::evaluateRegion(const MultidimArray<DOUBLE> &region)
const
{
#ifdef DEBUG
	std::cout << "evaluateRegion" << std::endl;
#endif
	DOUBLE avgDiff = 0;
	DOUBLE N = 0;
	const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	const MultidimArray<DOUBLE> &mVin = Vin();
	/*
#ifdef DEBUG
	MRCImage <DOUBLE> save, save2;
	save().initZeros(region);
	save2().initZeros(region);
#endif
*/
	FOR_ALL_ELEMENTS_IN_ARRAY3D(region)
	{
		DOUBLE Vinv = A3D_ELEM(mVin, k, i, j);
		if (Vinv <= 0 || (useMask && A3D_ELEM(iMask3D, k, i, j) == 0))
			continue;
		/*
#ifdef DEBUG
		save(k, i, j) = Vinv;
		save2(k, i, j) = A3D_ELEM(region, k, i, j);
#endif
*/
		DOUBLE vdiff = A3D_ELEM(region, k, i, j) - Vinv;
		DOUBLE vperc = (vdiff < 0) ? -vdiff : penalty * vdiff;
		avgDiff += vperc;
/*
#ifdef DEBUG
		std::cout << "(k,i,j)=(" << k << "," << i << "," << j << ") toSimulate=" << Vinv << " simulated=" << A3D_ELEM(region, k, i, j) << " vdiff=" << vdiff << " vperc=" << vperc << std::endl;
#endif*/
		++N;
	}
	/*
#ifdef DEBUG
	save.write("PPPtoSimulate.vol");
	save2.write("PPPsimulated.vol");
	std::cout << "Error=" << avgDiff / (N*range) << std::endl;
	std::cout << "Press any key\n";
	char c; std::cin >> c;
#endif
*/
	return avgDiff / (N*range);
}
#undef DEBUG

void ProgVolumeToPseudoatoms::insertRegion(const MultidimArray<DOUBLE> &region)
{
	const MultidimArray<DOUBLE> &mVcurrent = Vcurrent();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(region)
		A3D_ELEM(mVcurrent, k, i, j) = A3D_ELEM(region, k, i, j);
}

/* Optimize ---------------------------------------------------------------- */
static pthread_mutex_t mutexUpdateVolume = PTHREAD_MUTEX_INITIALIZER;

//#define DEBUG
void* ProgVolumeToPseudoatoms::optimizeCurrentAtomsThread(
	void * threadArgs)
{
#ifdef DEBUGFJ
	std::cout << "optimizeCurrentAtomsThread" << std::endl;
#endif
	Prog_Convert_Vol2Pseudo_ThreadParams *myArgs =
		(Prog_Convert_Vol2Pseudo_ThreadParams *)threadArgs;
	ProgVolumeToPseudoatoms *parent = myArgs->parent;
	std::vector< PseudoAtom > &atoms = parent->atoms;
	bool allowIntensity = parent->allowIntensity;
	bool allowMovement = parent->allowMovement;
	MultidimArray<DOUBLE> region, regionBackup;

	pthread_barrier_t *barrier = &(parent->barrier);
	do
	{
		std::cout << "optimizeCurrentAtomsThread loop" << std::endl;
		pthread_barrier_wait(barrier);
		std::cout << "optimizeCurrentAtomsThread passed barrier" << std::endl;
		if (parent->threadOpCode == KILLTHREAD)
			return NULL;

		myArgs->Nintensity = 0;
		myArgs->Nmovement = 0;
		idxtype nmax = atoms.size();
		for (idxtype n = 0; n < nmax; n++)
		{
			if ((n + 1) % parent->numThreads != myArgs->myThreadID)
				continue;

			parent->extractRegion(n, region, true);
			DOUBLE currentRegionEval = parent->evaluateRegion(region);
			parent->drawGaussian(atoms[n].location(0), atoms[n].location(1),
				atoms[n].location(2), region, -atoms[n].intensity);
			regionBackup = region;

#ifdef DEBUG
			std::cout << "Atom n=" << n << " current intensity=" << atoms[n].intensity << " -> " << currentRegionEval << std::endl;
#endif
			// Change intensity
			if (allowIntensity)
			{
				// Try with a Gaussian that is of different intensity
				DOUBLE tryCoeffs[8] = { 0, 0.1, 0.2, 0.5, 0.9, 0.99, 1.01, 1.1 };
				DOUBLE bestRed = 0;
				int bestT = -1;
				for (int t = 0; t < 8; t++)
				{
					region = regionBackup;
					parent->drawGaussian(atoms[n].location(0),
						atoms[n].location(1), atoms[n].location(2), region,
						tryCoeffs[t] * atoms[n].intensity);
					DOUBLE trialRegionEval = parent->evaluateRegion(region);
					DOUBLE reduction = trialRegionEval - currentRegionEval;
					if (reduction < bestRed)
					{
						bestRed = reduction;
						bestT = t;
#ifdef DEBUG
						std::cout << "    better -> " << trialRegionEval << " (factor=" << tryCoeffs[t] << ")" << std::endl;
#endif
					}
				}
				if (bestT != -1)
				{
					atoms[n].intensity *= tryCoeffs[bestT];
					region = regionBackup;
					parent->drawGaussian(atoms[n].location(0), atoms[n].location(1),
						atoms[n].location(2), region, atoms[n].intensity);
					pthread_mutex_lock(&mutexUpdateVolume);
					parent->insertRegion(region);
					pthread_mutex_unlock(&mutexUpdateVolume);
					currentRegionEval = parent->evaluateRegion(region);
					parent->drawGaussian(atoms[n].location(0),
						atoms[n].location(1), atoms[n].location(2), region,
						-atoms[n].intensity);
					regionBackup = region;
#ifdef DEBUG
					std::cout << "    finally -> " << currentRegionEval << " (intensity=" << atoms[n].intensity << ")" << std::endl;
#endif
					myArgs->Nintensity++;
				}
			}

			// Change location
			if (allowMovement && atoms[n].intensity > 0)
			{
				DOUBLE tryX[6] = { -0.45,0.5, 0.0 ,0.0, 0.0 ,0.0 };
				DOUBLE tryY[6] = { 0.0 ,0.0,-0.45,0.5, 0.0 ,0.0 };
				DOUBLE tryZ[6] = { 0.0 ,0.0, 0.0 ,0.0,-0.45,0.5 };
				DOUBLE bestRed = 0;
				int bestT = -1;
				for (int t = 0; t < 6; t++)
				{
					region = regionBackup;
					parent->drawGaussian(atoms[n].location(0) + tryZ[t],
						atoms[n].location(1) + tryY[t],
						atoms[n].location(2) + tryX[t],
						region, atoms[n].intensity);
					DOUBLE trialRegionEval = parent->evaluateRegion(region);
					DOUBLE reduction = trialRegionEval - currentRegionEval;
					if (reduction < bestRed)
					{
						bestRed = reduction;
						bestT = t;
					}
				}
				if (bestT != -1)
				{
					atoms[n].location(0) += tryZ[bestT];
					atoms[n].location(1) += tryY[bestT];
					atoms[n].location(2) += tryX[bestT];
					region = regionBackup;
					parent->drawGaussian(atoms[n].location(0),
						atoms[n].location(1), atoms[n].location(2), region,
						atoms[n].intensity);
					pthread_mutex_lock(&mutexUpdateVolume);
					parent->insertRegion(region);
					pthread_mutex_unlock(&mutexUpdateVolume);
					myArgs->Nmovement++;
				}
			}
		}

		pthread_barrier_wait(barrier);
	} while (true);
}

void ProgVolumeToPseudoatoms::optimizeCurrentAtoms()
{
	if (!allowIntensity && !allowMovement)
		return;
	bool finished = false;
	int iter = 0;
	do
	{
#ifdef DEBUGFJ
		std::cout << "optimizeCurrentAtoms loop" << std::endl;
#endif
		DOUBLE oldError = percentageDiff;

		threadOpCode = WORKTHREAD;
		// Launch workers
		pthread_barrier_wait(&barrier);
		// Wait for workers to finish
		pthread_barrier_wait(&barrier);
#ifdef DEBUGFJ
		std::cout << "optimizeCurrentAtoms loop Retrieve results" << std::endl;
#endif
		// Retrieve results
		int Nintensity = 0;
		int Nmovement = 0;
		for (int i = 0; i < numThreads; i++)
		{
			Nintensity += threadArgs[i].Nintensity;
			Nmovement += threadArgs[i].Nmovement;
		}
#ifdef DEBUGFJ
		std::cout << "optimizeCurrentAtoms loop  Remove all the removed atoms" << std::endl;
#endif
		// Remove all the removed atoms
		idxtype nmax = atoms.size();
		for (idxtype n = nmax; n > 0; n--)
			if (atoms[n-1].intensity == 0)
				atoms.erase(atoms.begin() + n-1);
#ifdef DEBUGFJ
		std::cout << "optimizeCurrentAtoms loop drawApproximation" << std::endl;
#endif
		drawApproximation();
		/*if (verbose > 0)
			std::cout << "Iteration " << iter << " error= " << percentageDiff
			<< " Natoms= " << atoms.size()
			<< " Intensity= " << Nintensity
			<< " Location= " << Nmovement
			<< std::endl;*/

		if (iter > 0)
			if ((oldError - percentageDiff) / oldError < stop)
				finished = true;
		iter++;
	} while (!finished);
}

/* Write ------------------------------------------------------------------- */
void ProgVolumeToPseudoatoms::writeResults()
{
	// Compute the histogram of intensities
	MultidimArray<DOUBLE> intensities;
	intensities.initZeros(atoms.size());
	FOR_ALL_ELEMENTS_IN_ARRAY1D(intensities)
		intensities(i) = atoms[i].intensity;
	Histogram1D hist;
	compute_hist(intensities, hist, 0, intensities.computeMax(), 100);
	
	if (verbose >= 2)
	{
		Vcurrent.write(fnOut + "_approximation.vol");
		hist.write(fnOut + "_approximation.hist");
		
	// Save the difference
		MRCImage<DOUBLE> Vdiff(Vin.getHeader());
		Vcurrent.setXmippOrigin();
		Vdiff.setData( Vin() - Vcurrent());
		const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
		if (useMask && XSIZE(iMask3D) != 0)
			FOR_ALL_ELEMENTS_IN_ARRAY3D(Vdiff())
			if (!iMask3D(k, i, j))
				Vdiff(k, i, j) = 0;
		Vdiff.write(fnOut + "_rawDiff.vol");

		Vdiff.setData(Vdiff()/range);
		Vdiff.write(fnOut + "_relativeDiff.vol");
	}

	// Write the PDB
	DOUBLE minIntensity = intensities.computeMin();
	DOUBLE maxIntensity = intensities.computeMax();
	if (maxIntensity - minIntensity < 1e-4)
	{
		dontScale = true;
		allowIntensity = false;
	}
	DOUBLE a = 0.99 / (maxIntensity - minIntensity);
	if (dontScale)
		a = 1;

	FILE *fhOut = NULL;
	fhOut = fopen((fnOut + ".pdb").c_str(), "w");
	if (!fhOut)
		REPORT_ERROR(fnOut + ".pdb");
	idxtype nmax = atoms.size();
	idxtype col = 1;
	if (intensityColumn == "Bfactor")
		col = 2;
	fprintf(fhOut, "REMARK xmipp_volume_to_pseudoatoms\n");
	fprintf(fhOut, "REMARK fixedGaussian %f\n", sigma*sampling);
	fprintf(fhOut, "REMARK intensityColumn %s\n", intensityColumn.c_str());
	for (idxtype n = 0; n < nmax; n++)
	{
		DOUBLE intensity = 1.0;
		if (allowIntensity)
			intensity = 0.01 + ROUND(100 * a*(atoms[n].intensity - minIntensity)) / 100.0;
		if (col == 1)
			fprintf(fhOut,
				"ATOM  %5d DENS DENS%5d    %8.3f%8.3f%8.3f%6.2f     1      DENS\n",
				n + 1, n + 1,
				(float)(atoms[n].location(2)*sampling),
				(float)(atoms[n].location(1)*sampling),
				(float)(atoms[n].location(0)*sampling),
				(float)intensity);
		else
			fprintf(fhOut,
				"ATOM  %5d DENS DENS%5d    %8.3f%8.3f%8.3f     1%6.2f      DENS\n",
				n + 1, n + 1,
				(float)(atoms[n].location(2)*sampling),
				(float)(atoms[n].location(1)*sampling),
				(float)(atoms[n].location(0)*sampling),
				(float)intensity);
	}
	fclose(fhOut);

	//if (verbose >= 2)
	//{
	//	PDBPhantom pdb;
	//	pdb.read(fnOut + ".pdb");
	//	distanceHistogramPDB(pdb, Nclosest, -1, 200, hist);
	//	hist.write(fnOut + "_distance.hist");
	//}
}

/* Run --------------------------------------------------------------------- */
void ProgVolumeToPseudoatoms::run()
{
	produceSideInfo();
	int iter = 0;
	DOUBLE previousNAtoms = 0;
	percentageDiff = 1;
	DOUBLE actualGrowSeeds = 0.;
	do
	{
#ifdef DEBUGFJ
		std::cout << "Run() Loop " << iter << std::endl;
#endif
		// Place seeds
		if (iter == 0) {
#ifdef DEBUGFJ
			std::cout << "Placing Initial seeds" << std::endl;
#endif
			placeSeeds(initialSeeds);
		}

		else
		{
			idxtype Natoms = atoms.size();
			actualGrowSeeds = growSeeds * std::min(1.0, 0.1 + (percentageDiff - targetError) / targetError);
			removeSeeds(FLOOR(Natoms*(actualGrowSeeds / 2) / 100));
			placeSeeds(FLOOR(Natoms*actualGrowSeeds / 100));
		}
		drawApproximation();

		if (iter == 0 && verbose > 0)
			std::cout << "Initial error with " << atoms.size()
			<< " pseudo-atoms " << percentageDiff << std::endl;

		// Optimize seeds until convergence
		optimizeCurrentAtoms();
		if (verbose > 0)
			std::cout << "Error with " << atoms.size() << " pseudo-atoms "
			<< percentageDiff << std::endl;
		writeResults();
		iter++;

		if (fabs(previousNAtoms - atoms.size()) / atoms.size() < 0.01*actualGrowSeeds / 100)
		{
			std::cout << "The required precision cannot be attained\n"
				<< "Suggestion: Reduce sigma and/or minDistance\n"
				<< "Writing best approximation with current parameters\n";

			break;
		}
		previousNAtoms = atoms.size();
	} while (percentageDiff > targetError);
	removeTooCloseSeeds();
	writeResults();

	// Kill threads
	threadOpCode = KILLTHREAD;
	pthread_barrier_wait(&barrier);
	free(threadIds);
	free(threadArgs);
}


void ProgVolumeToPseudoatoms::defineParams(cxxopts::Options &options)
{

	options.add_options()
		("i,input",			  "Input",		cxxopts::value<std::string>(),								"Input Volume")
		("o,output",          "rootname",	cxxopts::value<std::string>(),								"Rootname for output")
		("sigma",			  "s",			cxxopts::value<DOUBLE>()->default_value("1.5"),				"Sigma of Gaussians used")
		("initialSeeds",	  "N",			cxxopts::value<size_t>()->default_value("300"),				"Initial number of Atoms")
		("growSeeds",		  "percentage", cxxopts::value<size_t>()->default_value("30"),				"Percentage of growth, At each iteration the smallest percentage/2 pseudoatoms will be removed, and percentage new pseudoatoms will be created.")
		("filterInput",		  "f",			cxxopts::value<DOUBLE>(),									"Low-pass filter input using this threshold")
		("stop",			  "p",			cxxopts::value<DOUBLE>()->default_value("0.001"),			"Stop criterion (0<p<1) for inner iterations. At each iteration the current number of gaussians will be optimized until the average error does not decrease at least this amount relative to the previous iteration.")
		("targetError",		  "p",			cxxopts::value<DOUBLE>()->default_value("0.02"),			"Finish when the average representation error is below this threshold (in percentage; by default, 2%)")
		("dontAllowMovement", "true",		cxxopts::value<bool>()->default_value("false"),				"Don't allow pseudoatoms to move")
		("dontAllowIntensity","f",			cxxopts::value<DOUBLE>()->default_value("0.01"),			"Don't allow pseudoatoms to change intensity. f determines the fraction of intensity")
		("intensityColumn",	  "s",			cxxopts::value<std::string>()->default_value("Bfactor"),	"Where to write the intensity in the PDB file")
		("Nclosest",		  "N",			cxxopts::value<size_t>()->default_value("3"),				"N closest atoms, it is used only for the distance histogram")
		("minDistance",		  "d",			cxxopts::value<DOUBLE>()->default_value("0.001"),			"Minimum distance between two pseudoatoms (in Angstroms). Set it to -1 to disable")
		("penalty",			  "p",			cxxopts::value<DOUBLE>()->default_value("10"),				"Penalty for overshooting")
		("sampling_rate",	  "Ts",			cxxopts::value<DOUBLE>()->default_value("1"),				"Sampling rate Angstroms/pixel")
		("dontScale",		  "true",		cxxopts::value<bool>()->default_value("false"),				"Don't scale atom weights in the PDB")
		("binarize",		  "threshold",	cxxopts::value<DOUBLE>()->default_value("0.5"),				"Binarize the volume")
		("thr",				  "t",			cxxopts::value<size_t>()->default_value("1"),				"Number of Threads")
		("mask",			  "mask_type",	cxxopts::value<std::string>(),								"Which mask type to use. Options are real_file and binary_file")
		("maskfile",		  "f",			cxxopts::value<std::string>(),								"Path of mask file")
		("center",			  "c",			cxxopts::value<std::vector<DOUBLE>>()->default_value("0,0,0"), "Center of Mask")
		("v,verbose",		  "v",			cxxopts::value<int>()->default_value("0"), "Verbosity Level");
}

void ProgVolumeToPseudoatoms::readParams(cxxopts::ParseResult &result)
{
	if (result.count("i"))
		fnVol = result["i"].as<std::string>();
	else
		throw cxxopts::OptionException("Input volume (-i/--input) must be given!");

	if (result.count("o"))
		fnOut = result["o"].as<std::string>();
	else {
		fnOut = fnVol.withoutExtension();
	}
	if (result.count("filterInput")) {
		doInputFilter = true;
		inputFilterThresh = result["filterInput"].as<DOUBLE>();
	}
	else {
		doInputFilter = false;
	}
	mask_prm.allowed_data_types = INT_MASK;
	useMask = result.count("mask");
	if (useMask)
		mask_prm.readParams(result);
	sigma = result["sigma"].as<DOUBLE>();
	targetError = result["targetError"].as<DOUBLE>() / 100.0;
	stop = result["stop"].as<DOUBLE>();
	initialSeeds = result["initialSeeds"].as<size_t>();
	growSeeds = result["growSeeds"].as<size_t>();
	allowMovement = !(result["dontAllowMovement"].as<bool>());
	allowIntensity = !(result.count("dontAllowIntensity"));

	if (!allowIntensity)
		intensityFraction = result["dontAllowIntensity"].as<DOUBLE>();

	intensityColumn = result["intensityColumn"].as<std::string>();
	Nclosest = result["Nclosest"].as<size_t>();
	minDistance = result["minDistance"].as<DOUBLE>();
	penalty = result["penalty"].as<DOUBLE>();
	numThreads = result["thr"].as<size_t>();
	sampling = result["sampling_rate"].as<DOUBLE>();
	dontScale = result.count("dontScale");
	binarize = result.count("binarize");
	if (binarize)
		threshold =result["binarize"].as<DOUBLE>();
	else
		threshold = 0;
	verbose = result["verbose"].as<int>();
}

void ProgVolumeToPseudoatoms::printParameters() {

	std::cout << "fnVol             " << fnVol << std::endl;
	std::cout << "fnOut             " << fnOut << std::endl;
	std::cout << "useMask           " << (useMask?"true":"false") << std::endl;
	std::cout << "sigma             " << sigma << std::endl;
	std::cout << "targetError       " << targetError << std::endl;
	std::cout << "stop              " << stop << std::endl;
	std::cout << "initialSeeds      " << initialSeeds << std::endl;
	std::cout << "growSeeds         " << growSeeds << std::endl;
	std::cout << "allowMovement     " << (allowMovement ? "true" : "false") << std::endl;
	std::cout << "allowIntensity    " << (allowIntensity ? "true" : "false") << std::endl;
	std::cout << "intensityFraction " << intensityFraction << std::endl;
	std::cout << "intensityColumn   " << intensityColumn << std::endl;
	std::cout << "Nclosest          " << Nclosest << std::endl;
	std::cout << "minDistance       " << minDistance << std::endl;
	std::cout << "penalty           " << penalty << std::endl;
	std::cout << "numThreads        " << numThreads << std::endl;
	std::cout << "sampling          " << sampling << std::endl;
	std::cout << "binarize          " << (binarize ? "true" : "false") << std::endl;
	std::cout << "threshold         " << threshold << std::endl;
}

ProgVolumeToPseudoatoms::ProgVolumeToPseudoatoms(int argc, char ** argv) {
	cxxopts::Options options(argv[0], " - example command line options");
	options
		.positional_help("[optional args]")
		.show_positional_help();
	defineParams(options);

	try
	{
		cxxopts::ParseResult result = options.parse(argc, argv);
		readParams(result);
	}
	catch (const cxxopts::OptionException& e)
	{
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}
	
	
}



int main(int argc, char ** argv) {
	ProgVolumeToPseudoatoms prog = 	ProgVolumeToPseudoatoms(argc, argv);
	prog.produceSideInfo();
	prog.printParameters();
	DOUBLE R = 0.0;
	auto res = equidistantPoints(prog.mask_prm.get_binary_mask(), 5000, &prog.sigma);
	for (auto v : res) {
		PseudoAtom a;
		VEC_ELEM(a.location, 0) = v.z;
		VEC_ELEM(a.location, 1) = v.y;
		VEC_ELEM(a.location, 2) = v.x; 
		a.intensity = 1; 
		
		prog.atoms.push_back(a);
	}
	prog.drawApproximation();
	prog.Vcurrent.writeAs<float>("D:\\EMPIAR\\Vcurrent.mrc");
	prog.writeResults();
	//prog.run();
	return 0;
}
