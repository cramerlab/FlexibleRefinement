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
#include "omp.h"

//#define DEBUGFJ
namespace fs = std::filesystem;
using namespace gtom;

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


void ProgVolumeToPseudoatoms::placeSeedsEquidistantPoints() {
	MRCImage<int> mask = MRCImage<int>(mask_prm.get_binary_mask());
	mask.setZeroOrigin();
	FR_float3 MaskCenter = mask.getCenterOfMass();
	int3 Dims = toInt3(mask().xdim, mask().ydim, mask().zdim);

	MultidimArray<FR_float3> BestSolution;

	float a = 0, b = Dims.x / 2;
	RDOUBLE R = (a + b) / 2;
	FR_float3 Offset = make_FR_float3(0, 0, 0);
	std::vector<FR_float3> InsideMask;
	int outerLim = 2;
	for (int o = 0; o < outerLim; o++)
	{
		for (int i = 0; i < 10; i++)
		{
			R = (a + b) / 2;

			float Root3 = (float)sqrt(3);
			float ZTerm = (float)(2 * sqrt(6) / 3);
			float SpacingX = R * 2;
			float SpacingY = Root3 * R;
			float SpacingZ = ZTerm * R;
			int3 DimsSphere = toInt3(std::min(512, (int)std::ceil(Dims.x / SpacingX)),
				std::min(512, (int)std::ceil(Dims.y / SpacingX)),
				std::min(512, (int)std::ceil(Dims.z / SpacingX)));
			BestSolution.resize(std::min(512, (int)std::ceil(Dims.z / SpacingX)), std::min(512, (int)std::ceil(Dims.y / SpacingX)), std::min(512, (int)std::ceil(Dims.x / SpacingX)));

			RDOUBLE maxX = 0.0;
			RDOUBLE maxXY = 0.0;
			RDOUBLE maxXZ = 0.0;

			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(BestSolution)
			{
				DIRECT_A3D_ELEM(BestSolution, k, i, j) = make_FR_float3((2 * j + (i + k) % 2) * R + Offset.x, Root3 * (i + 1.0 / 3.0 * (k % 2))* R + Offset.y, ZTerm * k* R + Offset.z);
				if ((2 * j + (i + k) % 2) * R + Offset.x > maxX) {
					maxX = (2 * j + (i + k) % 2) * R + Offset.x;
					maxXY = Root3 * (i + 1.0 / 3.0 * (k % 2))* R + Offset.y;
					maxXZ = ZTerm * k* R + Offset.z;

				}
			}

			InsideMask.clear();
			InsideMask.reserve(BestSolution.nzyxdim);
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(BestSolution)
			{
				FR_float3 c = DIRECT_A3D_ELEM(BestSolution, k, i, j);
				int3 ip = toInt3((int)c.x, (int) c.y, (int)c.z);
				if (ip.x >= 0 && ip.x < Dims.x && ip.y >= 0 && ip.y < Dims.y && ip.z >= 0 && ip.z < Dims.z && DIRECT_A3D_ELEM(mask(),ip.z, ip.y, ip.x) > 0)
						InsideMask.emplace_back(DIRECT_A3D_ELEM(BestSolution, k, i, j));
			}
			if (InsideMask.size() == initialSeeds)
				break;
			else if (InsideMask.size() < initialSeeds)
				b = R;
			else
				a = R;
		}

		FR_float3 CenterOfPoints = mean(InsideMask);
		Offset = MaskCenter - CenterOfPoints;
		if (o != outerLim - 1) {
			a = 0.8f * R;
			b = 1.2f * R;
		}
	}


	for (auto p : InsideMask) {
		p = p + Offset;
	}
	sigma = R;
	Atoms.AtomPositions.reserve(InsideMask.size());
	Atoms.AtomWeights.reserve(InsideMask.size());
	for (auto v : InsideMask) {
		float3 pos = { v.x, v.y, v.z };
		Atoms.AtomPositions.push_back(pos);
		Atoms.AtomWeights.push_back(1.0);
	}
	if(interpolateValues)
		Atoms.IntensityFromVolume(Vin(), super);
}

void ProgVolumeToPseudoatoms::run()
{

	if (intensityColumn != "occupancy" && intensityColumn != "Bfactor")
		REPORT_ERROR( (std::string)"Unknown column: " + intensityColumn);

	Vin = MRCImage<RDOUBLE>::readAs(std::string(fnVol));
	Vin.setZeroOrigin();

	if (doInputFilter) {
		bandpassFilter(Vin(), inputFilterThresh, (RDOUBLE)0, (RDOUBLE)5);
		Vin.writeAs<float>(fnVol.withoutExtension() + "_lowpass.mrc", true);
	}

	

	if (binarize)
		Vin().binarize(threshold, 0);

	if (fnOut == "")
		fnOut = fnVol.withoutExtension();

	Vcurrent().initZeros(Vin());
	Vcurrent.setZeroOrigin();
	mask_prm.generate_mask(Vin());

	placeSeedsEquidistantPoints();
	Atoms.RasterizeToVolume(Vcurrent(), make_int3(Vcurrent().xdim, Vcurrent().ydim, Vcurrent().zdim), super);
	sigma /= sampling;
	sigma3 = 3 * sigma;
	super = 4.0;
	gaussianTable.resize(CEIL(sigma*4*sqrt(3.0) * gaussFactor));
	FOR_ALL_ELEMENTS_IN_ARRAY1D(gaussianTable)
		gaussianTable(i) = gaussian1D(i / (RDOUBLE)gaussFactor, sigma);

	energyOriginal = 0;
	RDOUBLE N = 0;
	RDOUBLE minval = 1e38, maxval = -1e38;
	MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	iMask3D.xinit = iMask3D.yinit = iMask3D.zinit = 0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(Vin())
	{
		
		if (useMask && iMask3D(k, i, j) == 0)
			continue;
		RDOUBLE v = Vin(k, i, j);
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

	// Create threads
	//pthread_barrierattr_t barrieratt;

	// Filter for the difference volume
	Filter.FilterBand = LOWPASS;
	Filter.FilterShape = REALGAUSSIAN;
	Filter.w1 = sigma;
	Filter.generateMask(Vin());
	Filter.do_generate_3dmask = false;
}

/* Draw approximation ------------------------------------------------------ */
void ProgVolumeToPseudoatoms::drawApproximation()
{
#ifdef DEBUGFJ
	std::cout << "drawApproximation" << std::endl;
#endif

	Atoms.RasterizeToVolume(Vcurrent(), make_int3(Vcurrent().xdim, Vcurrent().ydim, Vcurrent().zdim), oversampling);

	energyDiff = 0;
	RDOUBLE N = 0;
	percentageDiff = 0;
	const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
	const MultidimArray<RDOUBLE> &mVcurrent = Vcurrent();
	const MultidimArray<RDOUBLE> &mVin = Vin();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mVcurrent)
	{
		if (useMask && DIRECT_MULTIDIM_ELEM(iMask3D, n) == 0)
			continue;
		RDOUBLE Vinv = DIRECT_MULTIDIM_ELEM(mVin, n);
		if (Vinv <= 0)
			continue;
		RDOUBLE vdiff = Vinv - DIRECT_MULTIDIM_ELEM(mVcurrent, n);
		RDOUBLE vperc = fabs(vdiff);
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
RDOUBLE ProgVolumeToPseudoatoms::computeAverage(int k, int i, int j,
	MultidimArray<RDOUBLE> &V)
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
	RDOUBLE sum = 0;
	for (int kk = k0; kk <= kF; kk++)
		for (int ii = i0; ii <= iF; ii++)
			for (int jj = j0; jj <= jF; jj++)
				sum += V(kk, ii, jj);
	return sum / ((kF - k0 + 1)*(iF - i0 + 1)*(jF - j0 + 1));
}

/* Write ------------------------------------------------------------------- */
void ProgVolumeToPseudoatoms::writeResults()
{
	// Compute the histogram of intensities
	MultidimArray<RDOUBLE> intensities;
	intensities.initZeros(Atoms.AtomWeights.size());
	FOR_ALL_ELEMENTS_IN_ARRAY1D(intensities)
		intensities(i) = Atoms.AtomWeights[i];
	Histogram1D hist;
	compute_hist(intensities, hist, 0, intensities.computeMax(), 100);
	
	if (verbose >= 2)
	{
		Vcurrent.writeAs<float>(fnOut + "_approximation.mrc", true);
		hist.write(fnOut + "_approximation.hist");
		
	// Save the difference
		MRCImage<RDOUBLE> Vdiff(Vin.getHeader());
		Vdiff.setZeroOrigin();
		Vdiff.setData( Vin() - Vcurrent());
		const MultidimArray<int> &iMask3D = mask_prm.get_binary_mask();
		if (useMask && XSIZE(iMask3D) != 0)
			FOR_ALL_ELEMENTS_IN_ARRAY3D(Vdiff())
			if (!iMask3D(k, i, j))
				Vdiff(k, i, j) = 0;
		Vdiff.writeAs<float>(fnOut + "_rawDiff.mrc", true);

		Vdiff.setData(Vdiff()/range);
		Vdiff.writeAs<float>(fnOut + "_relativeDiff.mrc", true);
	}

	// Write the PDB
	RDOUBLE minIntensity = intensities.computeMin();
	RDOUBLE maxIntensity = intensities.computeMax();
	if (maxIntensity - minIntensity < 1e-4)
	{
		dontScale = true;

	}
	RDOUBLE a = 0.99 / (maxIntensity - minIntensity);
	if (dontScale)
		a = 1;

	FILE *fhOut = NULL;
	fhOut = fopen((fnOut + ".pdb").c_str(), "w");
	if (!fhOut)
		REPORT_ERROR(fnOut + ".pdb");
	idxtype nmax = Atoms.AtomPositions.size();
	idxtype col = 1;
	if (intensityColumn == "Bfactor")
		col = 2;
	fprintf(fhOut, "REMARK xmipp_volume_to_pseudoatoms\n");
	fprintf(fhOut, "REMARK fixedGaussian %f\n", sigma*sampling);
	fprintf(fhOut, "REMARK intensityColumn %s\n", intensityColumn.c_str());
	for (idxtype n = 0; n < nmax; n++)
	{
		RDOUBLE intensity = 1.0;
		if (interpolateValues)
			intensity = Atoms.AtomWeights[n];
		if (col == 1)
			fprintf(fhOut,
				"ATOM  %8d DENS DENS %7d    %8.3f%8.3f%8.3f%14.10f     1      DENS\n",
				n + 1, n + 1,
				(float)(Atoms.AtomPositions[n].x),
				(float)(Atoms.AtomPositions[n].y),
				(float)(Atoms.AtomPositions[n].z),
				(float)intensity);
		else
			fprintf(fhOut,
				"ATOM  %8d DENS DENS %7d    %8.3f%8.3f%8.3f     1%14.10f      DENS\n",
				n + 1, n + 1,
				(float)(Atoms.AtomPositions[n].x),
				(float)(Atoms.AtomPositions[n].y),
				(float)(Atoms.AtomPositions[n].z),
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


void ProgVolumeToPseudoatoms::defineParams(cxxopts::Options &options)
{

	options.add_options()
		("i,input",			  "Input",		cxxopts::value<std::string>(),								"Input Volume")
		("o,output",          "rootname",	cxxopts::value<std::string>(),								"Rootname for output")
		("sigma",			  "s",			cxxopts::value<RDOUBLE>()->default_value("1.5"),				"Sigma of Gaussians used")
		("oversampling",	  "s",			cxxopts::value<RDOUBLE>()->default_value("1.0"),				"Oversampling used when mapping atoms back to cartesian grid")
		("initialSeeds",	  "N",			cxxopts::value<size_t>()->default_value("300"),				"Initial number of Atoms")
		("filterInput",		  "f",			cxxopts::value<RDOUBLE>(),									"Low-pass filter input using this threshold")
		("dontAllowMovement", "true",		cxxopts::value<bool>()->default_value("false"),				"Don't allow pseudoatoms to move")
		("dontAllowNumberChange", "false",		cxxopts::value<bool>()->default_value("false"),				"Don't allow pseudoatom numbers to change")
		("InterpolateValues", "false",		cxxopts::value<bool>()->default_value("false"),				"Interpolate Initial Atom intensities")
		("dontAllowIntensity","f",			cxxopts::value<bool>()->default_value("false"),				"Don't allow pseudoatoms to change intensity. ")
		("intensityColumn",	  "s",			cxxopts::value<std::string>()->default_value("Bfactor"),	"Where to write the intensity in the PDB file")
		("Nclosest",		  "N",			cxxopts::value<size_t>()->default_value("3"),				"N closest atoms, it is used only for the distance histogram")
		("dontScale",		  "true",		cxxopts::value<bool>()->default_value("false"),				"Don't scale atom weights in the PDB")
		("binarize",		  "threshold",	cxxopts::value<RDOUBLE>()->default_value("0.5"),				"Binarize the volume")
		("thr",				  "t",			cxxopts::value<size_t>()->default_value("1"),				"Number of Threads")
		("mask",			  "mask_type",	cxxopts::value<std::string>(),								"Which mask type to use. Options are real_file and binary_file")
		("maskfile",		  "f",			cxxopts::value<std::string>(),								"Path of mask file")
		("center",			  "c",			cxxopts::value<std::vector<RDOUBLE>>()->default_value("0,0,0"), "Center of Mask")
		("v,verbose",		  "v",			cxxopts::value<int>()->default_value("0"), "Verbosity Level");
}

void ProgVolumeToPseudoatoms::readParams(cxxopts::ParseResult &result)
{
	initialAlgo = EQUIDISTANT_PLACEMENT;

	nIter = 10;
	interpolateValues = false;
	oversampling = 1.0;
	super = 1.0;
	if (result.count("i"))
		fnVol = result["i"].as<std::string>();
	else
		throw cxxopts::OptionException("Input volume (-i/--input) must be given!");

	if (result.count("o"))
		fnOut = result["o"].as<std::string>();
	else {
		fnOut = fnVol.withoutExtension();
	}
	sigma = result["sigma"].as<RDOUBLE>();
	oversampling = result["oversampling"].as<RDOUBLE>();
	super = oversampling;
	initialSeeds = result["initialSeeds"].as<size_t>();
	if (result.count("filterInput")) {
		doInputFilter = true;
		inputFilterThresh = result["filterInput"].as<RDOUBLE>();
	}
	else {
		doInputFilter = false;
	}
	allowMovement = !(result["dontAllowMovement"].as<bool>());
	allowAtomNumber = !(result["dontAllowNumberChange"].as<bool>());
	interpolateValues = result["InterpolateValues"].as<bool>();
	allowIntensity = !(result["dontAllowIntensity"].as<bool>());


	intensityColumn = result["intensityColumn"].as<std::string>();
	Nclosest = result["Nclosest"].as<size_t>();
	dontScale = result.count("dontScale");
	binarize = result.count("binarize");
	if (binarize)
		threshold = result["binarize"].as<RDOUBLE>();
	else
		threshold = 0;

	numThreads = result["thr"].as<size_t>();
	omp_set_num_threads(numThreads);
	mask_prm.allowed_data_types = INT_MASK;
	useMask = result.count("mask");
	if (useMask)
		mask_prm.readParams(result);
	else
		throw cxxopts::OptionException("Mask (--mask) must be given!");

	verbose = result["verbose"].as<int>();
}

void ProgVolumeToPseudoatoms::printParameters() {

	std::cout << "fnVol             " << fnVol << std::endl;
	std::cout << "fnOut             " << fnOut << std::endl;
	std::cout << "sigma             " << sigma << std::endl;
	std::cout << "oversampling      " << (oversampling ?"true":"false") << std::endl;
	std::cout << "initialSeeds      " << initialSeeds << std::endl;
	std::cout << "filterInput       " << inputFilterThresh << std::endl;
	std::cout << "allowMovement     " << (allowMovement ? "true" : "false") << std::endl;
	std::cout << "allowAtomNumber   " << (allowAtomNumber ? "true" : "false") << std::endl;
	std::cout << "InterpolateValues " << (interpolateValues ? "true" : "false") << std::endl;
	std::cout << "allowIntensity    " << (allowIntensity ? "true" : "false") << std::endl;
	std::cout << "intensityColumn   " << intensityColumn << std::endl;
	std::cout << "Nclosest          " << Nclosest << std::endl;
	std::cout << "dontScale         " << (dontScale ? "true" : "false") << std::endl;
	std::cout << "binarize          " << (binarize ? "true" : "false") << std::endl;
	std::cout << "threshold         " << threshold << std::endl;
	std::cout << "threads           " << numThreads << std::endl;
	std::cout << "useMask           " << (useMask?"true":"false") << std::endl;
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
		std::cerr << options.help() <<std::endl;
		exit(1);
	}
	
	
}



int main(int argc, char ** argv) {
	ProgVolumeToPseudoatoms prog = 	ProgVolumeToPseudoatoms(argc, argv);
	prog.printParameters();

	prog.run();
	prog.writeResults();
	return 0;
}
