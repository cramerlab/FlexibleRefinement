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
 /* This file contains functions related to the Radon Transform */

#ifndef _CONVERT_VOL2PSEUDO_HH
#define _CONVERT_VOL2PSEUDO_HH
#include "liblionImports.h"
#include "Types.h"
#include "readMRC.h"
#include "fourier_filter.h"
#include "my_mask.h"
#include "histogram.h"
#include "time.h"
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>

#include "cxxopts.hpp"
using namespace relion;

//#include "Types.h"
/*
#include <core/xmipp_image.h>
#include <data/mask.h>
#include <core/xmipp_threads.h>
#include <core/xmipp_program.h>
#include <vector>
#include <data/fourier_filter.h>
*/
/**@defgroup ConvertVol2Pseudo ConvertVol2Pseudo
   @ingroup ReconsLibrary */
   //@{
   /// Pseudoatom class
class PseudoAtom
{
public:
	/// Location
	Matrix1D<DOUBLE> location;

	/// Intensity
	DOUBLE           intensity;

	/// Empty constructor
	PseudoAtom();

	/// Show pseudo atom
	friend std::ostream& operator << (std::ostream &o, const PseudoAtom &f);
};

/// Comparison between pseudo atoms
bool operator <(const PseudoAtom &a, const PseudoAtom &b);

// Forward declaration
class ProgVolumeToPseudoatoms;

// Thread parameters
struct Prog_Convert_Vol2Pseudo_ThreadParams
{
	int myThreadID;
	ProgVolumeToPseudoatoms *parent;
	int Nintensity;
	int Nmovement;
};

enum placemenType { ORIGIANL_PLACEMENT=0, EQUIDISTANT_PLACEMENT=1 };

class ProgVolumeToPseudoatoms
{
public:
	/// Volume to convert
	FileName fnVol;

	/// Output volume
	FileName fnOut;

	int verbose = 0;

	// Mask
	Mask mask_prm;

	// Use mask
	bool useMask;

	/// Sigma
	DOUBLE sigma;

	DOUBLE oversampling;

	idxtype gaussFactor = 1000;

	/// Maximum error (as a percentage)
	DOUBLE targetError;

	/// Stop criterion
	DOUBLE stop;

	/// Initial seeds
	idxtype initialSeeds;

	idxtype NAtoms;

	/// Grow seeds
	DOUBLE growSeeds;

	/// Allow gaussians to move
	bool allowMovement;

	/// Allow gaussians to vary intensity
	bool allowIntensity;

	// Allow a variation of atom numbers
	bool allowAtomNumber;

	//Interpolate initial atom values
	bool interpolateValues;

	/** Intensity fraction.
		In case intensity is not allowed to change, this fraction
		is multiplied by the intensity range and all atoms will have
		this intensity value. */
	DOUBLE intensityFraction;

	/// Column for the intensity (if any)
	std::string intensityColumn;

	/// Mindistance
	DOUBLE minDistance;

	/// Penalization
	DOUBLE penalty;

	/// Number of threads
	idxtype numThreads;

	/// Sampling rate
	DOUBLE sampling;

	/// N closest atoms for the distance histogram
	size_t Nclosest;

	/// Don't scale the atom weights at the end
	bool dontScale;

	/// Binarize
	bool binarize;

	/// Threshold for the binarization
	DOUBLE threshold;

	placemenType initialAlgo;

public:

	ProgVolumeToPseudoatoms(int arg, char ** argv);

	/// Read parameters from command line
	void readParams(cxxopts::ParseResult &result);

	void printParameters();

	/// show parameters
	void show() const;

	/// define parameters
	void defineParams(cxxopts::Options &options);

	/// Prepare side info
	void produceSideInfo();

	/// Run
	void run();

	/// Place seeds
	void placeSeedsOriginal(int Nseeds);
	void placeSeedsEquidistantPoints();

	/// Remove seeds
	void removeSeeds(int Nseeds);

	/// Remove too close seeds
	void removeTooCloseSeeds();

	/// Compute average of a volume
	DOUBLE computeAverage(int k, int i, int j, MultidimArray<DOUBLE> &V);

	/// Draw a Gaussian on a volume
	void drawGaussian(DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity);

	/// Draw approximation
	void drawApproximation();

	/// Extract region around a Gaussian
	void extractRegion(int idxGaussian, MultidimArray<DOUBLE> &region,
		bool extended = false) const;

	/// Insert region
	void insertRegion(const MultidimArray<DOUBLE> &region);

	/// Evaluate region
	DOUBLE evaluateRegion(const MultidimArray<DOUBLE> &region) const;



	/// Optimize current atoms
	void optimizeCurrentAtoms();

	/// Optimize current atoms (thread)
	static void* optimizeCurrentAtomsThread(void * threadArgs);

	/// Write results
	void writeResults();

public:
	// Input volume
	MRCImage<DOUBLE> Vin;

	// Current approximation volume
	MRCImage<DOUBLE> Vcurrent;

	// Energy of the difference
	DOUBLE energyDiff;

	// Maximum percentage diffence
	DOUBLE percentageDiff;

	// Original energy
	DOUBLE energyOriginal;

	// List of atoms
	std::vector< PseudoAtom > atoms;

	// Maximum radius
	DOUBLE sigma3;

	// Percentil 1
	DOUBLE percentil1;

	// Range
	DOUBLE range;

	// Small atom intensity
	DOUBLE smallAtom;

	// Gaussian table
	MultidimArray<DOUBLE> gaussianTable;

	// Barrier
	pthread_barrier_t barrier;

	//FilterInput
	bool doInputFilter;

	//InputFilter Treshold
	DOUBLE inputFilterThresh;

#define KILLTHREAD -1
#define WORKTHREAD  0
	// Thread operation code
	int threadOpCode;

	// Pointer to thread arguments
	Prog_Convert_Vol2Pseudo_ThreadParams *threadArgs;

	// Pointer to threads
	pthread_t *threadIds;

	// Filter for the difference volume
	FourierFilter Filter;

	idxtype nIter;
};
//@}
#endif