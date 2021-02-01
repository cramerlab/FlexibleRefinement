// ImageHandler.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "liblionImports.h"
#include "readMRC.h"
#include "funcs.h"

using namespace relion;

int main(int argc, char **argv)
{
	FileName f1 = argv[1];
	FileName f2 = argv[2];
	MRCImage<RDOUBLE> im1 = MRCImage<RDOUBLE>::readAs(f1);
	MRCImage<RDOUBLE> im2 = MRCImage<RDOUBLE>::readAs(f2);
	
	FileName out;
	if (argc == 5) {
		FileName maskName = argv[3];
		out = argv[4];
		MRCImage<RDOUBLE> mask = MRCImage<RDOUBLE>::readAs(maskName);
		MultidimArray<RDOUBLE> masked1 = im1()*mask();
		MultidimArray<RDOUBLE> masked2 = im2()*mask();

		writeFSC(im1(), im2(), out + "_masked.fsc");
	}
	else
	{
		out = argv[3];
		writeFSC(im1(), im2(), out + ".fsc");
	}




	return EXIT_SUCCESS;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
