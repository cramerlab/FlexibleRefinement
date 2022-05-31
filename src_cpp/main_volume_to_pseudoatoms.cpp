#include "volume_to_pseudoatoms.h"

int main(int argc, char ** argv) {
	ProgVolumeToPseudoatoms prog = ProgVolumeToPseudoatoms(argc, argv);
	prog.printParameters();

	prog.run();
	prog.writeResults();
	return 0;
}
