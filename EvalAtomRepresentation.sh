foo(){
  NATOMS=$1
  for SAMPLING in 2.0 3.0 4.0; do
	    echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation.fsc
	    D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation.fsc
	    #echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_round2_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation_round2.fsc
	    #D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_round2_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation_round2.fsc
	    #echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_round3_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation_round3.fsc
	    #D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_round3_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation_round3.fsc
	    #echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_round4_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation_round4.fsc
	    #D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_round4_approximation.mrc D:/EMD/9233/AtomNumberSweep/output_N${NATOMS}k_OS${SAMPLING}_approximation_round4.fsc
		
	done
  sleep 1
}

for N in 100 200 400  600  800 1000; do
	foo $N > D:/EMD/9233/AtomNumberSweep/${N}_eval.log &
done