foo(){
  NATOMS=$1
  for SAMPLING in 1.0 2.0 3.0 4.0 8.0; do
	    echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation.fsc
	    D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation.fsc
	    echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_round2_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation_round2.fsc
	    D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_round2_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation_round2.fsc
	    echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_round3_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation_round3.fsc
	    D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_round3_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation_round3.fsc
	    echo D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_round4_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation_round4.fsc
	    D:/Software/FlexibleRefinement/x64/Release/ImageHandler.exe D:/EMD/9233/emd_9233_Scaled_1.5_masked.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_round4_approximation.mrc D:/EMD/9233/tmp/output_N${NATOMS}k_OS${SAMPLING}_approximation_round4.fsc
		
	done
  sleep 1
}

for N in 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000; do
	foo $N > D:/EMD/9233/tmp/${N}_eval.log &
done