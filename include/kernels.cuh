#pragma once 

namespace kernels{
	void MirrorPhaseFunction(int holoH, 
							 int holoW,  
							 int slmH,
							 int slmW,
							 int N, 
							 float *coordX_d, 
						     float *coordY_d, 
							 unsigned char *mpfPhase_d,
							 cufftComplex *holoField_d);
	
	void FFTMagnitude(int slmH, 
					  int slmW, 
					  cufftComplex *fftOutput_d, 
					  unsigned char *fftMagnitude_d);

	void VerticallyConcatenate(unsigned char *mpfPhase_d,
							   unsigned char *fftMagnitude_d,
							   unsigned char *combined_d,
							   int slmH, 
							   int slmW);
}

