#include <cufft.h>
#include <cuComplex.h>

namespace kernels {

    #define PI 3.141592654f

    // Mirror Phase Function Kernel
    __global__ void MirrorPhaseFunctionKernel(int holoH, 
                                              int holoW,  
                                              int slmH,
                                              int slmW,
                                              int N,
                                              float *coordX_d, 
                                              float *coordY_d, 
                                              unsigned char *mpfPhase_d,
                                              cufftComplex *holoField_d){  
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        // tweak xScale and yScale for your SLM
        float xScale = 1.0f / (holoW / 6);
        float yScale = 1.0f / (holoH / 6);
        float piscale = 1 / (2.0 * PI);

        // offset hologram to center of SLM
        int idx = slmW * (y + (slmH - holoH)/2) + (x + (slmW - holoW)/2); 
        float slope = 0, realSum = 0, imagSum = 0;
        if (y < holoH && x < holoW) {
            for (int k = 0; k < N; k++) {
                // compute mirror phase slope
                float xn = x - holoW/2; 
                float yn = y - holoH/2;
                slope = (xn * coordX_d[k] * xScale + yn * coordY_d[k] * yScale);
                // add real and imaginary components of complex exponential for each hologram point
                imagSum += __sinf(slope);
                realSum += __cosf(slope);
            }
            // compute phase of summed complex exponential and normalize to [0, 2pi]
            float phase = atan2(imagSum, realSum);
            
            // simulate hologram
            holoField_d[idx].x = __cosf(phase);
            holoField_d[idx].y = __sinf(phase); 
            
            // normalize phase to uint8 for display on SLM
            mpfPhase_d[idx] = (unsigned char)(255.0 * (phase + PI) * piscale);
        }
    }

    void MirrorPhaseFunction(int holoH, 
                             int holoW,  
                             int slmH,
                             int slmW,
                             int N, 
                             float *coordX_d, 
                             float *coordY_d, 
                             unsigned char *mpfPhase_d,
                             cufftComplex *holoField_d){

        int nt = 32;
        dim3 numThreads(nt, nt);
        int x = (holoW + numThreads.x - 1) / numThreads.x;
        int y = (holoH + numThreads.y - 1) / numThreads.y;
        dim3 numBlocks(x, y);

        MirrorPhaseFunctionKernel<<<numBlocks, numThreads>>>(holoH, 
                                                            holoW, 
                                                            slmH, 
                                                            slmW, 
                                                            N, 
                                                            coordX_d, 
                                                            coordY_d, 
                                                            mpfPhase_d,
                                                            holoField_d);
        cudaDeviceSynchronize();  
    }

    __global__ void FFTMagnitudeKernel(int slmH, 
                                       int slmW, 
                                       cufftComplex *fftOutput_d, 
                                       unsigned char *fftMagnitude_d){
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx = slmW * y + x;    
        int centerX = slmW / 2;
        int centerY = slmH / 2;               
        if (y < slmH && x < slmW) {
            cufftComplex shiftedValue = fftOutput_d[((y + centerY) % slmH) * slmW + (x + centerX) % slmW];
            fftMagnitude_d[idx] = (unsigned char)(255.0f * cuCabsf(shiftedValue) / 20000.0f);
        }
    }
    void FFTMagnitude(int slmH, 
                      int slmW, 
                      cufftComplex *fftOutput_d, 
                      unsigned char *fftMagnitude_d){

        int nt = 32;
        dim3 numThreads(nt, nt);
        int x = (slmW + numThreads.x - 1) / numThreads.x;
        int y = (slmH + numThreads.y - 1) / numThreads.y;
        dim3 numBlocks(x, y);

        FFTMagnitudeKernel<<<numBlocks, numThreads>>>(slmH, 
                                                      slmW,  
                                                      fftOutput_d, 
                                                      fftMagnitude_d);
        cudaDeviceSynchronize();  
    }

    __global__ void VerticallyConcatenateKernel(unsigned char *mpfPhase_d,
                                                unsigned char *fftMagnitude_d,
                                                unsigned char *combined_d,
                                                int slmH, 
                                                int slmW){
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx = slmW * y + x;    
        if (y < (2 * slmH) && x < slmW) {
            if(y < slmH){
                combined_d[idx] = mpfPhase_d[idx];
            }
            else{
                combined_d[idx] = fftMagnitude_d[(y - slmH) * slmW + x];
            }
            // border for visualization
            if(y < (slmH + 1) && y > (slmH - 1)){
                combined_d[idx] = 255.;
            }
        }
    }

    void VerticallyConcatenate(unsigned char *mpfPhase_d,
                               unsigned char *fftMagnitude_d,
                               unsigned char *combined_d,
                               int slmH, 
                               int slmW){
        int nt = 32;
        dim3 numThreads(nt, nt);
        int x = (slmW + numThreads.x - 1) / numThreads.x;
        int y = (2 * slmH + numThreads.y - 1) / numThreads.y;
        dim3 numBlocks(x, y);

        VerticallyConcatenateKernel<<<numBlocks, numThreads>>>(mpfPhase_d, 
                                                              fftMagnitude_d,  
                                                              combined_d, 
                                                              slmH, 
                                                              slmW);
        cudaDeviceSynchronize();  
    }
}

