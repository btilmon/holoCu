#include <iostream>
#include <string.h>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <unistd.h>
#include <chrono>
using namespace std::chrono;

#include <cuda_runtime.h>
#include <cufft.h>

#include "cmdline.h"
#include "kernels.cuh"
#include "optics.h"
#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int cnt = 1;

int headless(Optics optics){ 

    optics.N = std::pow(2, cnt);
    optics.numImages = 10;

    // device data
    deviceBuffer coordX_d(optics.N),
                 coordY_d(optics.N),
                 holoReal_d(optics.slmH * optics.slmW),
                 holoImag_d(optics.slmH * optics.slmW),
                 fftReal_d(optics.slmH * optics.slmW),
                 fftImag_d(optics.slmH * optics.slmW);
                 
    deviceBufferUINT8 mpfPhase_d(optics.slmW * optics.slmH),
                      fftMagnitude_d(optics.slmH * optics.slmW),
                      combined_d(2 * optics.slmH * optics.slmW);

    deviceBufferComplex holoField_d(optics.slmW * optics.slmH),
                        fftOutput_d(optics.slmW * optics.slmH);
    
    // host data
    hostBuffer coordX_h(optics.N), 
               coordY_h(optics.N),
               holoReal_h(optics.slmH * optics.slmW),
               holoImag_h(optics.slmH * optics.slmW);
               
    hostBufferUINT8 mpfPhase_h(optics.slmW * optics.slmH),
                    fftMagnitude_h(optics.slmH * optics.slmW), 
                    combined_h(2 * optics.slmH * optics.slmW);

    // Create FFT plan for simulating hologram
    cufftHandle plan;
    cufftPlan2d(&plan, optics.slmH, optics.slmW, CUFFT_C2C);

    std::vector<double> phaseRuntimes, simRuntimes;

    for(int i = 0; i < optics.numImages; i++){

        auto startPhase = std::chrono::high_resolution_clock::now();

        // generate projector pattern
        generateProjector(i, coordX_h.data, coordY_h.data, optics);

        // transfer coordinates to device
        cudaMemcpy(coordX_d.data, coordX_h.data, optics.N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(coordY_d.data, coordY_h.data, optics.N * sizeof(float), cudaMemcpyHostToDevice);
        
        // compute hologram phase based on mirror phase function
        kernels::MirrorPhaseFunction(optics.holoH, 
                                     optics.holoW,  
                                     optics.slmH,
                                     optics.slmW,
                                     optics.N,
                                     coordX_d.data, 
                                     coordY_d.data, 
                                     mpfPhase_d.data,
                                     holoField_d.data);

        auto endPhase = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> phaseT = endPhase - startPhase;
        phaseRuntimes.push_back((phaseT).count());

        // Simulate hologram magnitude for visualization
        auto startSim = std::chrono::high_resolution_clock::now();
        cufftExecC2C(plan, holoField_d.data, fftOutput_d.data, CUFFT_FORWARD);
        kernels::FFTMagnitude(optics.slmH, optics.slmW, fftOutput_d.data, fftMagnitude_d.data);
        auto endSim = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> simT = endSim - startSim;
        simRuntimes.push_back((simT).count());

        // Save phase and amplitude 
        // vertically concatenate phase and simulated hologram
        kernels::VerticallyConcatenate(mpfPhase_d.data, 
                                       fftMagnitude_d.data, 
                                       combined_d.data, 
                                       optics.slmH, 
                                       optics.slmW);
        cudaMemcpy(combined_h.data, combined_d.data, sizeof(unsigned char) * 2 * optics.slmH * optics.slmW, cudaMemcpyDeviceToHost);
        stbi_write_png(("../data/" + optics.projector + ".png").c_str(), 
                        optics.slmW, optics.slmH * 2, 1, combined_h.data, optics.slmW);
        cnt++
    }
    
    float phaseTime = std::accumulate(phaseRuntimes.begin(), phaseRuntimes.end(), 0.0) / phaseRuntimes.size();
    float simTime = std::accumulate(simRuntimes.begin(), simRuntimes.end(), 0.0) / simRuntimes.size();

    std::cout << "Runtimes (ms):" << std::endl;
    std::cout << "--------------" << std::endl;
    std::cout << std::left << std::setw(22) << "Phase Computation: " << phaseTime << std::endl;
    std::cout << std::left << std::setw(22) << "Hologram Simulation: " << simTime << std::endl;
    std::cout << std::left << std::setw(22) << "Total: " << phaseTime + simTime << "\n" << std::endl;
}

int main(int argc, char *argv[]){

    cmdline::parser args;
    args.add<std::string>("projector", '\0', "which sensor to simulate", false, "adaptive");
    args.add<std::string>("display", '\0', "display either phase or simulated hologram", false, "phase");
    args.add<int>("N", '\0', "number of hologram points", false, 75);
    args.parse_check(argc, argv);

    Optics optics(args);
    benchmark(optics);

    return 0;
}