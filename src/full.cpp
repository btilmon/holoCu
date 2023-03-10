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

// OpenGL / CUDA interop
#include <GL/glew.h>    // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library
#include "helpers.hpp"
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result

// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y){
	// create an OpenGL texture
	glGenTextures(1, gl_tex); // generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture, set GL_RED since we are displaying a 1 channel image on 3 channel display
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, size_x, size_y, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
	// Register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

int full(Optics optics){ 

     ////////////////////
    // Begin OpenGL Init
    ////////////////////
    // start GL context and O/S window using the GLFW helper library
    if (!glfwInit())
    {
        std::cerr << "ERROR: could not start GLFW3" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Retrieving monitors
    int count;
    GLFWmonitor **monitors = glfwGetMonitors(&count);
    for (int i = 0; i < count; ++i)
    {
        const char *name = glfwGetMonitorName(monitors[i]);
        std::cout << name << std::endl;
    }

    // Open window on secondary monitor
    GLFWwindow *window0 = glfwCreateWindow(optics.slmW, optics.slmH, "Monitor 0", monitors[0], NULL);
    if (!window0)
    {
        std::cerr << "ERROR: could not open window with GLFW3" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window0);
    glfwSwapInterval(0); // disable vsync

    // start GLEW extension handler
    // TODO: Do I need to call this every time I switch context?
    // https://stackoverflow.com/questions/35683334/call-glewinit-once-for-each-rendering-context-or-exactly-once-for-the-whole-app
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW\n";
        glfwTerminate();
        return -1;
    }

    // get version info
    const GLubyte *renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte *version = glGetString(GL_VERSION);   // version as a string
    // std::cout << "Renderer: " << renderer << std::endl;
    // std::cout << "OpenGL version supported " << version << std::endl; 

       // Common glfw settings
    // Ensure we can capture escape key
    glfwSetInputMode(window0, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited movement
    glfwSetInputMode(window0, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Common OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Get true width and height of the frame buffer
    int fbH, fbW;
    glfwGetFramebufferSize(window0, &fbW, &fbH);
    std::cout << "Frame buffer size: " << fbW << " x " <<  fbH << std::endl;
    if(fbW != optics.slmW || fbH != optics.slmH){
        std::cout << "\n---------------------------------------" << std::endl;
        std::cout << "WARNING: slmW and slmH must be same size as framebuffer (monitor/SLM) for accurate holographic projection, this only matters if using a real SLM." << std::endl;
        std::cout << "---------------------------------------\n" << std::endl;
        // return 0;
    }

    // find GPU
    findCudaDevice();

    // init cuda buffers
    const int input_bytes4k = optics.slmW * optics.slmH;

    // ------------- Preparing the scene -----------------
    // Vertices for the fullscreen quad
    const GLfloat QuadVboData[] = { 
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        };

    GLuint quadVao;
    glGenVertexArrays(1, &quadVao);
    glBindVertexArray(quadVao);
    GLuint quadVbo;
    glGenBuffers(1, &quadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(QuadVboData), QuadVboData, GL_STATIC_DRAW);
  
    // Compile GLSL program
    GLuint programID = loadShader("../src/passthrough.vert", 
                                    "../src/draw_tex.frag");


    // create texture that will receive the result of cuda kernel
	createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, optics.slmW, optics.slmH);

    // Connect with program
    GLuint texSampID = glGetUniformLocation(programID, "texIn");

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

    std::vector<double> phaseRuntimes, simRuntimes, renderRuntimes;

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

        ///////////////////////
        // Send to OpenGL
        ///////////////////////
        auto startRender = std::chrono::high_resolution_clock::now();
        // Map buffer objects to get CUDA device pointers
        cudaArray *texture_ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));
        int num_texels = optics.slmW * optics.slmH; // change to slm dims
        int num_values = num_texels;
        int size_tex_data = sizeof(GLubyte) * num_values;
        if(optics.display == "phase"){
            checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, mpfPhase_d.data, size_tex_data, cudaMemcpyDeviceToDevice));
        }
        if(optics.display == "hologram"){
            checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, fftMagnitude_d.data, size_tex_data, cudaMemcpyDeviceToDevice));
        }
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

        // Refreshing primary monitor
        glBindFramebuffer(GL_FRAMEBUFFER, 0); // Render to the screen
        glViewport(0, 0, optics.slmW, optics.slmH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(programID);
        // Bind and update texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);
        glUniform1i(texSampID, 0);
        // Attribute buffer: vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        // Draw the triangles
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisableVertexAttribArray(0);
        // Finish rendering
        glfwSwapBuffers(window0);
        glFinish();
        // Processing UI events
        glfwPollEvents();

        auto endRender = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> renderT = endRender - startRender;
        renderRuntimes.push_back((renderT).count());
        ///////////////////////
        // End Send to OpenGL
        ///////////////////////
    }
    
    float phaseTime = std::accumulate(phaseRuntimes.begin(), phaseRuntimes.end(), 0.0) / phaseRuntimes.size();
    float simTime = std::accumulate(simRuntimes.begin(), simRuntimes.end(), 0.0) / simRuntimes.size();
    float renderTime = std::accumulate(renderRuntimes.begin(), renderRuntimes.end(), 0.0) / renderRuntimes.size();

    std::cout << "Runtimes (ms):" << std::endl;
    std::cout << "--------------" << std::endl;
    std::cout << std::left << std::setw(22) << "Phase Computation: " << phaseTime << std::endl;
    std::cout << std::left << std::setw(22) << "Hologram Simulation: " << simTime << std::endl;
    std::cout << std::left << std::setw(22) << "Render to Screen: " << renderTime << std::endl;
    std::cout << std::left << std::setw(22) << "Total: " << phaseTime + simTime + renderTime << "\n" << std::endl;

    // OpenGL Cleanup
    glDeleteBuffers(1, &quadVbo);
    glDeleteVertexArrays(1, &quadVao);
    glDeleteProgram(programID);
    glfwTerminate();
}

int main(int argc, char *argv[]){

    cmdline::parser args;
    args.add<std::string>("projector", '\0', "which sensor to simulate", false, "adaptive");
    args.add<std::string>("display", '\0', "display either phase or simulated hologram", false, "phase");
    args.add<int>("N", '\0', "number of hologram points", false, 75);
    args.parse_check(argc, argv);

    Optics optics(args);
    full(optics);

    return 0;
}