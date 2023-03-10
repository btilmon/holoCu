#pragma once

struct deviceBuffer{
	deviceBuffer() : data(nullptr){}
	deviceBuffer(int count){ 
        cudaMalloc(&data, sizeof(float) * count); 
        cudaMemset(data, 0, sizeof(float) * count);
    }
	~deviceBuffer(){cudaFree(data);}
	float *data;
};

struct hostBuffer{
    hostBuffer() : data(nullptr){}
    hostBuffer(int count){ 
        data = new float [count]; 
        memset(data, 0, sizeof(float) * count);
    }
    ~hostBuffer(){delete[] data;}
    float *data;
};

struct deviceBufferUINT8{
	deviceBufferUINT8() : data(nullptr){}
	deviceBufferUINT8(int count){ 
        cudaMalloc(&data, sizeof(unsigned char) * count); 
        cudaMemset(data, 0, sizeof(unsigned char) * count);
    }
	~deviceBufferUINT8(){cudaFree(data);}
	unsigned char *data;
};

struct hostBufferUINT8{
    hostBufferUINT8() : data(nullptr){}
    hostBufferUINT8(int count){ 
        data = new unsigned char [count]; 
        memset(data, 0, sizeof(unsigned char) * count);
    }
    ~hostBufferUINT8(){delete[] data;}
    unsigned char *data;
};

struct deviceBufferComplex {
    deviceBufferComplex() : data(nullptr){}
    deviceBufferComplex(int count){ 
        cudaMalloc(&data, sizeof(cufftComplex) * count); 
        cudaMemset(data, 0, sizeof(cufftComplex) * count);
    }
    ~deviceBufferComplex(){cudaFree(data);}
    cufftComplex *data;
};

void printGpuProperties(cudaDeviceProp gpuProp){
    int count;
    cudaGetDeviceCount( &count );
    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties( &gpuProp, i );
        printf( " --- General Information for device %d ---\n", i );
        printf( "Name: %s\n", gpuProp.name );
        printf( "Compute capability: %d.%d\n", gpuProp.major, gpuProp.minor );
        printf( "Clock rate: %d\n", gpuProp.clockRate );
        printf( "Device copy overlap: " );
        if (gpuProp.deviceOverlap)
        printf( "Enabled\n" );
        else
        printf( "Disabled\n" );
        printf( "Kernel execition timeout : " );
        if (gpuProp.kernelExecTimeoutEnabled)
        printf( "Enabled\n" );
        else
        printf( "Disabled\n" );
        printf( " --- Memory Information for device %d ---\n", i );
        printf( "Total global mem: %ld\n", gpuProp.totalGlobalMem );
        printf( "Total constant Mem: %ld\n", gpuProp.totalConstMem );
        printf( "Max mem pitch: %ld\n", gpuProp.memPitch );
        printf( "Texture Alignment: %ld\n", gpuProp.textureAlignment );
        
        printf( " --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count: %d\n",
        gpuProp.multiProcessorCount );
        printf( "Shared mem per mp: %ld\n", gpuProp.sharedMemPerBlock );
        printf( "Registers per mp: %d\n", gpuProp.regsPerBlock );
        printf( "Threads in warp: %d\n", gpuProp.warpSize );
        printf( "Max threads per block: %d\n",
        gpuProp.maxThreadsPerBlock );
        printf( "Max thread dimensions: (%d, %d, %d)\n",
        gpuProp.maxThreadsDim[0], gpuProp.maxThreadsDim[1],
        gpuProp.maxThreadsDim[2] );
        printf( "Max grid dimensions: (%d, %d, %d)\n",
        gpuProp.maxGridSize[0], gpuProp.maxGridSize[1],
        gpuProp.maxGridSize[2] );
        printf( "\n" );
    }
}

std::unordered_multiset<int> pickSet(int N, int k, std::mt19937& gen){
    std::uniform_int_distribution<> dis(1, N);
    std::unordered_multiset<int> elems;
    while (elems.size() < k) {
        elems.insert(dis(gen) % N);
    }
    return elems;
}

std::vector<int> pick(int N, int k){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::unordered_multiset<int> elems = pickSet(N, k, gen);
    std::vector<int> result(elems.begin(), elems.end());
    std::shuffle(result.begin(), result.end(), gen);
    return result;
}

void generateProjector(int i, float *coordX_h, float *coordY_h, Optics optics){
    std::vector<int> xRand = pick(optics.holoW, optics.N);
    std::vector<int> yRand;
    if(optics.projector == "adaptive" || optics.projector == "fullFrame"){
        yRand = pick(optics.holoH, optics.N);
    }
    for(int j = 0; j < optics.N; j++){
        coordX_h[j] = xRand[j] - optics.holoW/2;
        if(optics.projector == "adaptive" || optics.projector == "fullFrame"){
            coordY_h[j] = yRand[j] - optics.holoH/2;
        }
        else{ // line 
            coordY_h[j] = (i % optics.Nlines) * (optics.holoH / optics.Nlines) - optics.holoH / 2;
        }
    }
}

