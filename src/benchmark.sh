#!/bin/bash

cd ../build
cmake -DBENCHMARK=ON .. && make 
echo -e "\n"
echo -e "Benchmarking MirrorPhaseFunctionKernel against cuFFT.\n"
./holoCu 



    

