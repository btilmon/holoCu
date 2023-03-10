#!/bin/bash

# export DISPLAY=:0.0 && xhost + # enables remote display through ssh, works on jetson nano
cd ../build
cmake -DBUILD_OPENGL=ON .. && make 
echo -e "\n"
echo -e "Running full version. Either phase or simulated hologram will be displayed on your monitor/SLM.\n"
./holoCu --projector="adaptive" --N=75 --display="hologram"



    

