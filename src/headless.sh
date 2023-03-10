#!/bin/bash

cd ../build
cmake -DBUILD_OPENGL=OFF .. && make 
echo -e "\n"
echo -e "Running headless version. Phase and simulated hologram will be saved to 'holoCu/data/'.\n"
./holoCu --projector="adaptive" --N=75 



    

