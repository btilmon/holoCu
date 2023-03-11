# Energy-Efficient Adaptive 3D Sensing [CVPR 2023]
This is a CUDA implementation/simulator of “Energy Efficient Adaptive 3D Sensing” in CVPR 2023. It enables real time sparse hologram generation on embedded NVIDIA GPU’s like Tegra X1 and most NVIDIA GPU’s like GTX 1660. We provide headless and full versions of the code; the headless version renders simulated hologram information to an image while the full version renders hologram information to a monitor or spatial light modulator using CUDA-OpenGL interop. It is real time since we generate sparse holograms with a fully fused CUDA kernel of our Fresnel Holography approach discussed in our paper. We also use holoCu to simulate various 3D sensors such as [EpiScan3D](http://www.cs.cmu.edu/~ILIM/episcan3d/html/index.html) and [Microsoft Kinect](https://azure.microsoft.com/en-us/products/kinect-dk)/[Intel RealSense](https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html). The core idea of our paper is that our approach achieves better active depth sensing than existing 3D sensors since we achieve higher SNR, and we validated this with holoCu in simulation and on a real Holoeye GAEA 1 Spatial Light Modulator. The simulated 3D sensors below, from left to right, are adaptive (ours), line scanning (EpiScan3D), and full frame (Kinect/Intel RealSense). From top to bottom: estimated phase, simulated hologram, real hologram from a spatial light modulator.



<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <p align="center"><img src="data/marketing/teaser.gif" alt="My GIF" width="100%" height="10%"></p>
  </div>
</div>

## CVPR 2023 Demo Video [(High Resolution Youtube Link)](https://www.youtube.com/watch?v=31lPWl-AU_w&feature=youtu.be)
<p align="center">This is a demo of real-time adaptive active stereo with our holographic projector using the full version of holoCu. The demo runs self contained on a NVIDIA Jetson Nano. See the YouTube link for a higher resolution demo.</p>

<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <p align="center"><img src="data/marketing/output.gif" alt="My GIF" width="80%" height="10%"></p>
  </div>
</div>

<br>



> **Energy-Efficient Adaptive 3D Sensing** <br>
> [Brevin Tilmon](https://btilmon.github.io/)<sup>1,2</sup>, [Zhanghao Sun](https://zhsun0357.github.io/)<sup>3</sup>, [Sanjeev Koppal](https://focus.ece.ufl.edu/people/)<sup>2</sup>, [Yicheng Wu](https://yichengwu.github.io/)<sup>1</sup>, [Georgios Evangelidis](https://sites.google.com/site/georgeevangelidis/)<sup>1</sup>, [Ramzi Zahreddine](https://www.linkedin.com/in/ramzi-zahreddine-42a09b87/)<sup>1</sup>, [Guru Krishnan](https://www.linkedin.com/in/krishnanguru/)<sup>1</sup>, [Sizhuo Ma](https://sizhuoma.netlify.app/)<sup>1</sup>, and [Jian Wang](https://jianwang-cmu.github.io/)<sup>1</sup> <br>
> Snap Research<sup>1</sup>, University of Florida<sup>2</sup>, Stanford University<sup>3</sup><br>
> CVPR 2023<br>
> Paper (Coming Soon)

## Dependencies
There are full and headless versions. The full version renders hologram information to either a monitor or SLM using OpenGL-CUDA interop, and the headless version simply renders hologram information to an image with no hardware required. You only need to install OpenGL for the full version.

* CUDA (Tested on CUDA Toolkit 10.2, CUDA Toolkit includes cuFFT and cuComplex)
* OpenGL (Only needed for full version.)
    * ```sudo apt-get install freeglut3-dev```
    * ```sudo apt-get install libglfw3-dev```


We tested on:
  * Tegra X1 GPU on NVIDIA Jetson Nano, Ubuntu 18.04 provided from Jetson Nano Developer Kit SD Card Image. We overclocked both CPU and GPU using ```sudo jetson-clocks```.
  * GTX 1660, Ubuntu 18.04

## Headless Version
OpenGL is not required for the headless version. The hologram phase and simulated hologram amplitude are saved to an image as ```data/<projector>.png```. After installing CUDA the hologram phase and simulated hologram amplitude can be computed after running:

```bash
mkdir build
cd build
cmake -DBUILD_OPENGL=OFF ..
make
cd ../src/
bash headless.sh
```

It should look like this:

```console
btilmon@linux:~$ bash headless.sh

Running headless version. Phase and simulated hologram will be saved to 'holoCu/data/'.

Sensor Settings: 
----------------
Sensor:              adaptive
N:                   75
Hologram Size (HxW): 1080x1920
SLM Size (HxW):      1080x1920


3D Sensor SNR:
--------------
Adaptive (Ours):               c75
Line Scanning (EpiScan3D):     c7.21688
Full Frame (Kinect/RealSense): c 


Runtimes (ms):
--------------
Phase Computation:    0.375219
Hologram Simulation:  1.02681
Total:                1.40202

```

Here is an example saved file:

<p align="left">
  <img src="data/marketing/adaptive.png" width="320" height="380">
</p>



## Full Version

OpenGL is required to display hologram information for the full version. If using a phase only spatial light modulator instead of a monitor, or just for visualization/debugging on a monitor, set ```--display="phase"``` in ```full.sh``` to display the hologram phase instead of the simulated hologram amplitude. Set ```--display="hologram"``` to display the simulated hologram for visualization on a monitor. After installing CUDA and OpenGL the selected hologram information should render to screen after running:

```bash
mkdir build
cd build
cmake -DBUILD_OPENGL=ON ..
make
cd ../src/
bash full.sh
```

It should look like this if using a line projector and displaying the simulated hologram. This is with GTX 1660:

```console
btilmon@linux:~$ bash full.sh

Running full version. Either phase or simulated hologram will be displayed on your monitor/SLM.

Sensor Settings: 
----------------
Sensor:              line
N:                   75
Hologram Size (HxW): 1080x1920
SLM Size (HxW):      1080x1920


3D Sensor SNR:
--------------
Adaptive (Ours):               c75
Line Scanning (EpiScan3D):     c7.21688
Full Frame (Kinect/RealSense): c 

HDMI-0
Frame buffer size: 1920 x 1080

GPU Device 0: "Turing" with compute capability 7.5

Runtimes (ms):
--------------
Phase Computation:    0.871509
Hologram Simulation:  1.01298
Render to Screen:     0.805597
Total:                2.69008

```

**See our adaptive active stereo demo at the top of the README for an example of what the full version can do.** If just visualizing on a monitor though, here is an example of visualizing the simualted line holograms on a monitor with the full version:

<p align="left">
  <img src="data/marketing/lineDemo.gif" alt="My GIF" width="320" height="180">
</p>




## Projectors

In our paper we compare different 3D sensing techniques including full frame (such as Kinect and Intel RealSense) and line scanning (such as EpiScan3D) to our proposed adaptive approach. An adaptive projector is used by default, set ```--projector="line"``` to use line scanning projector or ```--projector="fullFrame"``` for full frame projector instead in ```<full, headless>.sh```. 

## Benchmarks

Our fused CUDA implementation of Fresnel Holography (pointwise integration of diffractive patterns for each hologram point) is faster than Fourier Holography (taking the FFT of a desired projector image) for sparse holograms (~500 hologram points on GTX 1660, ~75 on Tegra X1), even when using the heavily optimized cuFFT for Fourier Holography. In the figure, we used GTX 1660, and MPF stands for Mirror Phase Function which is our Fresnel Holography approach, and FFT stands for Fourier Holography.

<p align="left">
  <img src="data/marketing/benchmark.png" width="225" height="140">
</p>
