# pragma once 

struct Optics {
    ////////////////
    // SLM settings
    ////////////////
    // slmW and slmH are the full physical slm resolution
    const int slmW = 1920;//3840/2;
    const int slmH = 1080;//2160/2;
    // holoW and holoH are for usually smaller than slm for computational efficiency
    // this reduces light redistribution and quality though 
    const int holoW = slmW / 1;
    const int holoH = slmH / 1;
    int gridsize = holoW * holoH * sizeof(float);

    /////////////////////////////////////////////
    // projector specific settings, see Section 3
    ///////////////////////////////////////////// 
    std::string projector;// = "adaptive";
    std::string display;// = "hologram";
    int N;
    int Nlines = 108;
    int numImages = 10;
    double T = 20; // ms
    double lineT = T / Nlines; // rolling shutter causes each line to be exposed less
    std::vector<std::string> displaySelector = {"phase", "hologram"};
    std::vector<std::string> projectorSelector = {"adaptive", "fullFrame", "line"};

    Optics(cmdline::parser& args){

        // set projector and display
        N = std::min(args.get<int>("N"), holoW * holoH);
        projector = args.get<std::string>("projector");
        display = args.get<std::string>("display");
        assert(std::find(displaySelector.begin(), displaySelector.end(), display) != displaySelector.end() 
            && "--display must be in <phase, hologram>");
        assert(std::find(projectorSelector.begin(), projectorSelector.end(), projector) != projectorSelector.end() 
            && "--projector must be in <adaptive, fullFrame, line>");

        std::cout << "\nSensor Settings: " << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << std::left << std::setw(21) << "Sensor: " << projector << std::endl;
        std::cout << std::left << std::setw(21) << "N: " << N << std::endl;
        std::cout << "Hologram Size (HxW): " << holoH << "x" << holoW << std::endl;
        std::cout  << std::left << std::setw(21) << "SLM Size (HxW): " << slmH << "x" << slmW << "\n\n" << std::endl; 

        // SNR calculations from Section 3
        std::cout << "3D Sensor SNR:" << std::endl;
        std::cout << "--------------" << std::endl;
        std::cout << std::left << std::setw(31) << "Adaptive (Ours): " << "c" << (double)N << std::endl;
        std::cout << std::left << std::setw(31) << "Line Scanning (EpiScan3D): " << "c" << (double)N * sqrt(T/lineT) / (T/lineT)  << std::endl;
        std::cout << "Full Frame (Kinect/RealSense): " << "c \n\n" << std::endl;

        if(projector == "line"){
            numImages = Nlines;
        }
    }
};


