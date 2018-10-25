
#ifndef H_SANDMAN
#define H_SANDMAN

// #Constants - I actually don't like this at all.  TODO - put them into the class

#define MAX_ELEMENTS 10000000
#define DEAD_WEIGHT 0.001
#define THETA_CRIT_NICKEL                               0.099138        ///critical angle for reflection of nickel (m=1) in degrees
#define THETA_CRIT_STANDARD_LAMBDA              1.0                     ///wavelength for which THETA_CRIT_NICKEL applies, in Å
#define NICKEL_REFLECTIVITY                             0.98            ///reflectivity of nickel below m=1
#define CRITICAL_REFLECTIVITY                   0.75            ///reflectivity of supermirror at critical edge

// TODO stick this in as a (static) inline function
#define DEGREES_TO_RADIANS M_PI/180.0
#define RADIANS_TO_DEGREES 180.0/M_PI 

#define SANDMAN_CUDA_THREADS 256




class Sandman {

  public:
  
  
  //Trajectory data arrays
  float *d_modFlux=NULL;      ///Pointer to GPU memory for neutron current
			      ///represented by individual trajectory
  float *d_pointsYH=NULL;     ///Pointer to GPU memory for horizontal
			      ///spatial component of phase space
  float *d_pointsThetaH=NULL; ///Pointer to GPU memory for horizontal theta
			      ///component of phase space
  float *d_pointsYV=NULL;     ///Pointer to GPU memory for vertical spatial
			      ///component of phase space
  float *d_pointsThetaV=NULL; ///Pointer to GPU memory for vertical theta
			      ///component of phase space
  float *d_lambdag=NULL;      ///Pointer to GPU memory for array of
			      ///wavelengths
  float *d_weightHg=NULL;     ///Pointer to GPU memory for horizontal
			      ///statistical weight
  float *d_weightVg=NULL;     ///Pointer to GPU memory for vertical
			      ///statistical weight


  //Temporary storage and variables for random numbers
  float *d_r1g=NULL;          ///Pointer to GPU memory for random number array buffer 1
  float *d_r2g=NULL;          ///Pointer to GPU memory for random number array buffer 2
  curandGenerator_t prngGPU;  ///Random number generator object on GPU
  unsigned int seed = 777;    ///Default random seed for mersenne twister
  
  float deltaLambdag=0.0;     ///Wavelength gap between histogram bins
  
  float *d_histogram1D=NULL;  ///Pointer to GPU memory for 1D histogram buffer (reusable)
  float *d_histogram2D=NULL;  ///Pointer to GPU memory for 2D histogram buffer (reusable)

  int numElements;            ///Number of elements in arrays (=number of trajectories)

  
  //Monitor arrays used as temporary storage and flags to be filled by
  //destructor
  float *d_lambdaMonHist=NULL; /// Pointer to GPU memory for wavelength beam
			       /// monitor histogram
  std::string lambdaFileName;  /// File name of wavelength beam monitor histogram
  float lambdaMin;             /// Minimum wavelength in wavelength histogram
  float lambdaMax;             /// Maximum wavelength in wavelength histogram
  int lambdaHistSize;          /// Number of elements in wavelength histogram array (max 100)


  float flux;
  float eFlux;
  
  float traj;
  float eTraj;
  

  Sandman();
  Sandman(const int nE);
  ~Sandman();


  void generateBothRandomArrays(void);
  void generateOneRandomArray(void);


  /// Sample creates the sample object from which rays are traced.
  void sample(
	      const float width,      ///< Width of sample (m)
	      const float height,     ///< Height of sample (m)
	      const float win_width,  ///< Width of the window through which all trajectories must pass (e.g. guide exit)
	      const float win_height, ///< Height of the window through which all trajectories must pass (e.g. guide exit)
	      const float hoffset,    ///< Horizontal offset of the sample relative to the centre of the window
	      const float voffset,    ///< Vertical offset of the sample relative to the centre of the window 
	      const float win_dist,   ///< Distance between the sample and the window
	      const float lambdaMin,  ///< Minimum wavelength to generate
	      const float lambdaMax   ///< Maximum wavelength to generate
	      );
  
  /// A guide element is a single piece of guide unit that forms the
  /// basis for all reflection geometries.  This function runs purely
  /// on the GPU and is not usually called by the user in the API, but
  /// it is provided in case someone needs to create a fancy guide
  /// shape that is not given by the other functions..
  void sandGuideElementCUDA(
			    const float length,           ///<Length of unit.  Generally, vendors sell 0.5 m or 1 m long pieces.
			    const float entr_width,       ///<Width of the entrance of the section (m)
			    const float exit_width,       ///<Width of the exit of the section (m)
			    const float exit_offset_h,    ///<Offset of the exit centre relative to the entrance centre, projected back along a vector normal to the entrance plane (m). 
			    const float mLeft,            ///<m-value of the left of the guide (relative to nickel)
			    const float mRight,           ///<m-value of the right of the guide (relative to nickel)
			    const float entr_height,      ///<Entrance height of the section (m)
			    const float exit_height,      ///<Exit height of the section (m)
			    const float mTop,             ///<m-value of the top of the guide (relative to nickel)
			    const float mBottom,          ///<m-value of the bottom of the guide (relative to nickel)
			    const float exit_offset_v     ///<Offset of the exit centre relative to the entrance centre, projected back along a vector normal to the entrance plane (m). 
			    );

  /// Loops over sandGuideElementCUDA to implement a simple, straight guide with constant cross section and one m value on all surfaces.
  void sandSimpleStraightGuide(
			       const float length,   ///<Total length of the guide (m)
			       const float width,    ///<Width of the guide (m)
			       const float height,   ///<Height of the guide (m)
			       const float mval      ///<m-value of the guide (relative to nickel)
			       );

  /// Loops over sandGuideElementCUDA to implement a simple, curved guide with constant cross section and one m value on all surfaces.
  void sandCurvedGuide(
		       const float length,          ///<Total length of the curved guide (m)
		       const float sectionLength,   ///<Length of each straight section of guide making up the curve (m) 
		       const float width,           ///<Width of the guide (m)
		       const float height,          ///<Height of the guide (m)
		       const float mval,            ///<m-value of the guide (relative to nickel)
		       const float radius           ///<Radius of curvature of the guide (m)
		       );

  /// Provides a model of the ILL Horizontal Curved Source moderator with no extra code.  References to follow...
  void sandILLHCSModerator(void);


  /// A general moderator with a single characteristic maxwellian temperature
  void sandModerator(
		     const float width,
		     const float height,
		     const float hoffset,
		     const float voffset,
		     const float temp,
		     const float num);

  /// Overloaded definition of moderator to have two 
  void sandModerator(
		     const float width1,
		     const float height1,
		     const float hoffset1,
		     const float voffset1,
		     const float temp1,
		     const float num1,
		     const float width2,
		     const float height2,
		     const float hoffset2,
		     const float voffset2,
		     const float temp2,
		     const float num2);

  void sandModerator(
		     const float width1,
		     const float height1,
		     const float hoffset1,
		     const float voffset1,
		     const float temp1,
		     const float num1,
		     const float width2,
		     const float height2,
		     const float hoffset2,
		     const float voffset2,
		     const float temp2,
		     const float num2,
		     const float width3,
		     const float height3,
		     const float hoffset3,
		     const float voffset3,
		     const float temp3,
		     const float num3);


  void sandRotation(const float angleH, const float angleV);
  void sandRotationH(const float angleH);

  void sandTranslationV(const float distanceV);
  void sandTranslationH(const float distanceH);

  void sandFreeSpaceCUDA(const float distance);
  void sandSkewCUDA(const float distance_m);
  void sandReflection(const float mirrorY1, const float mirrorY2, const float mirrorAngle1, const float mirrorAngle2, const float mValue);

void sandCollimateCUDA(const float divergenceH, const float divergenceV);

void sandApertureCUDA(const float window_width, const float window_height);


  void sandCountTrajectories(float *nSum, float *nSumErr);
  void sandCountNeutrons(float *nSum, float *nSumErr);

  void sandCountNeutronsSquareCorrected(float *nSum, float *nSumErr);


  void lambdaMonitor(const std::string filename, const float lambdaMin, const float dlambda, int histSize);
  void sandPosMonitorH(const std::string filename, const float min, const float dval, int histSize);

  void phaseSpaceMapHCPU(const char *filename);
  void phaseSpaceMapH(const char *filename, const float ymin, const float ymax, const float thetaMin, const float thetaMax);
  void phaseSpaceMapH(const char *filename);  // auto detects the boundaries for you

 private: 
  void displayWelcome(void);
  void report(void);

  void allocateArrays(void);
  void generateRandomArray(float *array);
  void zeroHistogram1D(void);
  void zeroHistogram2D(void);

  void executeLambdaMonitor(void);
  
  float arrayMinimum(const float *array, float *result);
  float arrayMaximum(const float *array, float *result);

 void sandReflectionH(const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom);
 void sandReflectionV(const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom);

 
  void sandGetPhaseSpaceH(float *h_pointsY, float *h_pointsTheta, float *h_weight);
  void sandGetPhaseSpaceV(float *h_pointsY, float *h_pointsTheta, float *h_weight);
  
};

#endif
