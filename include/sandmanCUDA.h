#include <curand.h>
#include <string>

#ifndef H_SANDMAN
#define H_SANDMAN




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
  float *d_weightVg=NULL;     /// Pointer to GPU memory for vertical
			      ///statistical weight

  //snapshots of the phase space for later processing on beam monitors
  float *d_pointsThetaHsnapshot = NULL;
  float *d_pointsYHsnapshot = NULL;
  float *d_pointsThetaVsnapshot = NULL;
  float *d_pointsYVsnapshot = NULL;

  float yminSnapshot, ymaxSnapshot;
  float thetaMinSnapshot, thetaMaxSnapshot;
  char filenameSnapshot[256];
  
  float sourceDeltaLambda = 1.0f;

  //Temporary storage and variables for random numbers
  float *d_tempArray=NULL;     /// Pointer to GPU memory for (e.g.)
				    /// bender channel number


  float *d_r1g=NULL;          ///< Pointer to GPU memory for random number array buffer 1
  float *d_r2g=NULL;          ///< Pointer to GPU memory for random number array buffer 2
  curandGenerator_t prngGPU;  ///< Random number generator object on GPU
  unsigned int seed = 777;    ///< Default random seed for mersenne twister
  
  float deltaLambdag=0.0;     ///< Wavelength gap between histogram bins
  
  float *d_histogram1D=NULL;  ///< Pointer to GPU memory for 1D histogram buffer (reusable)
  float *d_histogram2D=NULL;  ///< Pointer to GPU memory for 2D histogram buffer (reusable)

  int numElements;            ///< Number of elements in arrays (=number of trajectories)

  
  //Monitor arrays used as temporary storage and flags to be filled by
  //destructor
  float *d_lambdaMonHist=NULL; ///< Pointer to GPU memory for wavelength beam monitor histogram
  std::string lambdaFileName;  ///< File name of wavelength beam monitor histogram
  float lambdaMin;             ///< Minimum wavelength in wavelength histogram
  float lambdaMax;             ///< Maximum wavelength in wavelength histogram
  int lambdaHistSize;          ///< Number of elements in wavelength histogram array (max 100)


  
  float flux;
  float eFlux;
  
  float traj;
  float eTraj;

  Sandman(const bool& verbose=false);
  Sandman(const int nE, const bool&verbose=false);
  ~Sandman();

  void generateBothRandomArrays(void);
  void generateOneRandomArray(void);


  void sample(const float width, const float height, const float win_width, const float win_height, const float hoffset, const float voffset, const float win_dist, const float lambdaMin, const float lambdaMax, const std::string& monitorNameStem=std::string());

  void sandGuideElementCUDA(
			    const float length, 
			    const float entr_width, 
			    const float exit_width, 
			    const float exit_offset_h, 
			    const float mLeft, 
			    const float mRight, 
			    const float entr_height,
			    const float exit_height,
			    const float exit_offset_v,
			    const float mTop,
			    const float mBottom
			    );
  
  void sandSimpleStraightGuide(
			       const float length,
			       const float width,
			       const float height,
			       const float mval);

  void sandTaperedStraightGuide(
			       const float length,
			       const float entranceWidth,
			       const float entranceHeight,
			       const float exitWidth,
			       const float exitHeight,
			       const float mval);
  
  void sandCurvedGuide(
		       const float length,
		       const float sectionLength,
		       const float width,
		       const float height,
		       const float mval,
		       const float radius
		       );

  void sandVerticallyCurvedGuide(
		       const float length,
		       const float sectionLength,
		       const float width,
		       const float height,
		       const float mval,
		       const float radius
		       );
  
  void sandHorizontalBender(
			    const float length,
			    const float width,
			    const float height,
			    const int numChannels,
			    const float waferThickness,
			    const float radius,
			    const float mval
			    );

  void sandVerticalBender(
			    const float length,
			    const float width,
			    const float height,
			    const int numChannels,
			    const float waferThickness,
			    const float radius,
			    const float mval
			    );

  void ellipticOpeningGuide(
			    const float length, 
			    const float exit_width, 
			    const float exit_height, 
			    const float focalPoint1H, 
			    const float focalPoint2H, 
			    const float focalPoint1V, 
			    const float focalPoint2V, 
			    const float mNumber, 
			    const int numSections
			    );

  void ellipticClosingGuide(
			    const float length, 
			    const float entr_width, 
			    const float entr_height, 
			    const float focalPoint1H, 
			    const float focalPoint2H, 
			    const float focalPoint1V, 
			    const float focalPoint2V, 
			    const float mNumber, 
			    const int numSections
			    );

  void parabolicOpeningGuide(
			     const float length, 
			     const float entr_width, 
			     const float entr_height, 
			     const float focalPointH, 
			     const float focalPointV,
			     const float mNumber, 
			     const int numSections
			     );

  
  void parabolicClosingGuide(
			     const float length, 
			     const float entr_width, 
			     const float entr_height, 
			     const float focalPointH, 
			     const float focalPointV,
			     const float mNumber, 
			     const int numSections
			     );


  
  void sandILLHCSModerator(void);

  void sandPSIModerator(void);

  void sandBrillianceTransferModerator(
				       const float width,
				       const float height,
				       const float hoffset,
				       const float voffset);


  void sandModerator(
		     const float width,
		     const float height,
		     const float hoffset,
		     const float voffset,
		     const float temp,
		     const float num);

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
  void sandRotationV(const float angleV);

  void sandTranslationV(const float distanceV);
  void sandTranslationH(const float distanceH);

  void sandRollPhaseSpace(const float theta);

  void sandFreeSpaceCUDA(const float distance, const bool& verbose=true);
  void sandSkewCUDA(const float distance_m);
  void sandReflection(const float mirrorY1, const float mirrorY2, const float mirrorAngle1, const float mirrorAngle2, const float mValue);

  void sandCollimateCUDA(const float divergenceH, const float divergenceV);

  void sandApertureCUDA(const float window_width, const float window_height);
  void sandApertureV(const float window_height);
  void sandApertureH(const float window_width);


  void sandCountTrajectories(void);

  void sandCountNeutrons(void);
  void sandCountNeutronsSquareCorrected(void);
  void sandCountNeutronsCircleCorrected(void);

  void lambdaMonitor(const std::string filename, const float lambdaMin, const float dlambda, int histSize);
  void sandPosMonitorH(const std::string filename, const float min, const float dval, int histSize);

  void phaseSpaceMapHCPU(const char *filename);
  void phaseSpaceMapVCPU(const char *filename);
  void debugPosPosCPU(const char *filename);
  void phaseSpaceMapH(const char *filename, const float ymin, const float ymax, const float thetaMin, const float thetaMax);
  void phaseSpaceMapH(const char *filename);  // auto detects the boundaries for you
  void phaseSpaceMapV(const char *filename, const float ymin, const float ymax, const float thetaMin, const float thetaMax);





  //Unit testing functions
  void unitTestInitPhaseSpace(const float *ypoints, const float *thetapoints, const float *weight);
				     
  void unitTestGetPhaseSpace(float *ypoints, float *thetapoints, float *weight);





 private: 

  
  bool showCUDAsteps;
  void displayWelcome(void);
 

  void allocateArrays(void);
  void generateRandomArray(float *array);
  void zeroHistogram1D(void);
  void zeroHistogram2D(void);

  void executeLambdaMonitor(void);
  void executePhaseSpaceMapH(void);
  void executePhaseSpaceMapV(void);
  
  float arrayMinimum(const float *array, float *result);
  float arrayMaximum(const float *array, float *result);

 void sandReflectionH(const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom);
 void sandReflectionV(const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom);

 
  void sandGetPhaseSpaceH(float *h_pointsY, float *h_pointsTheta, float *h_weight);
  void sandGetPhaseSpaceV(float *h_pointsY, float *h_pointsTheta, float *h_weight);
  void sandDebugPosPos(float *h_pointsH, float *h_weightH, float *h_pointsV, float *h_weightV);

  void sandSqueezeHorizontalBenderChannels(const float width, const float numChannels, const float waferThickness);
  void sandUnSqueezeHorizontalBenderChannels(const float width, const float numChannels, const float waferThickness);

  void sandSqueezeVerticalBenderChannels(const float height, const float numChannels, const float waferThickness);
  void sandUnSqueezeVerticalBenderChannels(const float height, const float numChannels, const float waferThickness);

  
  void report(void);
};

#endif
