///////////////////////////////////////////////////////////////////////////////////
/// 
///   @sandmanCUDA.cu
///   @author Phil Bentley <phil.m.bentley@gmail.com
///   @version 1.0
///   
///   @section LICENSE
///
///   Copyright (c) 2016, Phil Bentley All rights reserved.
///
///   All rights reserved.

///   Redistribution and use in source and binary forms, with or
///   without modification, are permitted provided that the following
///   conditions are met:
///
///   * Redistributions of source code must retain the above copyright
///     notice, this list of conditions and the following disclaimer.
///   * Redistributions in binary form must reproduce the above
///     copyright notice, this list of conditions and the following
///     disclaimer in the documentation and/or other materials
///     provided with the distribution.
///   * Neither the name of the ESS nor the names of its contributors
///     may be used to endorse or promote products derived from this
///     software without specific prior written permission.
///
///   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
///   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
///   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
///   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
///   DISCLAIMED. IN NO EVENT SHALL PHIL BENTLEY BE LIABLE FOR
///   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
///   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
///   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
///   OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
///   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
///   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
///   OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
///   OF SUCH DAMAGE.
///
///   @section DESCRIPTION
///
///   Sandman is a ridiculously fast monte-carlo code for simulating
///   polychromatic neutron beams.
///
///   Sandman uses the math in neutron acceptance diagram shading
///   (nads, which is monochromatic) to implement a monte-carlo method
///   of ray tracing, by breaking up the simulation into two
///   independent planes with finite phase space boundaries.  This is
///   significantly faster than 3D tracing plane intersections, even
///   though it produces mathematically identical output.  The
///   limitation is that it can only simulate beams where the
///   horizontal and vertical phase spaces are independent
///   (e.g. rectangular neutron guides).
///   
///   This code provides to the user a shared library (.so) which you
///   can install on your system.  Thereafter, you create a sandman
///   program that represents the instrument simulation and link this
///   library.  Calling the sandman class public functions creates a
///   simulation of a neutron beam on an NVIDIA GPU using NVIDIA'S
///   CUDA API.
///
///   There are some fundamental differences between this code and
///   existing codes at the time of writing.
///
///   The geometry definition begins at the SAMPLE POSITION, and works
///   backwards.  This is for very good reason.  Start with the phase
///   space you need, and work from there.  It's also orders of
///   magnitude quicker in most cases to work like this.  To handle
///   this reverse tracing method, sandman's beam monitors and
///   calculations have been specially written in a way to provide
///   correct results in the backwards or forwards direction.  For
///   example, the beam monitor functions store a copy of the position
///   of the neutrons, and mirror the divergence; then at the end of
///   the calculation the statistical weight is calculated, so that
///   the beam profile at that position matches the result that VITESS
///   or MCSTAS would give you when simulating forwards.  Nonetheless,
///   there is nothing in the code that prevents you from doing a
///   forwards simulation --- if you insist!  Just define a sample
///   with sandman that is the same size as the moderator, and a
///   sandman moderator that is the same size as the instrument
///   sample, and set the mirror image parameter in the relevant
///   monitor functions to "false".
///
//////////////////////////////////////////////////////////////////////////////////


#include <curand.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <fstream>
#include <iostream>

#include "../include/sandmanCUDA.h"

#define DEBUG 1


// Physical Constants

const static float thetaCritNickel=0.099138f;
#define NICKEL_REFLECTIVITY 0.967f

const static float thetaCritStandardLambda = 1.0f;
const static int maxElements = 10000000;
const static float deadWeight = 0.01f;
const static float PI_FLOAT = 3.1415927f;


// Define colours for terminal output text highlighting

const static std::string color_red("\033[0;31m");
const static std::string color_green("\033[1;32m");
const static std::string color_yellow("\033[1;33m");
const static std::string color_cyan("\033[0;36m");
const static std::string color_magenta("\033[0;35m");
const static std::string color_reset("\033[0m");



std::string remove_extension(const std::string& filename) 
{
  
  size_t lastdot = filename.find_last_of(".");
  
  if (lastdot == std::string::npos) 
    return filename;
  
  return (filename.substr(0, lastdot)); 
}



__host__ __device__
static inline float radians2degrees(const float radians)
{
  return(radians * 180.0f / PI_FLOAT);
}

__host__ __device__
static inline float degrees2radians(const float degrees)
{
  return(degrees * PI_FLOAT / 180.0f);
}


__host__ __device__
static inline float square2circleFlux(const float num)
{
  //Ratio of area of circle to area of square is PI/4
  return ( num / (PI_FLOAT / 4.0f));
}




__host__ __device__
static float elliptic_curve(const float xpos, const float fp1, const float fp2, const float maxWidth) 
{
  //An ellipse where the entrance width is specified

  //find centre of ellipse independent of order of fp1 and fp2
  const float x0 = (fp1 + fp2) / 2.0f;
  
  //find the absolute focal point relative to x0
  float f = fabsf((fp2 - fp1) / 2.0f);

  //b value is half the width
  float b = maxWidth / 2.0f;

  //return the y value of the curve
  float y2 = b*b - (xpos-x0)*(xpos-x0)*b*b/(f*f-b*b);

  return(sqrtf(fabsf(y2)));
}




__host__ __device__
static float parabolic_closing_curve(float xpos, float foc, float inw) 
{
  float x0;
  float ans;
  
  x0 = (2.0f * foc + sqrtf(4.0f * foc * foc + inw * inw)) / 4.0f;
  
  ans = 2.0f * sqrtf((foc - x0) * (xpos - x0));
  
  return (ans);
}


__host__ __device__
static float parabolic_opening_curve(float xpos, float len, float foc, float outw) 
{
  float x0;
  float ans;
  
  x0 = (2.0f*foc + 2.0f*len -
	sqrtf(4.0f*foc*foc -
		  8.0f*foc*len + 4.0f*len*len +
		  outw*outw))/4.0f;
  
  ans = 2.0f*sqrtf((foc - x0)*(-x0 + xpos));
  
  return (ans);
}













__global__
static void global_countNeutrons0(float *numNeutrons, const float *weightH, const float *weightV, const int numElements)
{
  __shared__ float sharedTotal;


  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  //Try doing this one neutron per thread, for fun - and simpler code ;)


  //Boss thread zeros the shared counter
  if(tid == 0)
    {
      sharedTotal=0.0f;
      
    }
  
  __syncthreads();
  
  //Each thread adds the weight of its neutron to the shared total
  if(i<numElements)
    {
      atomicAdd(&sharedTotal, weightH[i]*weightV[i]);
    }
  
  __syncthreads();
  
  //Boss thread adds this total to the global total
  if(i<numElements);
  {
    if(tid == 0)
      {
	atomicAdd(&numNeutrons[0], sharedTotal);
      }
  }
  __syncthreads();
}





// Array reduction routines.  Tried many of these, some are faster than others.

__device__ 
void blockReduce1(float *array)
{
  // Interleaved addressing, reduction #1 from nvidia
  __shared__ float sharedTotal[512];
  
  int tid = threadIdx.x;

  //Work in local shared memory copy
  sharedTotal[tid] = array[tid];
  __syncthreads();

  for(unsigned int s=1; s < SANDMAN_CUDA_THREADS; s*=2)
    {
      if(tid % (2*s) == 0) 
	{
	  sharedTotal[tid] += sharedTotal[tid +s];
	}
      __syncthreads();
    }
  

  //Write back to block master thread
  if(tid == 0)
    {
      array[0] = sharedTotal[0];
    }
}




__device__ 
void blockReduce2(float *array)
{
  // Interleaved addressing, reduction #2 from nvidia
  __shared__ float sharedTotal[512];
  
  int tid = threadIdx.x;
  int index;

  //Work in local shared memory copy
  sharedTotal[tid] = array[tid];
  __syncthreads();

  for(unsigned int s=1; s < SANDMAN_CUDA_THREADS; s*=2)
    {
      index = 2 * s*tid;

      if(index < SANDMAN_CUDA_THREADS) 
	{
	  sharedTotal[index] += sharedTotal[index +s];
	}
      __syncthreads();
    }
  

  //Write back to block master thread
  if(tid == 0)
    {
      array[0] = sharedTotal[0];
    }
}



__device__ 
void blockReduce3(float *array)
{
  // Sequential addressing, reduction #3 from nvidia
  __shared__ float sharedTotal[512];
  
  int tid = threadIdx.x;



  //Work in local shared memory copy
  sharedTotal[tid] = array[tid];
  __syncthreads();

  for(unsigned int s=SANDMAN_CUDA_THREADS/2; s > 0; s>>=1)
    {
      if(tid < s)
	{
	  sharedTotal[tid] += sharedTotal[tid +s];
	}
      __syncthreads();
    }
  

  //Write back to block master thread
  if(tid == 0)
    {
      array[0] = sharedTotal[0];
    }
}




__device__ 
void blockReduce4_DO_NOT_USE(float *array)
{
  //DOES NOT WORK!  There is a bug somewhere...

  // Sequential addressing plus uroll loops
  __shared__ float sharedTotal[512];
  
  int tid = threadIdx.x;

  //Work in local shared memory copy
  sharedTotal[tid] = array[tid];
  __syncthreads();

  for(unsigned int s=SANDMAN_CUDA_THREADS/2; s > 32; s>>=1)
    {
      if(tid < s)
	{
	  sharedTotal[tid] += sharedTotal[tid +s];
	}
      __syncthreads();
    }

  if(tid <= 32)
    {
      sharedTotal[tid] += sharedTotal[tid+32];
      sharedTotal[tid] += sharedTotal[tid+16];
      sharedTotal[tid] += sharedTotal[tid+8];
      sharedTotal[tid] += sharedTotal[tid+4];
      sharedTotal[tid] += sharedTotal[tid+2];
      sharedTotal[tid] += sharedTotal[tid+1];
    }
  

  //Write back to block master thread
  if(tid == 0)
    {
      array[0] = sharedTotal[0];
    }

  __syncthreads();
}













__global__
static void global_countNeutrons(float *numNeutrons, const float *weightH, const float *weightV, const float deltaLambda, const float *modFlux, const int numTraj)
{
  //Shared memory per thread block We can just use 512 knowing that the number
  //of threads will be 128 or 256 or something
  __shared__ float sharedTotal[512];


  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  const float nTraj = (float) numTraj;
  float element;
  
  //Try doing this one neutron per thread, for fun - and simpler code ;)


  //All threads zero their shared counter

  //sharedTotal[i]=0.0f;      
  //__syncthreads();  //Probably not needed until the last moment before block reduction
  


  //Each thread adds the weight of its neutron to the shared total
  if(i<numTraj)
    {

      //sharedTotal[tid] = modFlux[i]*weightH[i]*weightV[i]/nElements;      
      element = weightH[i] * weightV[i]; //in units of fractions of neutrons
      
      if(modFlux != NULL)
      	{
	  //Don't just calculate efficiency, user has specified a moderator.
	  //In this case, each fractional neutron should be scaled by the
	  //moderator brightness sampled by that trajectory, and then
	  //normalised to the full simulation wavelength band and number of
	  //trajectories
      	  element = element * deltaLambda * modFlux[i]  / nTraj;
      	}

      sharedTotal[tid] = element;

      __syncthreads();
      
      
      // do block reduction on the shared memory using NVIDIA's tree method
      blockReduce1(sharedTotal);
      
      //Boss thread sums shared total and adds to the global total
      if(tid == 0)
	{
	  atomicAdd(&numNeutrons[0], sharedTotal[0]);
	}
      __syncthreads();
    }
}



__global__
static void global_countTrajectories(float *numNeutrons, const float *weightH, const float *weightV, const int numElements)
{
  //Shared memory per thread block
  __shared__ float sharedTotal[512];


  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  //Try doing this one neutron per thread, for fun - and simpler code ;)


  //All threads zero their shared counter

  //sharedTotal[i]=0.0f;      
  //__syncthreads();  //Probably not needed until the last moment before block reduction
  
  //Each thread adds the weight of its neutron to the shared total
  if(i<numElements)
    {
      sharedTotal[tid] = weightH[i]*weightV[i];
      __syncthreads();
      
      
      // do block reduction on the shared memory using NVIDIA's tree method
      blockReduce3(sharedTotal);
      
      //Boss thread sums shared total and adds to the global total
      if(tid == 0)
	{
	  atomicAdd(&numNeutrons[0], sharedTotal[0]);
	}
      __syncthreads();
    }
}




///Compresses the beam into a single channel of the bender, to model a multi-channel bender by a single channel process (n channels would have n branches otherwise).
  
__global__
static void global_squeezeBenderChannel(float *ypos, float *channelNumber, const float width, const float channelWidth, const float waferThickness, const int numElements)
{
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float relativeY;

  if(i<numElements)
    {
      relativeY = ypos[i] + width/2.0f;
      //The 'channelWidth' is defined as the empty space plus one
      //wafer thickness
      //Channel number starts at zero
      channelNumber[i] = floorf( relativeY / channelWidth );
      
      //Then we adjust the position to be within a single channel of
      //the right thickness for the OPTICS
      ypos[i] = relativeY;
      ypos[i] = ypos[i] / (channelNumber[i]+1.0f);
      ypos[i] = ypos[i] - 0.5f * (channelWidth-waferThickness);
    }
}


///Reverses the compression of the beam into a single channel of the
///bender, to model a multi-channel bender by a single channel process
///(n channels would have n branches otherwise).
__global__
static void global_unSqueezeBenderChannel(float *ypos, const float *channelNumber, const float width, const float channelWidth, const float waferThickness, const int numElements)
{
  // Calculates the total emitted neutron current represented by this
  // trajectory, based on its interception with one moderator surace
  // characterised by a single temperature temp, width width, positioned with
  // an offset hoffset, and a brightness num
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {

      //Device reverses the position adjustment of sandSqueeze...
      ypos[i] = ypos[i] + 0.5f * (channelWidth - waferThickness);
      ypos[i] = ypos[i] * (channelNumber[i]+1.0f);
      ypos[i] = ypos[i] - width/2.0f;
    }
}





__global__
static void global_copyArray(const float *source, float *destination, const int numElements, const bool invert)
{
  // Copies the values from one array to another
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  //These nested conditionals do not cause branching problems because all
  //threads evaluate the same path
  if(i<numElements)
    {
      if(invert)
	destination[i] = -source[i];
      else
	destination[i] = source[i];
    }
}









__host__ __device__
static float maxwellian(const float brightness0, const float tempK, const float lambda_A)
{

  //Describes a maxwellian curve based on the given parameters.  This
  //maxwellian is the same curve as used by existing codes, so it
  //should agree with those (nads, mcstas)
  
  //Defined __host__ __device__ so it can be unit tested if required

  const float h = 6.626076E-34;
  const float m = 1.6749284E-27;
  const float k = 1.380662E-23;

  const float a=(1.0E10*1.0E10*h*h)/(2.0*k*m*tempK);
  
  return(
	 brightness0*2.0*a*a*exp(-a/(lambda_A*lambda_A))/pow(lambda_A,5.0)
	 );
}



__host__ __device__
static float illHCS(const float lambda_A)
{

  //Defined __host__ __device__ so it can be unit tested if required

  return(
	 maxwellian(2.78E13f, 40.1f, lambda_A)
	 +
	 maxwellian(3.44E13, 145.8f, lambda_A)
	 +
	 maxwellian(1.022E13, 413.5f, lambda_A)
	 );
}



  

  
__global__
static void global_sandILLHCSModerator(float *d_modFluxH, float *d_weightH, const float *d_lambdag, const float *d_pointsYH, const int numElements)
{
  // Calculates the total emitted neutron current represented by this
  // trajectory, based on its interception with one moderator surace
  // characterised by a single temperature temp, width width, positioned with
  // an offset hoffset, and a brightness num
  
  float ymax, ymin;
  
  ymax = 0.206f/2.0f;
  ymin = -ymax;

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < numElements)
    {
      //The sample module assigns the scaling factor related to solid angle, now we do moderator brightness
      if(d_modFluxH[i] < 10.0f)
	{
	  //That check means we did not already calculate the flux, so we need to do it now:
	  d_modFluxH[i] = d_modFluxH[i] * illHCS(d_lambdag[i]);
	}
      //Modify the weight if the neutron misses For one moderator, it is an easy
      //window For multiple moderators, we need to set the weight to the initial
      //value, then add multiples of that to an initially zeroed accumulator
      if(d_pointsYH[i] > ymax || d_pointsYH[i] < ymin)
	{
	  d_weightH[i] = 0.0;
	}
    }
}



  
__global__
static void global_sandModerator1(float *d_modFluxH, float *d_weightH, const float *d_lambdag, const float *d_pointsYH, const int numElements, const float width, const float hoffset, const float temp, const float num)
{
  // Calculates the total emitted neutron current represented by this
  // trajectory, based on its interception with one moderator surace
  // characterised by a single temperature temp, width width, positioned with
  // an offset hoffset, and a brightness num
  
  float ymax, ymin;
  
  ymax = hoffset + width/2.0;
  ymin = hoffset - width/2.0;

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  //The sample module assigns the scaling factor related to solid angle, now we do moderator brightness
  if(d_modFluxH[i] < 10.0f)
    {
      //That check means we did not already calculate the flux, so we need to do it now:
      d_modFluxH[i] = d_modFluxH[i] * maxwellian(num, temp, d_lambdag[i]);
    }

  //Modify the weight if the neutron misses For one moderator, it is an easy
  //window For multiple moderators, we need to set the weight to the initial
  //value, then add multiples of that to an initially zeroed accumulator
  if(d_pointsYH[i] > ymax || d_pointsYH[i] < ymin)
    {
      d_weightH[i] = 0.0;
    }

}



__global__
static void global_sandSampleCUDA(float *d_pointsY, float *d_pointsTheta, float *d_weight, const float *d_r1, const float *d_r2, const float ox, const float oy, const float v1x, const float v2x, const float v2y, const int numElements)
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {
      d_pointsY[i] = oy + d_r2[i]*v2y;
      d_pointsTheta[i] = ox + d_r1[i]*v1x + d_r2[i]*v2x;
      d_weight[i] = 1.0;
    }
}




__global__
static void global_initArray(float *d_array, const float value, const int numElements)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
      
  if(i<numElements)
    {
      d_array[i] = value;
    }
}
 


     

__global__
static void global_sandZeroHistogram1D(float d_histogram[100])
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<100)
    {
	{
	  d_histogram[i] = 0.0f;
	}
    }
}



__global__
static void global_sandZeroHistogram2D(float d_histogram[100][100])
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j;

  if(i<100)
    {
      for(j=0; j<100; j++)
	{
	  d_histogram[i][j] = 0.0f;
	}
    }
}





__global__
static void global_sandSkewCUDA(float *d_pointsY, const float *d_pointsTheta, const float distance_m, const int numElements)
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {
      // Ignore dead neutrons
      //if(d_weight[i] > DEAD_WEIGHT)
	{
	  d_pointsY[i] = d_pointsY[i] + distance_m * d_pointsTheta[i];
	}
    }
}

__global__
static void global_rotation(float *d_pointsTheta, const float angle_radians, const int numElements)
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {
      // Ignore dead neutrons
      //if(d_weight[i] > DEAD_WEIGHT)
	{
	  d_pointsTheta[i] = d_pointsTheta[i] - angle_radians;
	}
    }
}


__global__
static void global_translation(float *d_pointsY, const float distance_m, const int numElements)
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {
      // Ignore dead neutrons
      //if(d_weight[i] > DEAD_WEIGHT)
	{
	  d_pointsY[i] = d_pointsY[i] - distance_m;
	}
    }
}


__device__ 
inline static float low_pass_filter(const float value, const float cutOff)
{
  // This function uses approximation to heaviside function with approximate
  // tanh running on hardware to avoid a branching if statement.  Important
  // for thread divergence.
  return( 0.5f + 0.5f*tanh(2000.0f*(-value+cutOff)));
}

__device__ 
inline static float high_pass_filter(const float value, const float cutOff)
{
  // High pass filter.  values greater than cutOff have > 0 return value
  // This function uses approximation to heaviside function with approximate
  // tanh running on hardware to avoid a branching if statement.  Important
  // for thread divergence.
  return( 0.5f + 0.5f*tanh(2000.0f*(value-cutOff)));
}




__global__
static void global_collimation(float *d_weight, const float *d_pointsTheta, const float lower_angle, const float upper_angle, const int numElements)
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {
      // Ignore dead neutrons
      //if(d_weight[i] > DEAD_WEIGHT)
	{
	  //Filter off lower points
	  d_weight[i] = d_weight[i] * high_pass_filter(d_pointsTheta[i], lower_angle);
	  
	  //Filter off higher points
	  d_weight[i] = d_weight[i] * low_pass_filter(d_pointsTheta[i], upper_angle);
	}
    }
}




__global__
static void global_aperture(float *d_weight, const float *d_pointsY, const float lower_position, const float upper_position, const int numElements)
{ 
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<numElements)
    {
      // Ignore dead neutrons
      //if(d_weight[i] > DEAD_WEIGHT)
	{
	  //Filter off lower points
	  d_weight[i] = d_weight[i] * high_pass_filter(d_pointsY[i], lower_position);
	  
	  //Filter off higher points
	  d_weight[i] = d_weight[i] * low_pass_filter(d_pointsY[i], upper_position);
	}
    }
}







__host__ __device__
inline float device_criticalReflectivity(float mValue)
{
	//Data taken from swiss neutronics.  approximates the correct m value using a quadratic fit to their data
	return(-0.01288f*mValue*mValue+0.98f);
}


__host__ __device__
inline float device_critical_theta(const float wavln, /**< Wavelength of incident neutrons. */
					  const float mValue)/**< m value of the surface. */
{
	float ans;
	ans = wavln * mValue / thetaCritStandardLambda;
	ans = degrees2radians(ans);
	ans = ans * thetaCritNickel;
	return( ans);
}


__host__ __device__
float device_reflectivity_slow(const float theta_rads,	/**< Angle of incidence in radians. */
		    const float lambda, const float mValue)		/**< m value of reflecting surface. */
{
	

	//m=1 critical angle
	const float thetaCritM1 = device_critical_theta(lambda, 1.0f);
	
	//general critical angle
	const float thetaCrit = device_critical_theta(lambda, mValue);
	
	const float dist = fabsf(theta_rads);
	
	float attn0;
	float attnGrad;
	float ans;
	
	if(dist <= thetaCritM1)
	{
	  //Flat at low angles below m=1
	  ans = device_criticalReflectivity(1.0);
       	}	
	else if(dist <= thetaCrit)
	{
	  //linear decay to the knee value above m=1
	  attnGrad = (device_criticalReflectivity(mValue) - device_criticalReflectivity(1.0)) / (thetaCrit - thetaCritM1);
	  attn0    = device_criticalReflectivity(1.0) - attnGrad*thetaCritM1;
	  	
	  ans = attn0 + attnGrad * dist;
	}
	else
	  {
	    ans = 0.0f;
	  }
	
	return(ans);
}


__device__
float device_reflectivity(const float theta_rads,	/**< Angle of incidence in radians. */
		    const float lambda, const float mValue)		/**< m value of reflecting surface. */
{
	

	//m=1 critical angle
	const float thetaCritM1 = device_critical_theta(lambda, 1.0f);
	
	//general critical angle
	const float thetaCrit = device_critical_theta(lambda, mValue);
	
	const float dist = fabsf(theta_rads);
	
	float attn0;
	float attnGrad;
	float ans=NICKEL_REFLECTIVITY;

	if(dist > thetaCritM1)
	  {
	    attnGrad=(device_criticalReflectivity(mValue) - NICKEL_REFLECTIVITY) / (thetaCrit - thetaCritM1);
	    attn0    = NICKEL_REFLECTIVITY - attnGrad*thetaCritM1;
	    ans = attn0 + attnGrad * dist;
	  }

	//Multiply by low pass above thetaCrit
	if(dist > thetaCrit)
	  {
	    ans = 0.0f;
	  }
	
	return(ans);
}



__device__
static float device_attenuate_alpha(const float valpha, const float lambda, const float theta, const float mValue)
{
//Attenuates the opacity of a vertex based on its divergence angle

  return (valpha * device_reflectivity(theta, lambda, mValue));
}



__global__
static void global_sandAllocateWavelength(float *d_lambdaH, const float *d_r1g, const float lambda1, const float deltaLambda, const int numElements)
{
  
  
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i<numElements)
      {
	d_lambdaH[i] = lambda1 + d_r1g[i]*deltaLambda;  //FMA this later

      }
}




__global__
static void global_lambdaMonitor(float *lambdaHist, const float lambdaMin, const float dLambda, int histSize, const float *lambda, const float *weightH, const float *weightV, const float *d_modflux, const float sourceDeltaLambda, const int numElements)
{
  __shared__ float sharedLambdaHist[100];

  int targetBin;
  int j;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  float element;


  //Try doing this one neutron per thread, for fun - and simpler code ;)

  // THIS IS SLOW, we need a faster way

  //Boss thread zeros the shared counter
  if(tid == 0)
    {
      for(j=0; j<100 && j<histSize; j++)
	{
	  sharedLambdaHist[j] = 0.0f;
	}
    }
  
  __syncthreads();
  
  //Each thread adds the weight of its neutron to the shared total
  if(i<numElements)
    {
      //Add horizontal bit
      //targetBin = (int) roundf(-0.5f + (lambda[i] - lambdaMin)/dLambda  );
      targetBin = (int) rintf(-0.5f + (lambda[i] - lambdaMin)/dLambda  );
      
      //This function agrees with VITESS "normalise with binsize = no"
      //be certain to send non-zero dLambda to this function!
      element = weightH[i] * weightV[i] / dLambda; // in units of fractions of neutrons per angstrom

      //Normalise to wavelength range
      element = element * sourceDeltaLambda; //in units of fractions of neutrons 

      if(d_modflux != NULL)
	{
	  element = element * d_modflux[i] / (float)numElements;
	}


      if( (targetBin > 0) && (targetBin < 100) && (targetBin < histSize) )
	{
	  atomicAdd(&sharedLambdaHist[targetBin], element);
	}
      // //Add vertical bit
      // targetBin = roundf( (lambdaV[i] - lambdaMin)/dLambda );
      // atomicAdd(&sharedLambdaHist[targetBin], weightV[i]);
    }
  
  __syncthreads();
  
  //Boss thread adds this total to the global total
  if(i<numElements);
  {
    if(tid == 0)
      {
	for(j=0; j<100 && j<histSize; j++)
	  atomicAdd(&lambdaHist[j], sharedLambdaHist[j]);
      }
  }
  __syncthreads();
}





__global__
static void global_arrayMinimum(const float *array, float globalMin[1], const int numElements)
{
  __shared__ float sharedMin;

  // This function DOES NOT WORK YET!  There is a race condition 

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  
  //Boss thread in first block initialises the global memory
  if(tid == 0 && bid == 0)
    {
      globalMin[0] = array[i];
    }

  __syncthreads();

  //Boss thread in warp initialises the shared memory
  if(tid == 0)
    {
      sharedMin = array[i];
    }
  
  __syncthreads();
  
  //Each thread checks it's value against the shared minimum, and overwrites it if necessary
  if(i < numElements)
    {
      //This has to be handled correctly, otherwise there is a race condition
      //at this point - the if statement is not synchronised and it overwrites
      if(array[i] < sharedMin)
	atomicExch(&sharedMin, array[i]);      
    }
  
  __syncthreads();
  
  //Boss thread overwrites global total if necessary
  if(i < numElements);
  {
    if(tid == 0)
      {
	if(sharedMin < globalMin[0]);
	  atomicExch(&globalMin[0], sharedMin);
      }
  }
  __syncthreads();
}




__global__
static void global_arrayMaximum(const float *array, float globalMax[1], const int numElements)
{
  __shared__ float sharedMax;

  
  // This function DOES NOT WORK YET!  There is a race condition 

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  
  //Boss thread in first block initialises the global memory
  if(tid == 0 && bid == 0)
    {
      globalMax[0] = array[i];
    }

  __syncthreads();
  

  //Boss thread initialises the shared counter
  if(tid == 0)
    {
      sharedMax = array[i];
    }
  
  __syncthreads();
  

  if(i < numElements)
    {
        //This has to be handled correctly, otherwise there is a race condition
      //at this point - the if statement is not synchronised and it overwrites
      
      if(array[i] > sharedMax)
	atomicExch(&sharedMax, array[i]);      
    }
  
  __syncthreads();
  
  //Boss thread in warp overwrites global total if necessary
  if(i < numElements);
  {
    if(tid == 0)
      {
	if(sharedMax > globalMax[0]);
	  atomicExch(&globalMax[0], sharedMax);
      }
  }
  __syncthreads();
}







__global__
static void global_Monitor1D(float *globalHist, const float min, const float dval, int histSize, const float *array, const float *weight, const int numElements)
{
  __shared__ float sharedHist[100];

  int targetBin;
  int j;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  //Try doing this one neutron per thread, for fun - and simpler code ;)

  // THIS IS SLOW, we need a faster, slightly more complex way

  //Boss thread zeros the shared counter
  if(tid == 0)
    {
      for(j=0; j<100 && j<histSize; j++)
	{
	  sharedHist[j] = 0.0f;
	}
    }
  
  __syncthreads();
  
  //Each thread adds the weight of its neutron to the shared total
  if(i<numElements)
    {
      //Add horizontal bit
      //targetBin = roundf( (array[i] - min)/dval );
      targetBin = rintf( (array[i] - min)/dval );
      atomicAdd(&sharedHist[targetBin], weight[i]);      
    }
  
  __syncthreads();
  
  //Boss thread adds this total to the global total
  if(i<numElements);
  {
    if(tid == 0)
      {
	for(j=0; j<100 && j<histSize; j++)
	  atomicAdd(&globalHist[j], sharedHist[j]);
      }
  }
  __syncthreads();
}







__global__
static void global_rebinnedPhaseSpaceH(float globalHist[100][100], const float *d_pointsY, const float *d_pointsTheta, const float yMin, const float dy, const float thetaMin, const float dtheta, int histSize, const float *d_weight, const int numElements)
{

  int targetBinY, targetBinTheta;
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < numElements)
    {
      //targetBinY = roundf( (d_pointsY[i] - yMin)/dy  );
      //targetBinTheta = roundf( (d_pointsTheta[i] - thetaMin)/dtheta  );

      targetBinY = rintf( (d_pointsY[i] - yMin)/dy  );
      targetBinTheta = rintf( (d_pointsTheta[i] - thetaMin)/dtheta  );


      if(targetBinY >= 0 && targetBinY < 100 && targetBinY < histSize)
	{
	  if(targetBinTheta >= 0 && targetBinTheta < 100 && targetBinTheta < histSize)
	    {
	      atomicAdd(&globalHist[targetBinTheta][targetBinY], d_weight[i]); 
	    }
	}
    }
}










__global__
static void global_sandReflection(float *d_pointsY, float *d_pointsTheta, const float *d_lambda, float
			   *d_weight, const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom, const int numElements)
{
  

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  bool finished=false;
 
  float mval, mirrorAngle, mirrorY;

  // The next bit of code loops over all particles until they are no longer
  // reflected in the mirror(s).  The way it is written at the moment is that
  // it keeps looping over the same particle until it is finished.  An
  // alternative way might be that each thread handles a single reflection
  // case, and a shared bool keeps all threads going until no particles are
  // reflected.  It might be the same speed, but I think this way is faster,
  // particularly with CUDA.

    /* This bit either goes to openMP or CUDA */

  if(i<numElements)
    {
      // Ignore dead neutrons
      if(d_weight[i] > deadWeight)
      	{
	  do
	    {
	      finished=true;
	      
	      
	      /* Reflect in the upper plane? */
	      if(d_pointsY[i] > mirrorYtop)
		{
		  mval = mTop;
		  mirrorAngle = mirrorAngleTop;
		  mirrorY = mirrorYtop;
		  finished = false;
		}
	      
	      /* Are we in the lower plane? */
	      if(d_pointsY[i] < mirrorYbottom)
		{
		  mval = mBottom;
		  mirrorAngle = mirrorAngleBottom;
		  mirrorY = mirrorYbottom;
		  finished = false;
		}

	      /* Do we need to do slow work? */
	      if(finished == false)
		{
		  d_weight[i] = device_attenuate_alpha(d_weight[i], d_lambda[i], fabsf(d_pointsTheta[i] - mirrorAngle), mval);
		  d_pointsTheta[i] = 2.0*mirrorAngle - d_pointsTheta[i];
		  
		  /* reflection in Y */
		  /* pointsY[i] = mirrorY - (pointsY[i] - mirrorY); */
		  d_pointsY[i] = 2.0*mirrorY - d_pointsY[i];
		}
	    }
	  while (finished == false);
	  }  
    }
}













void Sandman::allocateArrays(void)
{
  ///
  /// Private function to allocate arrays on the GPU for the instance of the
  /// sandman class.  Must be called by constructors.
  ///

  std::cout << "\tAllocating arrays" << std::endl;

  //Initialise random number generator
  std::cout << "\t\tCreating random number generator on GPU" << std::endl;
  checkCudaErrors(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));
  

  std::cout << "\t\tAllocating array pointers on device" << std::endl;
  //Allocate device memory for random numbers
  checkCudaErrors(cudaMalloc((void **)&d_r1g, numElements * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_r2g, numElements * sizeof(float)));

  //Allocate device memory for horizontal phase space
  checkCudaErrors(cudaMalloc((void **)&d_pointsYH, numElements * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_pointsThetaH, numElements * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_weightHg, numElements * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_lambdag, numElements * sizeof(float)));

  //Allocate device memory for vertical phase space
  checkCudaErrors(cudaMalloc((void **)&d_pointsYV, numElements * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_pointsThetaV, numElements * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_weightVg, numElements * sizeof(float)));

  //Allocate device memory for temporary array (bender channel number, other funcs)  
  checkCudaErrors(cudaMalloc((void **)&d_tempArray, numElements * sizeof(float)));

  //Allocate arrays for histograms
  checkCudaErrors(cudaMalloc((void **)&d_histogram2D, 100*100* sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_histogram1D, 100* sizeof(float)));

  //Moderator brightness curve
   if(d_modFlux == NULL)
     checkCudaErrors(cudaMalloc((void **)&d_modFlux, numElements * sizeof(float)));
   
   if(d_modFlux == NULL)
     {
       std::cerr << color_red << "ERROR:" << color_reset << " failure to allocate memory for moderator brightness curve" << std::endl;
       exit(1);
     }
}





Sandman::Sandman(const bool& verbose)
{
  
  ///
  /// Constructor, which will generate 100 trajectories and use the standard
  /// random seed of 777.
  /// 


  numElements = 100;
  int nDevices;

  flux = -1.0;
  eFlux = -1.0;
  traj = -1.0;
  eTraj = -1.0;

  showCUDAsteps=false;

  if(verbose)
    showCUDAsteps=true;

  displayWelcome();

  
  std::cout << color_yellow << "INITIALISING" << color_reset << std::endl;


  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) 
    {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("\tDevice Number: %d\n", i);
      printf("\t\tDevice name: %s\n", prop.name);
      printf("\t\tMemory Clock Rate (KHz): %d\n",
	     prop.memoryClockRate);
      printf("\t\tMemory Bus Width (bits): %d\n",
	     prop.memoryBusWidth);
      printf("\t\tPeak Memory Bandwidth (GB/s): %f\n\n",
	     2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

  std::cout << "\tAllocating arrays" << std::endl;
  allocateArrays();

}


Sandman::Sandman(const int nE, const bool& verbose)
{

  ///
  /// Constructor, which will generate 100 trajectories and use the standard
  /// random seed of 777.  
  /// @param nE an integer parameter to define how many
  /// trajectories should be generated.  
  /// \todo Check that the number of
  /// trajectories does not exceed available GPU memory
  /// 

  numElements = nE;
  int nDevices;

  flux = -1.0;
  eFlux = -1.0;
  traj = -1.0;
  eTraj = -1.0;

  showCUDAsteps=false;

  if(verbose)
    showCUDAsteps=true;

  displayWelcome();

  std::cout << color_yellow << "INITIALISING" << color_reset << std::endl;


  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) 
    {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("\tDevice Number: %d\n", i);
      printf("\t\tDevice name: %s\n", prop.name);
      printf("\t\tMemory Clock Rate (KHz): %d\n",
	     prop.memoryClockRate);
      printf("\t\tMemory Bus Width (bits): %d\n",
	     prop.memoryBusWidth);
      printf("\t\tPeak Memory Bandwidth (GB/s): %f\n\n",
	     2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

  std::cout << "\tAllocating arrays" << std::endl;

  allocateArrays();
}





Sandman::~Sandman(void)
{
  ///
  /// Destructor
  /// First launches histogram code, then cleans up memory.
  /// 

  std::cout << color_yellow << "CLEANING UP" << color_reset << std::endl;

   std::cout << "\tShutting down sandman." << std::endl;




  if(d_lambdaMonHist != NULL)
    {
      executeLambdaMonitor();
      checkCudaErrors(cudaFree(d_lambdaMonHist));
    }

  if(d_pointsThetaHsnapshot != NULL && d_pointsYHsnapshot != NULL)
    {
      executePhaseSpaceMapH();

      checkCudaErrors(cudaFree(d_pointsThetaHsnapshot));
      checkCudaErrors(cudaFree(d_pointsYHsnapshot));
    }


   std::cout << "\tFreeing up device memory" << std::endl;

   if(d_r1g != NULL)
     checkCudaErrors(cudaFree(d_r1g));

   if(d_r2g != NULL)
     checkCudaErrors(cudaFree(d_r2g));

   if(d_pointsYH != NULL)
     checkCudaErrors(cudaFree(d_pointsYH));

   if(d_pointsThetaH != NULL)
     checkCudaErrors(cudaFree(d_pointsThetaH));

   if(d_pointsYV != NULL)
     checkCudaErrors(cudaFree(d_pointsYV));

   if(d_pointsThetaV != NULL)
     checkCudaErrors(cudaFree(d_pointsThetaV));
   
   if(d_lambdag != NULL)
     checkCudaErrors(cudaFree(d_lambdag));

   if(d_weightHg != NULL)
     checkCudaErrors(cudaFree(d_weightHg));

   if(d_weightVg != NULL)
     checkCudaErrors(cudaFree(d_weightVg));

   if(d_tempArray != NULL)
     checkCudaErrors(cudaFree(d_tempArray));

   if(d_histogram1D != NULL)
     checkCudaErrors(cudaFree(d_histogram1D));

   if(d_histogram2D != NULL)
     checkCudaErrors(cudaFree(d_histogram2D));

   if(d_modFlux != NULL)
     checkCudaErrors(cudaFree(d_modFlux));
   

   std::cout << "\tShutting down random generator" << std::endl;

   checkCudaErrors(curandDestroyGenerator(prngGPU));

   report();

 }



void Sandman::report(void)
{
  ///
  /// Generates report of results
  /// 
  
  std::cout << color_yellow << "FINAL REPORT" << color_reset << std::endl;

  std::cout << "\tNeutron counter:" << std::endl;
  std::cout << "\t\tGot " << color_green << flux << color_reset << " neutrons per second total current" << std::endl;
  
  if(traj > 0.0f)
    {
      std::cout << "\tTrajectory counter:" << std::endl;
      std::cout << "\t\tGot " << traj << " trajectories" << std::endl;
    }
  
}



void Sandman::generateBothRandomArrays(void)
{
  ///
  /// Generates random numbers on both array buffer.  Use case: subsequent
  /// random generation of theta and y values in phase space map.
  /// 
  generateRandomArray(d_r1g);
  generateRandomArray(d_r2g);
}


void Sandman::generateOneRandomArray(void)
{
  ///
  /// Generates random numbers on only the first array buffers.  Use case:
  /// subsequent generation of wavelength values.
  ///   
  generateRandomArray(d_r1g);
}




void Sandman::sandCountNeutrons(void)
{
  ///
  /// Integrates over all trajectories to estimate the total neutron current.
  /// 
  /// @param nSum pointer to single host memory float to store answer 
  /// @param nSumErr pointer to single host memory float for statistical 
  /// error on total
  ///
  /// \todo Either provide or remove nSum nSumErr functionality
  ///   

  float *d_nSum;

  float h_nSum[1];  //count, error that way we have one memory transfer for everything


  std::cout << color_yellow << "NEUTRON COUNTER" << color_reset << std::endl;
  
  checkCudaErrors(cudaMalloc((void **)&d_nSum, sizeof(float)));

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

  if(showCUDAsteps)
    {
      printf("CUDA kernel count neutrons with %d blocks of %d threads\n", blocksPerGrid,
	 threadsPerBlock);
    }

  // Zero the count on the host
  h_nSum[0] = 0.0f;

  // Copy the zero total to device memory
  checkCudaErrors(cudaMemcpy(d_nSum, h_nSum, sizeof(float), cudaMemcpyHostToDevice));
  
 
  // static void global_countNeutrons(float *numNeutrons, const float *weightH, const float *weightV, const float *modFlux, const int numElements)

  global_countNeutrons<<<blocksPerGrid, threadsPerBlock>>>
    (d_nSum, d_weightHg, d_weightVg, sourceDeltaLambda, d_modFlux, numElements);

  //Copy total out of device memory for host reporting
  checkCudaErrors(cudaMemcpy(h_nSum, d_nSum, sizeof(float), cudaMemcpyDeviceToHost));
  
  flux = *h_nSum;
  //  eFlux = *d_nSum;
}





void Sandman::sandCountNeutronsSquareCorrected()
{
  ///
  /// Integrates over all trajectories to estimate the total neutron current,
  /// and divides by Pi/2 to normalise for square window beam area
  /// 
  /// @param nSum pointer to single host memory float to store answer 
  /// @param nSumErr pointer to single host memory float for statistical 
  /// error on total
  ///
  /// \todo Either provide or remove nSum nSumErr functionality
  ///   
  
  sandCountNeutrons();
    
  flux  = flux  / (PI_FLOAT/4.0f);

  std::cout << "Square beam corrected neutron counter:" << std::endl;
  std::cout << "  Got " << flux  << " pseudo neutrons (weight product from both planes)" << std::endl;
}




void Sandman::sandCountNeutronsCircleCorrected()
{
  ///
  /// Integrates over all trajectories to estimate the total neutron current,
  /// and divides by Pi/2 to normalise for square window beam area
  /// 
  /// @param nSum pointer to single host memory float to store answer 
  /// @param nSumErr pointer to single host memory float for statistical 
  /// error on total
  ///
  /// \todo Either provide or remove nSum nSumErr functionality
  ///   
  

  sandCountNeutrons();
    
  flux  = flux / (PI_FLOAT/2.0f);

  std::cout << "Circular beam corrected neutron counter:" << std::endl;
  std::cout << "  Got " << flux  << " pseudo neutrons (weight product from both planes)" << std::endl;
}



void Sandman::sandCountTrajectories(void)
{
  ///
  /// Integrates over all trajectories to estimate the total neutron current.
  /// 
  /// @param nSum pointer to single host memory float to store answer 
  /// @param nSumErr pointer to single host memory float for statistical 
  /// error on total
  ///
  /// \todo Either provide or remove nSum nSumErr functionality
  ///   

  float *d_nSum;

  float h_nSum[1];  //count, error that way we have one memory transfer for everything
  
  checkCudaErrors(cudaMalloc((void **)&d_nSum, sizeof(float)));

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

  if(showCUDAsteps)
    printf("CUDA kernel count neutrons with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);


  // Zero the count on the host
  h_nSum[0] = 0.0f;

  // Copy the zero total to device memory
  checkCudaErrors(cudaMemcpy(d_nSum, h_nSum, sizeof(float), cudaMemcpyHostToDevice));
  
  
  printf("Counting up phase space\n");

  // static void global_countNeutrons(float *numNeutrons, const float *weightH, const float *weightV, const float *modFlux, const int numElements)

  global_countTrajectories<<<blocksPerGrid, threadsPerBlock>>>
    (d_nSum, d_weightHg, d_weightVg, numElements);

  //Copy total out of device memory for host reporting
  checkCudaErrors(cudaMemcpy(h_nSum, d_nSum, sizeof(float), cudaMemcpyDeviceToHost));
  
  traj = *h_nSum;
  //  eFlux = *d_nSum;
}





void Sandman::lambdaMonitor(const std::string setFilename, const float setLambdaMin, const float setLambdaMax, int setLambdaHistSize)
{

  ///
  /// Sets up a wavelength spectrum histogram to be completed by the destructor.
  /// 
  /// @param setFilename std::string name of file to use for output of the histogram.
  /// @param setLambdaMin the minimum wavelength value to use
  /// @param setLambdaMax the maximum wavelength value to use
  /// @param setLambdaHistSize the number of bins in the histogram (max 100)
  ///


  std::string manipulatedFilename;
  
  lambdaMin = setLambdaMin;
  lambdaMax = setLambdaMax;
  if(abs(lambdaMax - lambdaMin) < 0.0001)
    {
      //That would produce an error, make wavelength band 1.0 angstroms
      lambdaMax = lambdaMin + 1.0f;
    }

  lambdaHistSize = setLambdaHistSize;
  
  if(lambdaHistSize > 100)
    {
      lambdaHistSize = 100;
    }


  manipulatedFilename = setFilename;
  manipulatedFilename = remove_extension(manipulatedFilename);
  
  manipulatedFilename = manipulatedFilename + "Lambda1D.dat";
  

  lambdaFileName = manipulatedFilename;
  //Allocate arrays.  The actual lambda monitor is called in the destructor
  //once the trajectory weights are known
  
  checkCudaErrors(cudaMalloc((void **)&d_lambdaMonHist, 100* sizeof(float)));

  if(d_lambdaMonHist == NULL)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " failure to allocate array d_lambdaMonHist" << std::endl;
      exit(1);
    }

}





void Sandman::executeLambdaMonitor(void)
{

  ///
  /// Performs the wavelength histogram calculation set up by lambdaMonitor,
  /// when called by the destructor.
  /// 


  float *h_lambdaHist=NULL;
  float runningLambda;

  float lambdaIntegral=0.0;

  if(lambdaHistSize > 100)
    lambdaHistSize = 100;

  int i;

  const float dLambda=(lambdaMax-lambdaMin) / (float)lambdaHistSize;

  std::ofstream outfile;

  std::cout << color_yellow << "LAMBDA HISTOGRAM CONSTRUCTION" << color_reset << std::endl;

  outfile.open(lambdaFileName.c_str());
  if(outfile.fail())
    {
      std::cerr << "ERROR opening file " << lambdaFileName << std::endl;
      return;
    }

  
  h_lambdaHist = (float*) malloc(lambdaHistSize*sizeof(float));
  
  if(h_lambdaHist == NULL)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " allocating host memory in executeLambdaMonitor" << std::endl;
      exit(1);
    }

  if(d_histogram1D == NULL)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " device memory pointer is NULL in executeLambdaMonitor" << std::endl;
      exit(1);
    }


#ifdef DEBUG
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
      std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
    if (errAsync != cudaSuccess)
      std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif


  // Zero the count histogram
  zeroHistogram1D();

#ifdef DEBUG
    if (errSync != cudaSuccess) 
      std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
    if (errAsync != cudaSuccess)
      std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif


  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    std::cout << "CUDA kernel lambdamonitor[" << lambdaHistSize << "] with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

  //void global_lambdaMonitor(float *lambdaHist, const float lambdaMin, const float dLambda, int histSize, const float *lambdaH, const float *lambdaV, const float *weightH, const float *weightV, const int numElements)

  global_lambdaMonitor<<<blocksPerGrid, threadsPerBlock>>>
    (d_histogram1D, lambdaMin, dLambda, lambdaHistSize, d_lambdag, d_weightHg, d_weightVg, d_modFlux, sourceDeltaLambda, numElements);

  //Copy total out of device memory for host reporting
  checkCudaErrors(cudaMemcpy(h_lambdaHist, d_histogram1D, lambdaHistSize*sizeof(float), cudaMemcpyDeviceToHost));


  //Write out file from host memory

  runningLambda = lambdaMin;

  for(i=0; i < lambdaHistSize; i++)
    {
      outfile << runningLambda << "  " << h_lambdaHist[i] << std::endl;
      runningLambda = runningLambda + dLambda;
      lambdaIntegral += h_lambdaHist[i]*dLambda;
    }
  
  outfile.close();

  std::cout << "\tLambda monitor file written.  Integral current = " << lambdaIntegral << " n/s" << std::endl;

  free(h_lambdaHist);
}





void Sandman::sandPosMonitorH(const std::string filename, const float min, const float max, int histSize)
{
  ///
  /// Sets up a position histogram to be completed by the destructor.
  /// 
  /// @param filename std::string name of file to use for output of the
  /// histogram.  
  /// @param min the minimum position value to use 
  /// @param max the maximum position value to use 
  /// @param histSize the number of bins in the histogram (max 100)
  ///
  /// \todo Complete this function, like the lambdahistrogram function
  ///

  float *h_hist;
  float runningX;
  if(histSize > 100)
    histSize = 100;

  int i;

  const float dval=fabs(max-min) / (float)histSize;

  std::ofstream outfile;

  outfile.open(filename.c_str());
  if(outfile.fail())
    {
      std::cerr << "ERROR opening file " << filename << std::endl;
      return;
    }

  
  h_hist = (float*) malloc(histSize*sizeof(float));
  
  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

  if(showCUDAsteps)
    printf("CUDA posMonitorH with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);
  
  printf("H position monitor\n");

  // Zero the count histogram
  zeroHistogram1D();

   
  //void global_lambdaMonitor(float *lambdaHist, const float lambdaMin, const float dLambda, int histSize, const float *lambdaH, const float *lambdaV, const float *weightH, const float *weightV, const int numElements)

  global_Monitor1D<<<blocksPerGrid, threadsPerBlock>>>
    (d_histogram1D, min, dval, histSize, d_pointsYH, d_weightHg, numElements);

  //Copy total out of device memory for host reporting
  checkCudaErrors(cudaMemcpy(h_hist, d_histogram1D, histSize*sizeof(float), cudaMemcpyDeviceToHost));


  //Write out file from host memory

  runningX = min;

  for(i=0; i<histSize; i++)
    {
      outfile << runningX << "  " << h_hist[i] << std::endl;
      runningX = runningX + dval;
    }
  

  outfile.close();

  free(h_hist);
  
}






void Sandman::phaseSpaceMapH(const char *filename, const float ymin, const float ymax, const float thetaMin, const float thetaMax)
{
  //Create snapshots  
  strcpy(filenameSnapshot, filename);
  yminSnapshot = ymin;
  ymaxSnapshot = ymax;
  thetaMinSnapshot = thetaMin;
  thetaMaxSnapshot = thetaMax;
  
  
  if( d_pointsThetaHsnapshot != NULL ||
      d_pointsYHsnapshot != NULL )
    {
      std::cout << color_red << "ERROR:" << color_reset << " only one type of beam monitor snapshot can be created" << std::endl;
      exit(1);
    }
  
  checkCudaErrors(cudaMalloc((void **)&d_pointsThetaHsnapshot, numElements*sizeof(float)));
  
  if(d_pointsThetaHsnapshot == NULL)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " failed to allocate device memory in setupPhaseSpaceMapH for theta" << std::endl;
      exit(1);
    }
  
  checkCudaErrors(cudaMalloc((void **)&d_pointsYHsnapshot, numElements*sizeof(float)));
  
  if(d_pointsYHsnapshot == NULL)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " failed to allocate device memory in setupPhaseSpaceMapH for Y" << std::endl;
      exit(1);
    }

  if(d_pointsYH == NULL || d_pointsThetaH == NULL)
    {
      std::cerr << "OMG: Copying from unallocated array" << std::endl;
      exit(1);
    }

  //If we get here, then the memory was allocated just fine.
  
  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    std::cout << "CUDA kernel copyArray for phaseSpaceMapH with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
 

  //Snapshot negative theta
  global_copyArray<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsThetaH, d_pointsThetaHsnapshot, numElements, true);


  //Snapshot positive Y
  global_copyArray<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsYH, d_pointsYHsnapshot, numElements, false);
  
}












void Sandman::executePhaseSpaceMapH(void)
{
  ///
  /// Computes a full phase space map in the horizontal plane
  /// 
  /// @param filename pointer to const char name of file to use for output of
  /// the histogram.  
  ///
  /// @param ymin the minimum position value to use (m)
  /// 
  /// @param ymax the maximum position value to use (m)
  ///
  /// @param thetaMin  the minimum divergence value to use (radians)
  ///
  /// @param thetaMax the maximum divergence value to use (radians)
  ///
  
  float *h_histogram=NULL;
  float *d_boundary=NULL;
  
  float runningY = yminSnapshot;
  float runningTheta = thetaMinSnapshot;
  float dy = fabs(ymaxSnapshot-yminSnapshot)/100.0f;
  float dtheta = fabs(thetaMaxSnapshot - thetaMinSnapshot)/100.0f;
  
  std::cout << color_yellow << "HORIZONTAL ACCEPTANCE DIAGRAM CONSTRUCTION" << color_reset << std::endl;
  
  h_histogram = (float*) malloc(100*100*sizeof(float));
  
  if(h_histogram == NULL)
    {
      std::cerr << "Error allocating host memory in phaseSpaceMapH" << std::endl;
      exit(1);
    }
  
  std::ofstream dataFile;
  int i,j;
  
  // Allocate device float for min, max etc
  checkCudaErrors(cudaMalloc((void **)&d_boundary, sizeof(float)));
  if(d_boundary == NULL)
    {
      std::cerr << "Error allocating device memory in phaseSpaceMapH for d_boundary" << std::endl;
      exit(1);
    }
  
  
  
  
   // Zero the count histogram
  zeroHistogram2D();
  
  
#ifdef DEBUG
  cudaError_t errSync  = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess) 
    std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
  if (errAsync != cudaSuccess)
    std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif
  
  
  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CUDA kernel rebinnedPhaseSpaceH with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
  
  
  
  //void global_rebinnedPhaseSpaceH(float globalHist[100][100], const float *d_pointsY, const float *d_pointsTheta, const float yMin, const float dy, const float thetaMin, const float dtheta, int histSize, const float *d_weight, const int numElements)
  
  global_rebinnedPhaseSpaceH<<<blocksPerGrid, threadsPerBlock>>>
    ((float (*)[100])d_histogram2D, d_pointsYHsnapshot, d_pointsThetaHsnapshot, yminSnapshot, dy, thetaMinSnapshot, dtheta, 100, d_weightHg, numElements);
  
  
#ifdef DEBUG
  if (errSync != cudaSuccess) 
    std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
  if (errAsync != cudaSuccess)
    std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif
  
  
  
  
  
  //Get data from GPU
  
  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram2D, 100*100 * sizeof(float), cudaMemcpyDeviceToHost));
  
  
  
  
  
  dataFile.open(filenameSnapshot);
  
  if(!dataFile.good())
    {
      std::cerr << "ERROR opening " << filenameSnapshot << " for writing" << std::endl;
      return;
    }
  else
    std::cout << "\tWriting 2D monitor file " << filenameSnapshot << std::endl;
  
  
  for(i=0; i<100; i++)
    {
      for(j=0; j<100; j++)
	{
	  runningTheta = thetaMinSnapshot + dtheta * (float) j;
	  runningY = yminSnapshot + dy * (float) i;
	  //[theta][y]
	  dataFile << runningTheta << " " << runningY << "  " << h_histogram[j*100+i] << std::endl;
	}
    }
  
  dataFile.close();
  
  free(h_histogram);
  
  if(d_boundary != NULL)
    checkCudaErrors(cudaFree(d_boundary));
}











void Sandman::phaseSpaceMapH(const char *filename)
{
  ///
  /// Computes a full phase space map in the horizontal plane, autodetecting
  /// the boundaries.
  /// 
  /// @param filename pointer to const char name of file to use for output of
  /// the histogram.  
  ///
  /// 
  /// \todo THIS FUNCTION DOES NOT WORK.  It relies on the 2D histogram code,
  /// which currently has a race condition that needs fixing.
  ///
  
  // float *h_histogram=NULL;
  float *d_boundary=NULL;

  // float runningY;
  // float runningTheta;
  float dy;
  float dtheta;
  
  float thLo, thHi, yLo, yHi;


  // h_histogram = (float*) malloc(100*100*sizeof(float));

  // if(h_histogram == NULL)
  //   {
  //     std::cerr << "Error allocating host memory in phaseSpaceMapH" << std::endl;
  //     exit(1);
  //   }

  // std::ofstream dataFile;
  // int i,j;

  // Allocate device float for min, max etc
  checkCudaErrors(cudaMalloc((void **)&d_boundary, sizeof(float)));
  if(d_boundary == NULL)
    {
      std::cerr << "Error allocating device memory in phaseSpaceMapH for d_boundary" << std::endl;
      exit(1);
    }
  

  
  //Autodetect minimum and maximum theta
  std::cout << "  Phase space theta minimum:" << std::endl;
  thLo = arrayMinimum(d_pointsThetaH, d_boundary);
  std::cout << "  Phase space theta maximum:" << std::endl;
  thHi = arrayMaximum(d_pointsThetaH, d_boundary);
  dtheta = fabs(thLo-thHi)/100.0f;
  
  //Pad by one bin
  thLo = thLo - dtheta;
  thHi = thHi + dtheta;



  //Autodetect minimum and maximum y
  std::cout << "  Phase space Y minimum:" << std::endl;
  yLo = arrayMinimum(d_pointsYH, d_boundary);
  std::cout << "  Phase space Y maximum:" << std::endl;
  yHi = arrayMaximum(d_pointsYH, d_boundary);

  //Pad by one bin
  dy = fabs(yHi - yLo)/100.0f;
  yLo = yLo - dy;
  yHi = yHi + dy;

  //Pipe this now through the other function
  
  //void Sandman::phaseSpaceMapH(const char *filename, const float ymin, const float ymax, const float thetaMin, const float thetaMax)
  phaseSpaceMapH(filename, yLo, yHi, thLo, thHi);
  

//   // Zero the count histogram
//   zeroHistogram2D();


// #ifdef DEBUG
//     cudaError_t errSync  = cudaGetLastError();
//     cudaError_t errAsync = cudaDeviceSynchronize();
//     if (errSync != cudaSuccess) 
//       std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
//     if (errAsync != cudaSuccess)
//       std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
// #endif




//   printf("2D histogram phase space H...\n\n");
  
//    int threadsPerBlock = 256;
//    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
//    std::cout << "CUDA kernel rebinnedPhaseSpaceH, auto boundary detect, with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
 


//    //void global_rebinnedPhaseSpaceH(float globalHist[100][100], const float *d_pointsY, const float *d_pointsTheta, const float yMin, const float dy, const float thetaMin, const float dtheta, int histSize, const float *d_weight, const int numElements)

//     // global_rebinnedPhaseSpaceH<<<blocksPerGrid, threadsPerBlock>>>
//     //   ((float (*)[100])d_histogram2D, d_pointsYH, d_pointsThetaH, ymin, dy, thetaMin, dtheta, 100, d_weightHg, numElements);

//     global_rebinnedPhaseSpaceH<<<blocksPerGrid, threadsPerBlock>>>
//       ((float (*)[100])d_histogram2D, d_pointsYH, d_pointsThetaH, yLo, dy, thLo, dtheta, 100, d_weightHg, numElements);

// #ifdef DEBUG
//     if (errSync != cudaSuccess) 
//       std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
//     if (errAsync != cudaSuccess)
//       std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
// #endif





  // //Get data from GPU
   
  //  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram2D, 100*100 * sizeof(float), cudaMemcpyDeviceToHost));




  
  // dataFile.open(filename);
  
  // if(!dataFile.good())
  //   {
  //     std::cerr << "ERROR opening " << filename << " for writing" << std::endl;
  //     return;
  //   }
  // else
  //   std::cout << "Writing 2D monitor file " << filename << std::endl;
  
  
  // for(i=0; i<100; i++)
  //   {
  // for(j=0; j<100; j++)
  //   {
  // runningTheta = thLo + dtheta * (float) j;
  // runningY = yLo + dy * (float) i;
  // //[theta][y]
  // dataFile << runningTheta << " " << runningY << "  " << h_histogram[j*100+i] << std::endl;
  //   }
// }
  
//   dataFile.close();
  
  // free(h_histogram);

  if(d_boundary != NULL)
    checkCudaErrors(cudaFree(d_boundary));
}



 




void Sandman::phaseSpaceMapHCPU(const char *filename)
{

  ///
  /// Computes a full phase space map in the horizontal plane, autodetecting
  /// the boundaries.  This fuction runs on the CPU and requires the full
  /// phase space to be copied over to host Ram, so it is SLOOOOOW.
  ///
  /// @param filename pointer to const char name of file to use for output of
  /// the histogram.  
  /// 
  /// However, it is provided because it is probably very good for unit
  /// testing etc.
  ///

  
  float *h_pointsY=NULL;
  float *h_pointsTheta=NULL;
  float *h_weight=NULL;

  int dumped=0;

  h_pointsY = (float*) malloc(numElements*sizeof(float));
  if(h_pointsY == NULL)
    {
      std::cerr << "phaseSpaceMapH cannot allocate memory for h_pointsY" << std::endl;
      exit(1);
    }

  h_pointsTheta = (float*) malloc(numElements*sizeof(float));

  if(h_pointsTheta == NULL)
    {
      std::cerr << "phaseSpaceMapH cannot allocate memory for h_pointsTheta" << std::endl;
      exit(1);
    }

  h_weight = (float*) malloc(numElements*sizeof(float));

  if(h_weight == NULL)
    {
      std::cerr << "phaseSpaceMapH cannot allocate memory for h_weight" << std::endl;
      exit(1);
    }



  std::ofstream dataFile;
  int i;

  //Get data from GPU
  
  sandGetPhaseSpaceH(h_pointsY, h_pointsTheta, h_weight);


  
  dataFile.open(filename);
  
  if(!dataFile.good())
    {
      std::cerr << "ERROR opening " << filename << " for writing" << std::endl;
      return;
    }
  
  //Limit the output to 20000 points - this could be a shit load of data
  for(i=0; i<numElements && dumped<200000; i++)
    {
      if(h_weight[i] > deadWeight)
	{
	  dataFile << h_pointsTheta[i]*180.0f/PI_FLOAT << "\t" << h_pointsY[i] << "\t" << h_weight[i] << std::endl;
	  dumped++;
	}
    }
  
  dataFile.close();

  free(h_pointsY);
  free(h_pointsTheta);
  free(h_weight);
}




void Sandman::phaseSpaceMapVCPU(const char *filename)
{

  ///
  /// Computes a full phase space map in the vertical plane, autodetecting
  /// the boundaries.  This fuction runs on the CPU and requires the full
  /// phase space to be copied over to host Ram, so it is SLOOOOOW.
  ///
  /// @param filename pointer to const char name of file to use for output of
  /// the histogram.  
  /// 
  /// However, it is provided because it is probably very good for unit
  /// testing etc.
  ///

  
  float *h_pointsY=NULL;
  float *h_pointsTheta=NULL;
  float *h_weight=NULL;

  h_pointsY = (float*) malloc(numElements*sizeof(float));
  if(h_pointsY == NULL)
    {
      std::cerr << "phaseSpaceMapH cannot allocate memory for h_pointsY" << std::endl;
      exit(1);
    }

  h_pointsTheta = (float*) malloc(numElements*sizeof(float));

  if(h_pointsTheta == NULL)
    {
      std::cerr << "phaseSpaceMapH cannot allocate memory for h_pointsTheta" << std::endl;
      exit(1);
    }

  h_weight = (float*) malloc(numElements*sizeof(float));

  if(h_weight == NULL)
    {
      std::cerr << "phaseSpaceMapH cannot allocate memory for h_weight" << std::endl;
      exit(1);
    }



  std::ofstream dataFile;
  int i;

  //Get data from GPU
  
  sandGetPhaseSpaceV(h_pointsY, h_pointsTheta, h_weight);


  
  dataFile.open(filename);
  
  if(!dataFile.good())
    {
      std::cerr << "ERROR opening " << filename << " for writing" << std::endl;
      return;
    }
  
  //Limit the output to 200000 points - this could be a shit load of data
  for(i=0; i<numElements && i<200000; i++)
    {
      if(h_weight[i] > deadWeight)
	dataFile << h_pointsTheta[i]*180.0f/PI_FLOAT << "\t" << h_pointsY[i] << "\t" << h_weight[i] << std::endl;
    }
  
  dataFile.close();

  free(h_pointsY);
  free(h_pointsTheta);
  free(h_weight);
}




void Sandman::debugPosPosCPU(const char *filename)
{

  ///
  /// Computes a full phase space map in the vertical plane, autodetecting
  /// the boundaries.  This fuction runs on the CPU and requires the full
  /// phase space to be copied over to host Ram, so it is SLOOOOOW.
  ///
  /// @param filename pointer to const char name of file to use for output of
  /// the histogram.  
  /// 
  /// However, it is provided because it is probably very good for unit
  /// testing etc.
  ///

  
  float *h_pointsH=NULL;
  float *h_weightH=NULL;
  float *h_pointsV=NULL;
  float *h_weightV=NULL;



  h_pointsH = (float*) malloc(numElements*sizeof(float));
  if(h_pointsH == NULL)
    {
      std::cerr << "DebugPosPosCPU cannot allocate memory for h_pointsH" << std::endl;
      exit(1);
    }

  h_pointsV = (float*) malloc(numElements*sizeof(float));
  if(h_pointsV == NULL)
    {
      std::cerr << "DebugPosPosCPU cannot allocate memory for h_pointsV" << std::endl;
      exit(1);
    }

  h_weightH = (float*) malloc(numElements*sizeof(float));
  if(h_weightH == NULL)
    {
      std::cerr << "DebugPosPosCPU cannot allocate memory for h_weightH" << std::endl;
      exit(1);
    }

  h_weightV = (float*) malloc(numElements*sizeof(float));
  if(h_weightV == NULL)
    {
      std::cerr << "DebugPosPosCPU cannot allocate memory for h_weightV" << std::endl;
      exit(1);
    }

  std::ofstream dataFile;
  int i;
  int dumped=0;

  //Get data from GPU
  
  sandDebugPosPos(h_pointsH, h_weightH, h_pointsV, h_weightV);


  
  dataFile.open(filename);
  
  if(!dataFile.good())
    {
      std::cerr << "ERROR opening " << filename << " for writing" << std::endl;
      return;
    }
  
  //Limit the function to considering 100000 points - this could be a shit load of data
  for(i=0; i<numElements && dumped < 100000; i++)
    {
      if(h_weightH[i] > deadWeight && h_weightV[i] > deadWeight)
	{
	  dataFile << h_pointsH[i] << "\t" << h_pointsV[i] << "\t" << h_weightH[i]*h_weightV[i] << std::endl;
	  dumped++;
	}
    }
  
  dataFile.close();

  free(h_pointsH);
  free(h_pointsV);
  free(h_weightH);
  free(h_weightV);
}






void Sandman::sandSkewCUDA(const float distance_m)
{

  ///
  /// Calls the CUDA kernels to compute a skew operation on both phase space
  /// maps to propagate the beam a certain distance within the small angle
  /// limit.
  ///
  /// @param distance_m the distance the beam must propagate in metres.
  ///
  
   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   
   if(showCUDAsteps)  
     printf("CUDA kernel skew with %d blocks of %d threads\n", blocksPerGrid,
	    threadsPerBlock);

  

   //void device_sandSkewCUDA(float *d_pointsY, const float *d_pointsTheta, float *d_weight, const float distance_m, const int numElements)
   
    global_sandSkewCUDA<<<blocksPerGrid, threadsPerBlock>>>
      (d_pointsYH, d_pointsThetaH, distance_m, numElements);

    global_sandSkewCUDA<<<blocksPerGrid, threadsPerBlock>>>
      (d_pointsYV, d_pointsThetaV, distance_m, numElements);
   
}




void Sandman::sandCollimateCUDA(const float divergenceH, const float divergenceV)
{
  ///
  /// Calls the CUDA kernels to compute a collimation operation, setting the
  /// weight to zero on trajectories falling outside the divergence window
  /// requested.
  ///
  /// @param divergenceH the half width divergence limit in the horizontal plane (radians)
  ///
  /// @param divergenceV the half width divergence limit in the vertical plane (radians)
  ///
  
   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel collimation at %f and %f with %d blocks of %d threads\n", divergenceH, divergenceV, blocksPerGrid,
	  threadsPerBlock);

   // void global_collimation(float *d_weight, const float *d_pointsTheta, const float lower_angle, const float upper_angle, const int numElements)


    global_collimation<<<blocksPerGrid, threadsPerBlock>>>
      (d_weightHg, d_pointsThetaH, -fabs(divergenceH), fabs(divergenceH), numElements);

    global_collimation<<<blocksPerGrid, threadsPerBlock>>>
      (d_weightVg, d_pointsThetaV, -fabs(divergenceV), fabs(divergenceV), numElements);

   
}




void Sandman::sandApertureCUDA(const float window_width, const float window_height)
{
  ///
  /// Calls the CUDA kernels to compute an aperture operation, setting the
  /// weight to zero on trajectories falling outside the position window
  /// requested.
  ///
  /// @param window_width the full width of the window in metres.
  ///
  /// @param window_height the full height of the window in metres.
  ///
  
   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel aperture of width %f and height %f with %d blocks of %d threads\n", window_width, window_height, blocksPerGrid,
	    threadsPerBlock);

   // void global_collimation(float *d_weight, const float *d_pointsTheta, const float lower_angle, const float upper_angle, const int numElements)


    global_aperture<<<blocksPerGrid, threadsPerBlock>>>
      (d_weightHg, d_pointsYH, -fabs(window_width/2.0f), fabs(window_width/2.0f), numElements);

    global_aperture<<<blocksPerGrid, threadsPerBlock>>>
      (d_weightVg, d_pointsYV, -fabs(window_height/2.0f), fabs(window_height/2.0f), numElements);
}





void Sandman::sandModerator(const float width,
		     const float height,
		     const float hoffset,
		     const float voffset,
		     const float temp,
		     const float num)
{

  ///
  /// Calls the CUDA kernels to compute a single moderator window, which sets
  /// the weight to zero on trajectories falling outside the position window
  /// requested, and calculates the neutron current represented by the
  /// trajectory.
  ///
  /// @param width the width of the moderator in metres
  ///
  /// @param height the height of the moderator in metres
  /// 
  /// @param hoffset the perpendicular horizontal offset of the moderator
  /// (left is positive, imagined from a view top down with the moderator at
  /// the bottom and the sample at the top, relative to the beam axis centre
  /// at the guide entrance.
  ///
  /// @param voffset the perpendicular vertical offset of the moderator (up is
  /// positive, imagined from a side view with the moderator on the left and
  /// the sample to the right, relative to the beam axis centre at the guide
  /// entrance.
  ///
  /// @param temp the characteristic temperature of the maxwellian distribution (kelvin)
  ///
  /// @param num the characteristic brightness of the maxwellian distribution
  /// (neutrons per second per cm2 per steradian per angstrom)
  /// 
  /// @note the maxwellian distribution calculation is the same used in MCSTAS
  /// (and nads).  VITESS uses a different definition of brightness and solid
  /// angle.
  ///
  


   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel sandModerator of width %f and height %f with %d blocks of %d threads\n", width, height, blocksPerGrid,
	    threadsPerBlock);

   
   
   //static void global_sandModerator1(float *d_modFluxH, float *d_weightH, const float *d_lambdag, const float *d_pointsYH, const int numElements, const float width, const float hoffset, const float temp, const float num)

    global_sandModerator1<<<blocksPerGrid, threadsPerBlock>>>
      (d_modFlux, d_weightHg, d_lambdag, d_pointsYH, numElements, width, hoffset, temp, num);

    global_sandModerator1<<<blocksPerGrid, threadsPerBlock>>>
      (d_modFlux, d_weightVg, d_lambdag, d_pointsYV, numElements, width, hoffset, temp, num);


}





void Sandman::sandILLHCSModerator(void)
{
  ///
  /// A tool to call a standard moderator kernel providing a triple maxwellian
  /// moderator matching the ILL horizontal cold source dimensions, based on
  /// the work of E. Farhi in 2008-2009 to calculate the absolute brightness
  /// via extrapolation.  This benchmark moderator was used in the NADS work,
  /// so is a useful cross-check.
  ///

  sandApertureCUDA(0.186, 0.186);
  
   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel sandILLHCSModerator with %d blocks of %d threads\n", blocksPerGrid,
	    threadsPerBlock);

   //Moderator brightness curve
   if(d_modFlux == NULL)
     checkCudaErrors(cudaMalloc((void **)&d_modFlux, numElements * sizeof(float)));
   
   if(d_modFlux == NULL)
     {
       std::cerr << color_red << "ERROR:" << color_reset << " failure to allocate memory for moderator brightness curve" << std::endl;
       exit(1);
     }


   //global_sandILLHCSModerator(float *d_modFluxH, float *d_weightH, const float *d_lambdag, const float *d_pointsYH, const int numElements)
    global_sandILLHCSModerator<<<blocksPerGrid, threadsPerBlock>>>
      (d_modFlux, d_weightHg, d_lambdag, d_pointsYH, numElements);

    global_sandILLHCSModerator<<<blocksPerGrid, threadsPerBlock>>>
      (d_modFlux, d_weightVg, d_lambdag, d_pointsYV, numElements);


}





















void Sandman::sandReflectionH(const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom)
{
  ///
  /// Calls the CUDA kernels to compute a single channel guide reflection in
  /// the horizontal plane
  ///
  /// @param mirrorYtop upper mirror surface in phase space (since this is horizontal, top = left) in metres
  ///
  /// @param mirrorYbottom lower mirror surface in phase space (since this is horizontal, bottom = right) in metres
  /// 
  /// @param mirrorAngleTop angle of inclination of upper mirror surface (radians)
  /// 
  /// @param mirrorAngleBottom angle of inclination of lower mirror surface (radians)
  /// 
  /// @param mTop supermirror m value of upper mirror
  /// 
  /// @param mBottom supermirror m value of lower mirror
  /// 
  /// 
  /// @note the maths from this operation is a carbon copy of the nads code 
  ///
  
  
   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel reflection with %d blocks of %d threads\n", blocksPerGrid,
	    threadsPerBlock);

  


   /* void device_sandReflection(float *d_pointsY, float *d_pointsTheta, const float *d_lambda, float
    *d_weight, const float mirrorY1, const float mirrorY2, const float mirrorAngle1, const float mirrorAngle2, const float mValue, const int numElements) */


    global_sandReflection<<<blocksPerGrid, threadsPerBlock>>>
      (d_pointsYH, d_pointsThetaH, d_lambdag, d_weightHg, mirrorYtop, mirrorYbottom, mirrorAngleTop, mirrorAngleBottom, mTop, mBottom, numElements);

}


void Sandman::sandReflectionV(const float mirrorYtop, const float mirrorYbottom, const float mirrorAngleTop, const float mirrorAngleBottom, const float mTop, const float mBottom)
{

  ///
  /// Calls the CUDA kernels to compute a single channel guide reflection in
  /// the vertical plane
  ///
  /// @param mirrorYtop upper mirror surface in phase space in metres
  ///
  /// @param mirrorYbottom lower mirror surface in phase space in metres
  /// 
  /// @param mirrorAngleTop angle of inclination of upper mirror surface (radians)
  /// 
  /// @param mirrorAngleBottom angle of inclination of lower mirror surface (radians)
  /// 
  /// @param mTop supermirror m value of upper mirror
  /// 
  /// @param mBottom supermirror m value of lower mirror
  /// 
  /// 
  /// @note the maths from this operation is a carbon copy of the nads code 
  ///

  
   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel reflection with %d blocks of %d threads\n", blocksPerGrid,
	    threadsPerBlock);

  


   /* void device_sandReflection(float *d_pointsY, float *d_pointsTheta, const float *d_lambda, float
    *d_weight, const float mirrorY1, const float mirrorY2, const float mirrorAngle1, const float mirrorAngle2, const float mValue, const int numElements) */


    global_sandReflection<<<blocksPerGrid, threadsPerBlock>>>
      (d_pointsYV, d_pointsThetaV, d_lambdag, d_weightVg, mirrorYtop, mirrorYbottom, mirrorAngleTop, mirrorAngleBottom, mTop, mBottom, numElements);

}



void Sandman::sandRotation(const float angleH, const float angleV)
{
  ///
  /// Calls the CUDA kernels to shift both horizontal and vertical phase spaces in the theta plane (rotation of beam)
  ///
  /// @param angleH horizontal angle of beam rotation (radians)
  ///
  /// @param angleV vertical angle of beam rotation (radians)
  /// 
  /// \todo Check in NADS and document the positive / negative axes of this function.
  ///

  

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel rotation with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);
   
  //static void global_rotation(float *d_pointsTheta, const float angle_radians, const int numElements)
  
  
  global_rotation<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsThetaH, angleH, numElements);
  
  global_rotation<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsThetaV, angleV, numElements);
  
}



void Sandman::sandRotationH(const float angleH)
{
  ///
  /// Calls the CUDA kernel to shift the horizontal phase space in the theta plane (rotation of beam)
  ///
  /// @param angleH horizontal angle of beam rotation (radians)
  /// 
  /// \todo Check in NADS and document the positive / negative axes of this function.
  ///

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel rotationH with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);
   
  //static void global_rotation(float *d_pointsTheta, const float angle_radians, const int numElements)
  
  
  global_rotation<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsThetaH, angleH, numElements);
  
}




void Sandman::sandTranslationH(const float distance)
{
  ///
  /// Calls the CUDA kernel to shift the horizontal phase space in the y plane (shift of beam axis)
  ///
  /// @param distance horizontal shift of beam (metres)
  /// 
  /// \todo Check in NADS and document the positive / negative axes of this function.
  ///

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel translationH with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);
   

  //static void global_translation(float *d_pointsY, const float distance_m, const int numElements)

  
  global_translation<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsYH, distance, numElements);
  
}


void Sandman::sandTranslationV(const float distance)
{
  ///
  /// Calls the CUDA kernel to shift the vertical phase space in the y plane (shift of beam axis)
  ///
  /// @param distance vertical shift of beam (metres)
  /// 

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel translationV with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);
   

  //static void global_translation(float *d_pointsY, const float distance_m, const int numElements)

  
  global_translation<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsYV, distance, numElements);
  
}






void Sandman::sandFreeSpaceCUDA(const float distance, const bool& verbose)
{
  ///
  /// Free space is another name for skew operation.  This models the flight
  /// of a neutron beam in the small angle limit by skewing the phase space.
  ///
  /// @param distance distance to transport the neutron beam (metres)
  /// 

  //This could be a sub-module, so only display if the user explicitly calls
  //this function without flagging the verbose option
  if(verbose)
    std::cout << color_yellow << "FREE SPACE" << color_reset << std::endl;

  sandSkewCUDA(distance);
}




void Sandman::sandGuideElementCUDA(
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
			  )
{
  
  ///
  /// Models a single piece of neutron guide by calling associated class
  /// functions, which in turn call cuda kernels.
  ///
  /// @param length length of guide element in metres
  /// 
  /// @param entr_width width of entrance of guide element in metres
  ///
  /// @param exit_width width of exit of guide element in metres
  ///
  /// @param exit_offset_h horizontal offset of beam centre at the exit, relative to the entrance, in metres.
  ///
  /// @param mLeft the supermirror m value of the left side of the guide (left when looking at sample from neutron point of view)
  ///
  /// @param mRight the supermirror m value of the right side of the guide (right when looking at sample from neutron point of view)
  ///
  /// @param entr_height height of entrance of guide element in metres
  ///
  /// @param exit_height height of exit of guide element in metres
  ///
  /// @param exit_offset_v vertical offset of beam centre at the exit, relative to the entrance, in metres.
  ///
  /// @param mTop the supermirror m value of the top side of the guide
  ///
  /// @param mBottom the supermirror m value of the bottom side of the guide
  ///

  
  const float guideAngleTop = atan( (exit_offset_v + 0.5*(exit_height - entr_height)) / length);
  const float guideAngleBot = atan( (exit_offset_v + 0.5*(entr_height - exit_height)) / length);

  const float guideAngleLeft = atan( (exit_offset_h + 0.5*(exit_width - entr_width)) / length);
  const float guideAngleRight = atan( (exit_offset_h + 0.5*(entr_width - exit_width)) / length);

  //Propagate the neutrons to the end of the guide first
  sandSkewCUDA(length);
  
  //Reflect the vertical plane
  //sandReflectionH(const float mirrorY1, const float mirrorY2, const float mirrorAngle1, const float mirrorAngle2, const float mTop, const float mBottom, const int numElements)
  
  // sandReflectionH(
  // 		  0.5f*exit_width + exit_offset_h, // mirror top 
  // 		  -0.5f*exit_width + exit_offset_h,// mirror bottom
  // 		  guideAngleTop,
  // 		  guideAngleBot,
  // 		  mTop,
  // 		  mBottom);

  // sandReflectionV(
  // 		  0.5f*exit_height + exit_offset_v, mirror top 
  // 		  -0.5f*exit_height + exit_offset_v,mirror bottom
  // 		  guideAngleLeft,
  // 		  guideAngleRight,
  // 		  mLeft,
  // 		  mRight);

  //ERROR - this was H, width, top, bottom!
  sandReflectionV(
  		  0.5f*exit_height + exit_offset_v, //mirror top 
  		  -0.5f*exit_height + exit_offset_v,//mirror bottom
  		  guideAngleTop,
  		  guideAngleBot,
  		  mTop,
  		  mBottom);

  //ERROR - this was V, height, left right!
  sandReflectionH(
  		  0.5f*exit_width + exit_offset_h, //mirror top 
  		  -0.5f*exit_width + exit_offset_h,//mirror bottom
  		  guideAngleLeft,
  		  guideAngleRight,
  		  mLeft,
  		  mRight);

}


void Sandman::sandSimpleStraightGuide(
		       const float length,
		       const float width,
		       const float height,
		       const float mval
				      )
{

  ///
  /// A simple utility function for a straight guide of constant cross section
  /// and a single m value
  ///
  /// @param length length of guide in metres
  /// 
  /// @param width width of guide in metres
  ///
  /// @param height height of guide in metres
  ///
  /// @param mval the supermirror m value of all surfaces
  ///


  //Before we do anything else, kill neutrons missing the entrance of the guide.
  sandApertureCUDA(width, height);

  sandGuideElementCUDA(length,
		       width,
		       width,
		       0.0,
		       mval,
		       mval,
		       height,
		       height,
		       0.0,
		       mval,
		       mval);
}



void Sandman::sandCurvedGuide(
		       const float length,
		       const float sectionLength,
		       const float width,
		       const float height,
		       const float mval,
		       const float radius
			      )
{

  ///
  /// A simple utility function for a curved guide of constant cross section
  /// and a single m value
  ///
  /// @param length length of guide in metres
  /// 
  /// @param sectionLength length of guide sections in metres (typically 0.5,
  /// 1, or 2 metres in practice)
  ///
  /// @param width width of guide in metres
  ///
  /// @param height height of guide in metres
  ///
  /// @param mval the supermirror m value of all surfaces
  ///
  /// @param radius the radius of curvature of the guide in metres
  ///

  int i=0;

  //Before we do anything else, kill neutrons missing the entrance of the guide.
  sandApertureCUDA(width, height);

  std::cout << color_yellow << "CURVED GUIDE CHANNEL" << color_reset << std::endl; 

  if(radius != 0.0)
    {
      //Break into sections
      int numSections = (int) round(length / sectionLength);
      float sectionAngle;

      //Special case - one section.  
      //This is two tweaks of rotation surrounding a short, straight guide piece 
      //the piece plane at the centre lies along the tangent of the curve at that point
      if(2.0*sectionLength > length)
	{
	  sectionAngle = asin(0.5*length / radius);
			
			
	  std::cout << "\tsection " << i+1 << " ";
	  sandRotationH(sectionAngle);
	  
	  //sandSimpleStraightGuide(length, width, height, mval);
	  sandGuideElementCUDA(length,
		       width,
		       width,
		       0.0,
		       mval,
		       mval,
		       height,
		       height,
		       0.0,
		       mval,
		       mval);
	  
	  sandRotationH(sectionAngle);
	  
	  std::cout << "\tCurved guide channel finished" << std::endl;
	  
	  return;
	  


	}


      //Otherwise we do normal curved guide
      sectionAngle = 2.0*asin(0.5*sectionLength / radius);

      //Normal case, many sections of finite length
		for(i=0; i<numSections; i++)
		{
			if(i != numSections-1)	//if we are not doing the last iteration so do a normal straight guide plus rotation
			{
			       
				//sandSimpleStraightGuide(sectionLength, width, height, mval);
				sandGuideElementCUDA(sectionLength,
						     width,
						     width,
						     0.0,
						     mval,
						     mval,
						     height,
						     height,
						     0.0,
						     mval,
						     mval);
				
				std::cout << "\tsection " << i+1 << std::endl;
				sandRotationH(sectionAngle);
			}
			
			else	//This is the last section, so take care with the length if it's not an integer multiple of sections
					//also, there is no rotation.  The next module axis is aligned with this last piece, just as the 
					//entrance is aligned with the previous axis
			{
				float lastPiece = length - (float)i * sectionLength;

				if(lastPiece <= 0.0)	//i don't think that this can happen, but never mind
					break;
				
				std::cout << "\tsection " << i+1 << std::endl;
				//sandSimpleStraightGuide(lastPiece, width, height, mval);
				sandGuideElementCUDA(lastPiece,
						     width,
						     width,
						     0.0,
						     mval,
						     mval,
						     height,
						     height,
						     0.0,
						     mval,
						     mval);
				
			}	

		}
		
		std::cout << "\tcurved guide channel finished" << std::endl;
		

    }

}





void Sandman::ellipticOpeningGuide(const float length, const float exitWidth, const float exitHeight, const float focalPoint1H, const float focalPoint2H, const float focalPoint1V, const float focalPoint2V, const float mNumber, const int numSections)
{
  //Models a focussing elliptic guide by using straight sections
  //Focal lengths are defined relative to the entrance plane
  
  float section_length;
  float pieceEntrWidth;
  float pieceExitWidth;
  float pieceEntrHeight;
  float pieceExitHeight;
  float pieceStartx;
  float pieceEndx;
  float focalpoint1V, focalpoint2V;
  float focalpoint1H, focalpoint2H;
  
	
  const char* filenameH = "hEllipseOpeningProfile.dat";
  const char* filenameV = "vEllipseOpeningProfile.dat";
  
  std::cout << color_yellow << "OPENING HALF ELLIPSE" << color_reset << std::endl; 

  std::ofstream dataFileH;
  dataFileH.open(filenameH);
  
  if(dataFileH.fail())
    {
      std::cerr << "ERROR opening file " << filenameH << std::endl;
      return;
    }
  
  std::ofstream dataFileV;
  dataFileV.open(filenameV);
  
  if(dataFileV.fail())
    {
      std::cerr << "ERROR opening file " << filenameV << std::endl;
      return;
    }
  
  
  int i;
  
  //Break guide into sections
  section_length = length / (float) numSections;
  
  std::cout << "\tLength " << length << " m and exit width = " << exitWidth  << " m focus2 at " << focalPoint2H << " " << focalPoint2V << " and focus1 at " << focalPoint1H << " " << focalPoint1V << " formed by " << numSections << " sections of " << section_length << " m" << std::endl;
  
  //Loop over converging guide approximations printing out the widths
  
  std::cout << "\tProfile:" << std::endl;
  std::cout << "\txpos width" << std::endl;
  
  for (i = 0; i < numSections; i++) 
    {
      pieceStartx = section_length * (float) i;
      pieceEndx = section_length * (float) (i + 1);
      
      //Take JNADS curves and put these into two dimensions
      pieceEntrWidth = 2.0f * elliptic_curve(pieceStartx, focalPoint1H, focalPoint2H, exitWidth);
      pieceExitWidth = 2.0f * elliptic_curve(pieceEndx, focalPoint1H, focalPoint2H, exitWidth);
      std::cout << "\t" << pieceStartx << "  " << pieceEntrWidth << "  " << pieceExitWidth << " H" << std::endl;
      dataFileH << "\t" << pieceStartx << "  " << pieceEntrWidth << std::endl;
      
      pieceEntrHeight = 2.0f * elliptic_curve(pieceStartx, focalPoint1V, focalPoint2V, exitHeight);
      pieceExitHeight = 2.0f * elliptic_curve(pieceEndx, focalPoint1V, focalPoint2V, exitHeight);
      std::cout << "\t" << pieceStartx << "  " << pieceEntrHeight << " V" << std::endl;
      dataFileV << "\t" << pieceStartx << "  " << pieceEntrHeight << std::endl;
      
      if (i == (numSections - 1)) 
	{
	  std::cout << "\t" << pieceEndx << "  exit " << pieceExitWidth << std::endl;
	  dataFileH << "\t" << pieceEndx << "  exit " << pieceExitWidth << std::endl;
	  
	  std::cout << "\t" << pieceEndx << "  exit " << pieceExitHeight << std::endl;
	  dataFileV << "\t" << pieceEndx << "  exit " << pieceExitHeight << std::endl;
	}
      
      sandGuideElementCUDA(length,
			   pieceEntrWidth,
			   pieceExitWidth,
			   0.0f,
			   mNumber,
			   mNumber,
			   pieceEntrHeight,
			   pieceExitHeight,
			   0.0f,
			   mNumber,
			   mNumber
			   );
    }
  
  
  
  std::cout << "\tElliptic opening guide finished" << std::endl;
  
  dataFileH.close();
  dataFileV.close();
}






void Sandman::ellipticClosingGuide(const float length, const float entrWidth, const float entrHeight, const float focalPoint1H, const float focalPoint2H, const float focalPoint1V, const float focalPoint2V, const float mNumber, const int numSections)
{
        //Models a focussing elliptic guide by using straight sections

        float section_length;
        float pieceEntrWidth;
        float pieceExitWidth;
	float pieceEntrHeight;
	float pieceExitHeight;
        float pieceStartx;
        float pieceEndx;
        float focalpoint1V, focalpoint2V;
	float focalpoint1H, focalpoint2H;

	
	const char* filenameH = "hEllipseClosingProfile.dat";
	const char* filenameV = "vEllipseClosingProfile.dat";

	std::cout << color_yellow << "CLOSING HALF ELLIPSE" << color_reset << std::endl;

	std::ofstream dataFileH;
	dataFileH.open(filenameH);

	if(dataFileH.fail())
	  {
	    std::cerr << "ERROR opening file " << filenameH << std::endl;
	    return;
	  }
	
	std::ofstream dataFileV;
	dataFileV.open(filenameV);
	
	if(dataFileV.fail())
	  {
	    std::cerr << "ERROR opening file " << filenameV << std::endl;
	    return;
	  }
	
	  
	int i;
	  
	//Break guide into sections
	section_length = length / (float) numSections;
	
	std::cout << "\tLength " << length << " m and focus2 at " << focalPoint2H << " " << focalPoint2V << " and focus1 at " << focalPoint1H << " " << focalPoint1V << " formed by " << numSections << " sections of m=" << mNumber;
	
	//Loop over converging guide approximations printing out the widths
	
	std::cout << "\tProfile:" << std::endl;
	std::cout << "\txpos  width" << std::endl;
	
	for (i = 0; i < numSections; i++) 
	  {
	    pieceStartx = section_length * (float) i;
	    pieceEndx = section_length * (float) (i + 1);
	    
	    //Take JNADS curves and put these into two dimensions
	    pieceEntrWidth = 2.0f * elliptic_curve(pieceStartx, focalPoint1H, focalPoint2H, entrWidth);
	    pieceExitWidth = 2.0f * elliptic_curve(pieceEndx, focalPoint1H, focalPoint2H, entrWidth);
	    std::cout << "\t" << pieceStartx << "  " << pieceEntrWidth << " H" << std::endl;
	    dataFileH << "\t" << pieceStartx << "  " << pieceEntrWidth << std::endl;
	    
	    pieceEntrHeight = 2.0f * elliptic_curve(pieceStartx, focalPoint1V, focalPoint2V, entrHeight);
	    pieceExitHeight = 2.0f * elliptic_curve(pieceEndx, focalPoint1V, focalPoint2V, entrHeight);
	    std::cout << "\t" << pieceStartx << "  " << pieceEntrHeight << " V" << std::endl;
	    dataFileV << "\t" << pieceStartx << "  " << pieceEntrHeight << std::endl;
	    
	    if (i == numSections - 1) 
	      {
		std::cout << "\t" << pieceEndx << "  " << pieceExitWidth << std::endl;
		dataFileH << "\t" << pieceEndx << "  " << pieceExitWidth << std::endl;

		std::cout << "\t" << pieceEndx << "  " << pieceExitHeight << std::endl;
		dataFileV << "\t" << pieceEndx << "  " << pieceExitHeight << std::endl;
	      }
	    
	    sandGuideElementCUDA(length,
				 pieceEntrWidth,
				 pieceExitWidth,
				 0.0f,
				 mNumber,
				 mNumber,
				 pieceEntrHeight,
				 pieceExitHeight,
				 0.0f,
				 mNumber,
				 mNumber
				 );
	  }
	
	
	
	std::cout << "\tElliptic closing guide finished" << std::endl;
	
	dataFileH.close();
	dataFileV.close();
}










void Sandman::sandHorizontalBender(
				   const float length,
				   const float width,
				   const float height,
				   const int numChannels,
				   const float waferThickness,
				   const float radius,
				   const float mval
				   )
{
  //This is a one-off, but malloc is expensive to use repetitively, so use
  //array of dedicated channel number floats
  const float nChannels = (float) numChannels;

  std::cout << color_yellow << "MULTI-CHANNEL BENDER" << color_reset << std::endl;
  
  if(nChannels < 1.0)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " attempt to use horizontal bender with < 1 channels" << std::endl;
      exit(1);
    }

  const float opticalWidth = (width / nChannels) - waferThickness*(nChannels - 1.0f);

  if(opticalWidth < 0.001)
    {
      std::cerr << color_red << "ERROR:" << color_reset << " optical width is less than 1 mm in horizontal bender module (value is " << opticalWidth << ")" << std::endl;
      exit(1);
    }

  std::cout << nChannels << " channel bender " << width << " wide and of length " << length << " from wafers of thickness " << waferThickness << std::endl;

  //First squeeze the neutrons into the channel
  sandSqueezeHorizontalBenderChannels(width, nChannels, waferThickness);

  //Propagate a normal curved guide with 20 cm long pieces
  sandCurvedGuide(length, 0.2f, opticalWidth, height, mval, radius);

  //UnSqueeze the neutrons out of the channel
  sandUnSqueezeHorizontalBenderChannels(width, nChannels, waferThickness);
  
  
}









void Sandman::sample(
		     const float width, 
		     const float height, 
		     const float win_width, 
		     const float win_height, 
		     const float hoffset, 
		     const float voffset, 
		     const float win_dist, 
		     const float lambdaMin, 
		     const float lambdaMax, 
		     const std::string& monitorNameStem)
{

  ///
  /// Generates the initial beam phase space from the given requirements.
  ///
  /// @param width width of sample in metres
  ///
  /// @param height height of sample in metres
  ///
  /// @param win_width the width of the beam at the exit of the guide in metres
  ///
  /// @param win_height the height of the beam at the exit of the guide in metres
  ///
  /// @param hoffset the horizontal offset of the sample relative to the beam
  /// axis (metres) positive is left as viewed from the guide exit --- this is
  /// almost certainly zero in most cases
  ///
  /// @param vertical offset of the sample relative to the beam axis (metres)
  /// positive being up --- this is almost certainly zero in most cases
  ///
  /// @param win_dist the distance from the guide exit to the sample position
  ///
  /// @param lambdaMin the minimum neutron wavelength needed at the sample position
  ///
  /// @param lambdaMax the maximum neutron wavelength needed at the sample position
  ///

   const float yMaxH = hoffset + 0.5*win_width;
   const float yMinH = hoffset - 0.5*win_width;

   const float yMaxV = voffset + 0.5*win_height;
   const float yMinV = voffset - 0.5*win_height;


   const float thetaMaxH = atan( (0.5*width + 0.5*win_width + hoffset) / win_dist);
   const float thetaMinH = atan( (-0.5*width - 0.5*win_width + hoffset) / win_dist);

   const float thetaMaxV = atan( (0.5*height + 0.5*win_height + voffset) / win_dist);
   const float thetaMinV = atan( (-0.5*height - 0.5*win_height + voffset) / win_dist);

   const float thetaMaxPrimeH = atan( (0.5*width - 0.5*win_width + hoffset) / win_dist);
   const float thetaMinPrimeH = atan( (-0.5*width + 0.5*win_width + hoffset) / win_dist);

   const float thetaMaxPrimeV = atan( (0.5*height - 0.5*win_height + voffset) / win_dist);
   const float thetaMinPrimeV = atan( (-0.5*height + 0.5*win_height + voffset) / win_dist);


   // The next part comes from
   // http://mathworld.wolfram.com/TrianglePointPicking.html 
   // v1 is along x
   // (theta) axis, v2 is up the right diagonal line

   const float oxH = thetaMinH;
   const float oyH = yMinH;
   const float v1xH = thetaMaxPrimeH - thetaMinH; // v1y is zero
   const float v2xH = thetaMaxH - thetaMaxPrimeH;
   const float v2yH = yMaxH - yMinH;

   const float oxV = thetaMinV;
   const float oyV = yMinV;
   const float v1xV = thetaMaxPrimeV - thetaMinV; // v1y is zero
   const float v2xV = thetaMaxV - thetaMaxPrimeV;
   const float v2yV = yMaxV - yMinV;

   //Normalisation of solid angle (NOTE: moderator units are per cm2!)
   const float a1  = 100.0f * 100.0f * width * height;
   const float a2  = 100.0f * 100.0f * win_width * win_height;
   const float deltaAdeltaO = a1 * a2 / (100.0f * 100.0f * win_dist*win_dist);


   std::cout << "\tSolid angle normalisation: " << deltaAdeltaO << std::endl;

   deltaLambdag = fabs(lambdaMax-lambdaMin);
   sourceDeltaLambda = deltaLambdag;
   if(deltaLambdag < 0.0001) // Zero wavelength band is going to screw up the
			     // maths.  Put in an artificial, small band
			     // hidden from the user
     deltaLambdag = 0.01;


   /// \todo Replace this maxElements with the memory-dependent check

   if(numElements > maxElements)
     {
       std::cerr << "\tMaximum number of elements exceeded." << std::endl;
       exit(1);
     }

   
   //Generate 1 array of random numbers for wavelength
   generateOneRandomArray();




   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

   if(showCUDAsteps)
     printf("CUDA kernel sample wavelength allocation with %d blocks of %d threads\n", blocksPerGrid,
	    threadsPerBlock);
   
   global_sandAllocateWavelength<<<blocksPerGrid, threadsPerBlock>>>
      (d_lambdag, d_r1g, lambdaMin, deltaLambdag, numElements);



   // printf("CUDA kernel sample Vertical wavelength allocation with %d blocks of %d threads\n", blocksPerGrid,
   // 	  threadsPerBlock);
   //  global_sandAllocateWavelength<<<blocksPerGrid, threadsPerBlock>>>
   //    (d_lambdaVg, d_r2g, lambdaMin, deltaLambdag, numElements);




    // Report to user the memory usage for the work
    size_t freeMemBytes, totalMemBytes;
    checkCudaErrors(cudaMemGetInfo( &freeMemBytes, &totalMemBytes)) ;
    
    int freeMem = (int)freeMemBytes ;
    int totalMem = (int)totalMemBytes ;
    int allocMem = totalMem - freeMem ;

    printf("\tGPU mem: alloc = %i MB, free = %i MB, tot = %i MB\n", allocMem/1024/1024, freeMem/1024/1024, totalMem/1024/1024);

    printf("\t-------------------------\n");
    printf("\tMemory used: %i percent\n", 100*allocMem/totalMem);
    printf("\t-------------------------\n");




   //Generate 2 arrays of random numbers
    generateBothRandomArrays();

    if(showCUDAsteps)
      printf("CUDA kernel sample Horizontal with %d blocks of %d threads\n", blocksPerGrid,
	     threadsPerBlock);
    
    global_sandSampleCUDA<<<blocksPerGrid, threadsPerBlock>>>
      (d_pointsYH, d_pointsThetaH, d_weightHg, d_r1g, d_r2g, oxH, oyH, v1xH, v2xH, v2yH, numElements);
    

   //Generate 2 new arrays of random numbers
    generateBothRandomArrays();

    if(showCUDAsteps)
      printf("CUDA kernel sample Vertical with %d blocks of %d threads\n", blocksPerGrid,
	     threadsPerBlock);
    
    global_sandSampleCUDA<<<blocksPerGrid, threadsPerBlock>>>
      (d_pointsYV, d_pointsThetaV, d_weightVg, d_r1g, d_r2g, oxV, oyV, v1xV, v2xV, v2yV, numElements);


    // Initialise trajectory brightness with the solid angle calculation
    if(showCUDAsteps)
      std::cout << "CUDA kernel initArray on moderator flux with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
    global_initArray<<<blocksPerGrid, threadsPerBlock>>>
      (d_modFlux, deltaAdeltaO, numElements);


    //Perhaps the user wants to get a phase space map at the sample position.
    //If that is the case, we need to do a little back and forth fudgery

    if(!monitorNameStem.empty())
      {
	//Here we go backwards to the sample plane, snapshot a monitor, then
	//go forwards back to the guide entrance, and take an insignificant
	//rounding error hit
	
	std::cout << "\tGenerating snapshot for sample position monitor" << std::endl;
	
	//Go backwards
	sandFreeSpaceCUDA(-win_dist);
	
	//Create phase space snapshots
	phaseSpaceMapH( 
		       (monitorNameStem + "Horizontal2D.dat").c_str(), 
		       yMinH, 
		       yMaxH,
		       thetaMinH,
		       thetaMaxH
			);

	//Go forwards again before continuing with the rest
	sandFreeSpaceCUDA(win_dist);
      }
	
}













//////////////////////////////////////////
//                                      //
//       Private Functions              //
//                                      //
//       and                            //
//                                      //
//       kernels                        //
//                                      //
//////////////////////////////////////////


///
/// Unit test setup function to seed the Y values
///
/// @param ypoints pointer to host memory that needs to be copied over
///

void Sandman::unitTestInitPhaseSpace(const float *ypoints, const float *pointsTheta, const float *weight)
{
  //Copy to device (lets use horizontal plane) to overwrite 
  checkCudaErrors(cudaMemcpy(d_pointsYH, ypoints, 32*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pointsThetaH, pointsTheta, 32*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_weightHg, weight, 32*sizeof(float), cudaMemcpyHostToDevice));
  
}



void Sandman::unitTestGetPhaseSpace(float *ypoints, float *pointsTheta, float *weight)
{
  //Copy to device (lets use horizontal plane) to overwrite 
  checkCudaErrors(cudaMemcpy(ypoints, d_pointsYH, 32*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(pointsTheta, d_pointsThetaH, 32*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(pointsTheta, d_weightHg, 32*sizeof(float), cudaMemcpyDeviceToHost));
}



void Sandman::displayWelcome(void)
{
  ///
  /// Presents welcome message when called by constructor.
  ///

  std::cout << "****************************************" << std::endl;
  std::cout << "*                                      *" << std::endl;
  std::cout << "*               " << color_red << "SANDMAN" << color_reset << "                *" << std::endl;
  std::cout << "*                                      *" << std::endl;
  std::cout << "*   Implementation of SAND in C++:     *" << std::endl;
  std::cout << "*   Neutron beam transport on GPU      *" << std::endl;
  std::cout << "*                                      *" << std::endl;
  std::cout << "*   " << color_yellow << "phil.m.bentley@gmail.com" << color_reset << " 2016      *" << std::endl;
  std::cout << "*                                      *" << std::endl;
  std::cout << "*   Released under BSD license         *" << std::endl;
  std::cout << "*                                      *" << std::endl;
  std::cout << "****************************************" << std::endl;
}


void Sandman::generateRandomArray(float *array)
{
  ///
  /// Presents welcome message when called by constructor.
  ///

  #ifdef DEBUG
  cudaError_t errSync  = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess) 
    std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
  if (errAsync != cudaSuccess)
    std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif
  

  printf("\tGenerating random numbers on GPU\n");
  checkCudaErrors(curandGenerateUniform(prngGPU, (float *) array, numElements));

#ifdef DEBUG
  if (errSync != cudaSuccess) 
    std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
  if (errAsync != cudaSuccess)
    std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif
}


void Sandman::zeroHistogram1D(void)
{
  printf("\tZeroing 1D histogram\n");
  
  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(100 + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    std::cout << "CUDA kernel zero 1d histogram[" << 100 << "] with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;
  
  //void global_sandZeroHistogram1D(float *d_histogram, const int numElements)
  
  global_sandZeroHistogram1D<<<blocksPerGrid, threadsPerBlock>>>
    (d_histogram1D);
#ifdef DEBUG
  cudaError_t errSync  = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess) 
    std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
  if (errAsync != cudaSuccess)
    std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif
}


void Sandman::zeroHistogram2D(void)
{
  printf("\tZeroing 2D histogram.\n");

   int threadsPerBlock = SANDMAN_CUDA_THREADS;
   int blocksPerGrid =(100*100 + threadsPerBlock - 1) / threadsPerBlock;
   if(showCUDAsteps)
     printf("CUDA kernel zeroHistogram2D with %d blocks of %d threads\n", blocksPerGrid,
	    threadsPerBlock);
   
   //void global_sandZeroHistogram1D(float *d_histogram, const int numElements)
   
    global_sandZeroHistogram2D<<<blocksPerGrid, threadsPerBlock>>>
      ((float (*)[100])d_histogram2D);
}



float Sandman::arrayMinimum(const float *d_array, float *d_answer)
{
  float h_answer[1];
  
  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel arrayMin %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);


  // // Zero the count on the host
  // h_answer[0] = 10000.0f;

  // // Copy the zero total to device memory
  // checkCudaErrors(cudaMemcpy(d_answer, h_answer, sizeof(float), cudaMemcpyHostToDevice));
  
#ifdef DEBUG
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
      std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
    if (errAsync != cudaSuccess)
      std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif

  
  printf("\tCounting up phase space\n");

  //void global_countNeutrons(float *numNeutrons, const float *weight, const int numElements)
   
  global_arrayMinimum<<<blocksPerGrid, threadsPerBlock>>>
    (d_array, d_answer, numElements);

#ifdef DEBUG
    if (errSync != cudaSuccess) 
      std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
    if (errAsync != cudaSuccess)
      std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif

  //Copy answer out of device memory for host reporting
  checkCudaErrors(cudaMemcpy(h_answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    if (errSync != cudaSuccess) 
      std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
    if (errAsync != cudaSuccess)
      std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
#endif
  
  
  printf("Got %f minimum\n", h_answer[0]);
  return(h_answer[0]);
}



float Sandman::arrayMaximum(const float *d_array, float *d_answer)
{
  float h_answer[1];  //for debugging
  
  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel arrayMax %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);


  // // Zero the count on the host
  // h_answer[0] = -10000.0f;

  // // Copy the zero total to device memory
  // checkCudaErrors(cudaMemcpy(d_answer, h_answer, sizeof(float), cudaMemcpyHostToDevice));
  
  

   
  global_arrayMaximum<<<blocksPerGrid, threadsPerBlock>>>
    (d_array, d_answer, numElements);

  //Copy total out of device memory for host reporting
  checkCudaErrors(cudaMemcpy(h_answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost));
  
  
  printf("Got %f maximum\n", h_answer[0]);
  return(h_answer[0]);
}



void Sandman::sandGetPhaseSpaceH(float *h_pointsY, float *h_pointsTheta, float *h_weight)
{
  
    // Copy the data off the card to make sure it makes sense back at the host
    checkCudaErrors(cudaMemcpy(h_pointsY, d_pointsYH, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_pointsTheta, d_pointsThetaH, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_weight, d_weightHg, numElements * sizeof(float), cudaMemcpyDeviceToHost));
}


void Sandman::sandGetPhaseSpaceV(float *h_pointsY, float *h_pointsTheta, float *h_weight)
{
  
    // Copy the data off the card to make sure it makes sense back at the host
    checkCudaErrors(cudaMemcpy(h_pointsY, d_pointsYV, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_pointsTheta, d_pointsThetaV, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_weight, d_weightVg, numElements * sizeof(float), cudaMemcpyDeviceToHost));
}




void Sandman::sandDebugPosPos(float *h_pointsH, float *h_weightH, float *h_pointsV, float *h_weightV)
{
  
    // Copy the data off the card to make sure it makes sense back at the host
    checkCudaErrors(cudaMemcpy(h_pointsH, d_pointsYH, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_weightH, d_weightHg, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_pointsV, d_pointsYV, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_weightV, d_weightVg, numElements * sizeof(float), cudaMemcpyDeviceToHost));
}


///Calculates which channel number (left to right) the neutron sits in, then
///shifts all phase space to fit in a single version of that channel so that
///the curved guide module can be used to do the transport for a multi-channel
///bender.  Then the opposite function "unSqueeze..." reverses this process
void Sandman::sandSqueezeHorizontalBenderChannels(const float width, const float numChannels, const float waferThickness)
{
  //Channel width in this case includes one wafer on the far side
  float channelWidth = (width + waferThickness) / numChannels; //(this calc has last channel wafer inside the guide substrate mathematically)
 
  //Device now computes
  //relativeY = ypos[i] + width/2.0;
  //channelNumber = roundf( relativeY / channelwidth );
  //That is stored in tempArray
  
  //Then we adjust the position to be within a single channel of the right thickness for the OPTICS
  //ypos[i] = ypos[i] + width/2.0f;
  //ypos[i] = ypos[i] / channelNumber;
  //ypos[i] = ypos[i] - 0.5f * (channelWidth-waferThickness);

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel squeeze h bender with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);


  global_squeezeBenderChannel<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsYH, d_tempArray, width, channelWidth, waferThickness, numElements);
  
  
}


///"Squeeze...() calculates which channel number (left to right) the neutron
///sits in, then shifts all phase space to fit in a single version of that
///channel so that the curved guide module can be used to do the transport for
///a multi-channel bender.  This function "unSqueeze..." reverses this process
///after the bender has been done
void Sandman::sandUnSqueezeHorizontalBenderChannels(const float width, const float numChannels, const float waferThickness)
{
  //Channel width in this case includes one wafer on the far side
  float channelWidth = (width + waferThickness) / numChannels; //(this calc has last channel wafer inside the guide substrate mathematically)
  
  //Device reverses the position adjustment of sandSqueeze...
  //ypos[i] = ypos[i] + 0.5f * (channelWidth - waferThickness);
  //ypos[i] = ypos[i] * channelNumber;
  //ypos[i] = ypos[i] - width/2.0f;

  int threadsPerBlock = SANDMAN_CUDA_THREADS;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  if(showCUDAsteps)
    printf("CUDA kernel unsqueeze h bender with %d blocks of %d threads\n", blocksPerGrid,
	   threadsPerBlock);


  global_unSqueezeBenderChannel<<<blocksPerGrid, threadsPerBlock>>>
    (d_pointsYH, d_tempArray, width, channelWidth, waferThickness, numElements);
  
}

