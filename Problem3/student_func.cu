/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"  //This header is only needed to satisfy the indexing in Eclipse...
#include <iostream>

__global__ void findMax(float* d_array, const size_t numRows,
                                  const size_t numCols, float* d_blockMaxes){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int arraySize=numRows*numCols;

	for (int stride=blockDim.x/2; stride >0; stride /= 2){
		if( gid+stride < arraySize)
			if(d_array[gid] < d_array[gid+stride])
				d_array[gid] = d_array[gid+stride];
		__syncthreads();
	}
	if(tid==0)
		d_blockMaxes[blockIdx.x] = d_array[gid];
}

__global__ void findMin(float* d_array, const size_t numRows,
                                  const size_t numCols, float* d_blockMins){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int arraySize=numRows*numCols;

	for (int stride=blockDim.x/2; stride > 0; stride /= 2){
		if( gid + stride < arraySize)
			if(d_array[gid] > d_array[gid+stride])
				d_array[gid] = d_array[gid+stride];
		__syncthreads();
	}
	if(tid==0)
		d_blockMins[blockIdx.x] = d_array[gid];
}

__global__ void calcHisto(const float* const d_array, size_t arraySize,
		                  int* d_histoOut,
		                  float lumMin, float lumRange, int numBins ){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;

	if (gid<arraySize){
		int binNum = (d_array[gid] - lumMin) / lumRange * numBins;

		if (binNum == numBins)
			--binNum;

		atomicAdd(&(d_histoOut[binNum]), 1);
	}
}

__global__ void calcScan(int* d_histogram, unsigned int* d_cdf, size_t histoSize){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;

	if(gid>=histoSize){
		return;
	}
	if(gid>0 && gid < histoSize)
		d_cdf[gid]=d_histogram[gid-1];
	else if (gid==0){
		d_cdf[0]=0;
	}
	__syncthreads();

    int addVal;
    for(int stride = 1; stride < histoSize; stride*=2){
    //for(int stride = 1; stride <= 1; stride*=2){
		addVal=0;
		if ( gid-stride >=0 ){
			addVal=d_cdf[gid-stride];
		//	if(gid==2)
		//		printf("\nBeleptem az if-be");
			//addVal = d_cdf[gid] + d_cdf[gid-stride];
		}
		//else{
		//	addVal = d_cdf[gid];
		//}
		//if(gid==2){
		//	printf("\ngid=2; addVal=%d; d_cdf1=%d d_cdf2=%d", addVal, d_cdf[1], d_cdf[2]);
		//}
		__syncthreads();
		d_cdf[gid] += addVal;
		//d_cdf[gid] = addVal;
		__syncthreads();
		//if (gid==2)
		//	printf("\nupdate utan d_cdf[2]=%d", d_cdf[2]);
	}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  //Here are the steps you need to implement
  //  1) find the minimum and maximum value in the input logLuminance channel
  //     store in min_logLum and max_logLum

	dim3 blockSize{512};
	dim3 gridSize{ numRows*numCols/blockSize.x +1 };
	float* d_tmpLogLuminance;
	const size_t imageSizeBytes = numRows*numCols*sizeof(float);
	checkCudaErrors( cudaMalloc( &d_tmpLogLuminance, imageSizeBytes ) );
	checkCudaErrors( cudaMemcpy( d_tmpLogLuminance, d_logLuminance, imageSizeBytes, cudaMemcpyDeviceToDevice) );

	float* d_blockMaxes;
	float* d_blockMins;
	checkCudaErrors( cudaMalloc( &d_blockMaxes, gridSize.x * sizeof(float)) );
	checkCudaErrors( cudaMalloc( &d_blockMins,  gridSize.x * sizeof(float)) );

	findMax <<<gridSize, blockSize>>> (d_tmpLogLuminance, numRows, numCols, d_blockMaxes);
	cudaDeviceSynchronize();
	float* tmpBlockVals = new float[gridSize.x];
	checkCudaErrors( cudaMemcpy( tmpBlockVals, d_blockMaxes, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost) );

	max_logLum=tmpBlockVals[0];
	for(size_t i=0; i<gridSize.x; ++i){
		if(tmpBlockVals[i] > max_logLum)
			max_logLum = tmpBlockVals[i];
	}

	checkCudaErrors( cudaMemcpy( d_tmpLogLuminance, d_logLuminance, imageSizeBytes, cudaMemcpyDeviceToDevice) );
	findMin <<<gridSize, blockSize>>> (d_tmpLogLuminance, numRows, numCols, d_blockMins);
	cudaDeviceSynchronize();
	checkCudaErrors( cudaMemcpy( tmpBlockVals, d_blockMins, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost) );
	min_logLum=tmpBlockVals[0];
	for(size_t i=0; i<gridSize.x; ++i){
		if(tmpBlockVals[i] < min_logLum)
			min_logLum = tmpBlockVals[i];
	}

	//DebugCode OK
//		float* logLuminance = new float[numRows*numCols];
//		checkCudaErrors( cudaMemcpy( logLuminance, d_logLuminance, numRows*numCols * sizeof(float), cudaMemcpyDeviceToHost) );
//		    //Search the max and min
//		float maxVal=logLuminance[0];
//		float minVal=logLuminance[0];
//
//		for( int idx=1; idx<numRows*numCols; ++idx){
//			if(logLuminance[idx] < minVal)
//				minVal=logLuminance[idx];
//
//			if(logLuminance[idx] > maxVal)
//				maxVal=logLuminance[idx];
//		}
//
//		if(max_logLum == maxVal)
//			std::cout << "\nMaximum is correct! Value: " << max_logLum;
//		else
//			std::cout <<"\nMaximum is not OK! GPU result: " << max_logLum << " CPU result: " << maxVal;
//
//		if(min_logLum == minVal)
//			std::cout << "\nMinimum is correct! Value: " << min_logLum;
//		else
//			std::cout <<"\nMinimum is not OK!! GPU result: " << min_logLum << " CPU result: " << minVal;



  //2) subtract them to find the range
    float lumRange = max_logLum - min_logLum;


  //3) generate a histogram of all the values in the logLuminance channel using
  //     the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    int* d_histogram;
    checkCudaErrors( cudaMalloc( &d_histogram, numBins * sizeof(int)) );
    checkCudaErrors( cudaMemset( d_histogram, 0, numBins * sizeof(int)) );

    calcHisto<<<gridSize, blockSize>>> (d_logLuminance, numRows*numCols, d_histogram,
    		                            min_logLum, lumRange, numBins );
	cudaDeviceSynchronize();


    //DEBUG OK
//    int* debugHisto = new int[numBins];
//    checkCudaErrors( cudaMemcpy(debugHisto, d_histogram, numBins*sizeof(int), cudaMemcpyDeviceToHost) );
//    float* debugImage = new float[numRows*numCols];
//    checkCudaErrors( cudaMemcpy(debugImage, d_logLuminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost) );
//    int* serialHisto = new int[numBins];
//
//    for (int i=0; i<numBins; ++i)
//    	serialHisto[i] = 0;
//    int binNum;
//    for (int i=0; i<numRows*numCols; ++i){
//    	binNum = (debugImage[i] - min_logLum) / lumRange * numBins;
//    	if(binNum<numBins)
//    		serialHisto[binNum]++;
//    	else
//    		serialHisto[numBins-1]++;
//    }
//    //for (int i=0; i<numBins; ++i)
//    //   	std::cout << '\n' << i <<". bin; serial: " << serialHisto[i] << " parallel: " << debugHisto[i];
//    for (int i=0; i<numBins; ++i)
//        	if (serialHisto[i] != debugHisto[i])
//        		std::cout << "\n Histo not equal at " << i << " serialHisto= " << serialHisto[i] << " debugHisto= " << debugHisto[i];

    /*  4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    blockSize.x=1024;
    gridSize.x=numBins/blockSize.x ;
    //std::cout << "\ngridSize: " << gridSize.x << "  numBins:" << numBins << '\n';

    calcScan <<<gridSize, blockSize>>>(d_histogram, d_cdf, numBins);
	cudaDeviceSynchronize();

    //DEBUG OK
//    //int* debugHisto = new int[numBins];
//    checkCudaErrors( cudaMemcpy(debugHisto, d_histogram, numBins*sizeof(int), cudaMemcpyDeviceToHost) );
//    int* serialCdf = new int[numBins];
//
//    int acc=0;
//    for(int i=0; i<numBins; ++i){
//    	serialCdf[i] = acc;
//    	acc += debugHisto[i];
//    }
//
//    int* debugCdf = new int[numBins];
//    checkCudaErrors( cudaMemcpy(debugCdf, d_cdf, numBins*sizeof(int), cudaMemcpyDeviceToHost) );
//
//    for (int i=0; i<numBins; ++i)
//    	if (serialCdf[i] != debugCdf[i])
//    		std::cout << "\n Cdf not equal at " << i << " serialCdf= " << serialCdf[i] << " debugCdf= " << debugCdf[i];
//    //  	std::cout << '\n' << i <<". bin; histogram: " << debugHisto[i] << " serial: " << serialCdf[i] << " parallel: " << debugCdf[i];
//    	else{
//    		std::cout << "\n Cdf equal at " << i << " serialCdf= " << serialCdf[i];
//    	}

	checkCudaErrors( cudaFree( d_tmpLogLuminance ) );
	checkCudaErrors( cudaFree( d_histogram ) );
	checkCudaErrors( cudaFree( d_blockMaxes ) );
	checkCudaErrors( cudaFree( d_blockMins ) );
}
