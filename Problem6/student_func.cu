//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]

    In this assignment we will do 800 iterations.
*/

#include "utils.h"
#include "loadSaveImage.h"
#include <thrust/host_vector.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__
bool isMasked(const uchar4* const d_sourceImg, size_t serialIndex){
	/**
	 * Returns true if point part of sourceImg
	 */
	return (d_sourceImg[serialIndex].x != 255) || (d_sourceImg[serialIndex].y != 255) || (d_sourceImg[serialIndex].z != 255) ;
}

__global__
void checkInteriorAndBorder(const uchar4* const d_sourceImg,
		                    const size_t numRowsSource, const size_t numColsSource,
		                    int* d_isInterior,
							int* d_isBorder){
	size_t gidX = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gidY = blockIdx.y * blockDim.y + threadIdx.y;

	//boundary check
	if( (gidX >= numColsSource) || (gidY >= numRowsSource))
		return;

	size_t serialIndex = gidY * numColsSource + gidX;

	//Check mask
	if( isMasked(d_sourceImg, serialIndex) ){
		//The source image has to have a frame -> We do not check if gidX==0 || gidY == 0 || gidX == numColsSource etc...
		int neighborIndex;
		//Check the top neighbor
		neighborIndex = (gidY-1) * numColsSource + gidX;
		if( !isMasked(d_sourceImg, neighborIndex) ){
			d_isInterior[serialIndex]=0;
		    d_isBorder[serialIndex]=1;
		    return;
		}
		//Check the bottom neighbor
		neighborIndex = (gidY+1) * numColsSource + gidX;
		if( !isMasked(d_sourceImg, neighborIndex) ){
			d_isInterior[serialIndex]=0;
			d_isBorder[serialIndex]=1;
			return;
		}
		//Check the left neighbor
		neighborIndex = gidY * numColsSource + gidX-1;
		if( !isMasked(d_sourceImg, neighborIndex) ){
			d_isInterior[serialIndex]=0;
		    d_isBorder[serialIndex]=1;
			return;
		}
		//Check the left neighbor
		neighborIndex = gidY * numColsSource + gidX+1;
		if( !isMasked(d_sourceImg, neighborIndex) ){
			d_isInterior[serialIndex]=0;
			d_isBorder[serialIndex]=1;
			return;
		}

		//IF not border -> interior
		d_isInterior[serialIndex]=1;
		d_isBorder[serialIndex]=0;
		return;
	}
	else{ //Out of the mask
		d_isInterior[serialIndex]=0;
		d_isBorder[serialIndex]=0;
	}

}

__global__
void separateChannels(const uchar4* const d_img,
		const size_t numRowsSource, const size_t numColsSource,
		int* d_chRed,
		int* d_chGreen,
		int* d_chBlue){

	size_t gid = blockDim.x*blockIdx.x + threadIdx.x;

	if(gid >= numRowsSource * numColsSource){
		return;
	}

	d_chRed[gid]=d_img[gid].x;
	d_chGreen[gid]=d_img[gid].y;
	d_chBlue[gid]=d_img[gid].z;
}

__global__
void initBuffers(const uchar4* const d_img,
		const size_t numRowsSource, const size_t numColsSource,
		float* d_bufferRed,
		float* d_bufferGreen,
		float* d_bufferBlue){

	size_t gid = blockDim.x*blockIdx.x + threadIdx.x;

	if(gid >= numRowsSource * numColsSource){
		return;
	}

	d_bufferRed[gid]=d_img[gid].x;
	d_bufferGreen[gid]=d_img[gid].y;
	d_bufferBlue[gid]=d_img[gid].z;
}

__global__
void doJacobiIteration(int *d_isInterior, int *d_isBorder,
		const size_t numRowsSource, const size_t numColsSource,
		float* bufferIn, float* bufferOut,
		int *d_sourceImg, int *d_destImg){
	size_t gidX = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gidY = blockIdx.y * blockDim.y + threadIdx.y;

	//boundary check
	if( (gidX >= numColsSource) || (gidY >= numRowsSource))
		return;

	size_t serialIndex = gidY * numColsSource + gidX;

	//Check if pixel is interior
	if(d_isInterior[serialIndex] == 0){
		return;
	}

	size_t topIndex = (gidY-1) * numColsSource + gidX;
	size_t bottomIndex = (gidY+1) * numColsSource + gidX;
	size_t leftIndex = gidY * numColsSource + gidX - 1;
	size_t rightIndex = gidY * numColsSource + gidX + 1;

	float borderSum=0.0;
	float blendedSum=0.0;


	if(d_isInterior[topIndex]==1){
		blendedSum += bufferIn[topIndex];
	}
	else{
		borderSum += d_destImg[topIndex];
	}

	if(d_isInterior[bottomIndex]==1){
		blendedSum += bufferIn[bottomIndex];
	}
	else{
		borderSum += d_destImg[bottomIndex];
	}

	if(d_isInterior[leftIndex]==1){
		blendedSum += bufferIn[leftIndex];
	}
	else{
		borderSum += d_destImg[leftIndex];
	}

	if(d_isInterior[rightIndex]==1){
		blendedSum += bufferIn[rightIndex];
	}
	else{
		borderSum += d_destImg[rightIndex];
	}

	float sum2= 4.f * (float)d_sourceImg[serialIndex];
    sum2 -= (float)d_sourceImg[topIndex] + (float)d_sourceImg[bottomIndex];
    sum2 -= (float)d_sourceImg[leftIndex] + (float)d_sourceImg[rightIndex];

	float newVal = (blendedSum + borderSum + sum2)/4.f;

	bufferOut[serialIndex] = min (255.f, max(0.f, newVal));


}

__global__
void combineChannels(const size_t numRowsSource, const size_t numColsSource, int* d_isInterior,
		float *redCh, float *greenCh, float *blueCh,
		uchar4 *outImage){
	size_t gid = blockDim.x*blockIdx.x + threadIdx.x;

	if(gid >= numRowsSource * numColsSource){
		return;
	}

	if(d_isInterior[gid]==1){
		outImage[gid].x=(unsigned char)redCh[gid];
		outImage[gid].y=(unsigned char)greenCh[gid];
		outImage[gid].z=(unsigned char)blueCh[gid];
	}

}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
//     0) Copy data to the device, allocate memory for the variables
	   uchar4 *d_sourceImg;
	   checkCudaErrors( cudaMalloc((void**)&d_sourceImg, numRowsSource*numColsSource*sizeof(uchar4)) );
	   checkCudaErrors( cudaMemcpy(d_sourceImg, h_sourceImg, numRowsSource*numColsSource*sizeof(uchar4), cudaMemcpyHostToDevice) );

	   uchar4 *d_destImg;
	   checkCudaErrors( cudaMalloc((void**)&d_destImg, numRowsSource*numColsSource*sizeof(uchar4)) );
	   checkCudaErrors( cudaMemcpy(d_destImg, h_destImg, numRowsSource*numColsSource*sizeof(uchar4), cudaMemcpyHostToDevice) );


	   int *d_isInterior, *d_isBorder;
	   checkCudaErrors( cudaMalloc((void**)&d_isInterior, numRowsSource*numColsSource*sizeof(int)) );
	   checkCudaErrors( cudaMalloc((void**)&d_isBorder, numRowsSource*numColsSource*sizeof(int)) );

	   int *d_redChSource, *d_greenChSource, *d_blueChSource;
	   checkCudaErrors( cudaMalloc((void**)&d_redChSource, numRowsSource*numColsSource*sizeof(int)) );
	   checkCudaErrors( cudaMalloc((void**)&d_greenChSource, numRowsSource*numColsSource*sizeof(int)) );
	   checkCudaErrors( cudaMalloc((void**)&d_blueChSource, numRowsSource*numColsSource*sizeof(int)) );

	   int *d_redChDest, *d_greenChDest, *d_blueChDest;
	   checkCudaErrors( cudaMalloc((void**)&d_redChDest, numRowsSource*numColsSource*sizeof(int)) );
	   checkCudaErrors( cudaMalloc((void**)&d_greenChDest, numRowsSource*numColsSource*sizeof(int)) );
	   checkCudaErrors( cudaMalloc((void**)&d_blueChDest, numRowsSource*numColsSource*sizeof(int)) );

	   dim3 blockSize(32,32);
	   dim3 gridSize(numColsSource/blockSize.x+1, numRowsSource/blockSize.y+1);

//     1) Compute a mask of the pixels from the source image to be copied
//        The pixels that shouldn't be copied are completely white, they
//        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

//     2) Compute the interior and border regions of the mask.  An interior
//        pixel has all 4 neighbors also inside the mask.  A border pixel is
//        in the mask itself, but has at least one neighbor that isn't.

	   checkInteriorAndBorder <<<gridSize, blockSize >>>(d_sourceImg,
	                                                     numRowsSource, numColsSource,
	                                                     d_isInterior,
	  				                                     d_isBorder);
	   cudaDeviceSynchronize();
	   checkCudaErrors(cudaGetLastError());

//	   //Check interior and border calculation
//	   int* h_isInterior = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   int* h_isBorder = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_isInterior, d_isInterior, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   checkCudaErrors( cudaMemcpy(h_isBorder, d_isBorder, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//
//	   saveIntImageBV(h_isInterior, 255,
//			   numRowsSource, numColsSource,
//	   				"debugImages/debugIsInterior.png");
//
//	   saveIntImageBV(h_isBorder, 255,
//	   			   numRowsSource, numColsSource,
//	   	   				"debugImages/debugIsBorder.png");

//     3) Separate out the incoming image into three separate channels

	   dim3 blockSize_separate(32);
	   dim3 gridSize_separate( numColsSource*numRowsSource / blockSize_separate.x +1 );

	   separateChannels<<<gridSize_separate,  blockSize_separate >>>(d_sourceImg,
	   		                 numRowsSource, numColsSource,
	   		                 d_redChSource,
	   		                 d_greenChSource,
	   		                 d_blueChSource);

	   separateChannels<<<gridSize_separate,  blockSize_separate >>>(d_destImg,
	   	   		                 numRowsSource, numColsSource,
	   	   		                 d_redChDest,
	   	   		                 d_greenChDest,
	   	   		                 d_blueChDest);
	   cudaDeviceSynchronize();
	   checkCudaErrors(cudaGetLastError());

	   //Check color channel separation
//	   int* h_redChSource = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_redChSource, d_redChSource, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   saveIntImageBV(h_redChSource, 1, numRowsSource, numColsSource, "debugImages/h_redChSource.png");
//	   int* h_greenChSource = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_greenChSource, d_greenChSource, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   saveIntImageBV(h_greenChSource, 1, numRowsSource, numColsSource, "debugImages/h_greenChSource.png");
//	   int* h_blueChSource = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_blueChSource, d_blueChSource, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   saveIntImageBV(h_blueChSource, 1, numRowsSource, numColsSource, "debugImages/h_blueChSource.png");
//
//	   int* h_redChDest = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_redChDest, d_redChDest, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   saveIntImageBV(h_redChDest, 1, numRowsSource, numColsSource, "debugImages/h_redChDest.png");
//	   int* h_greenChDest = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_greenChDest, d_greenChDest, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   saveIntImageBV(h_greenChDest, 1, numRowsSource, numColsSource, "debugImages/h_greenChDest.png");
//	   int* h_blueChDest = (int*)malloc(numRowsSource*numColsSource*sizeof(int));
//	   checkCudaErrors( cudaMemcpy(h_blueChDest, d_blueChDest, numRowsSource*numColsSource*sizeof(int), cudaMemcpyDeviceToHost) );
//	   saveIntImageBV(h_blueChDest, 1, numRowsSource, numColsSource, "debugImages/h_blueChDest.png");

//     4) Create two float(!) buffers for each color channel that will
//        act as our guesses.  Initialize them to the respective color
//        channel of the source image since that will act as our intial guess.

	   float *d_bufferRed, *d_bufferGreen, *d_bufferBlue;
	   checkCudaErrors( cudaMalloc((void**)&d_bufferRed, 2*numRowsSource*numColsSource*sizeof(float)) ); //Allocate memory for TWO(!) buffers!!!
	   checkCudaErrors( cudaMalloc((void**)&d_bufferGreen, 2*numRowsSource*numColsSource*sizeof(float)) ); //Allocate memory for TWO(!) buffers!!!
	   checkCudaErrors( cudaMalloc((void**)&d_bufferBlue,  2*numRowsSource*numColsSource*sizeof(float)) ); //Allocate memory for TWO(!) buffers!!!

	   initBuffers<<<gridSize_separate,  blockSize_separate >>>(d_sourceImg,
	   	   	   		                 numRowsSource, numColsSource,
	   	   	   		                 d_bufferRed,
	   	   	   		                 d_bufferGreen,
	   	   	   		                 d_bufferBlue);
	   cudaDeviceSynchronize();
	   checkCudaErrors(cudaGetLastError());

	   //Check the initialized buffers
//	   float* h_bufferRed = (float*)malloc(numRowsSource*numColsSource*sizeof(float));
//	   checkCudaErrors( cudaMemcpy(h_bufferRed, d_bufferRed, numRowsSource*numColsSource*sizeof(float), cudaMemcpyDeviceToHost) );
//	   saveFloatImageBV(h_bufferRed, 1, numRowsSource, numColsSource, "debugImages/h_bufferRed.png");
//
//	   float* h_bufferGreen = (float*)malloc(numRowsSource*numColsSource*sizeof(float));
//	   checkCudaErrors( cudaMemcpy(h_bufferGreen, d_bufferGreen, numRowsSource*numColsSource*sizeof(float), cudaMemcpyDeviceToHost) );
//	   saveFloatImageBV(h_bufferGreen, 1, numRowsSource, numColsSource, "debugImages/h_bufferGreen.png");
//
//	   float* h_bufferBlue = (float*)malloc(numRowsSource*numColsSource*sizeof(float));
//	   checkCudaErrors( cudaMemcpy(h_bufferBlue, d_bufferBlue, numRowsSource*numColsSource*sizeof(float), cudaMemcpyDeviceToHost) );
//	   saveFloatImageBV(h_bufferBlue, 1, numRowsSource, numColsSource, "debugImages/h_bufferBlue.png");

//     5) For each color channel perform the Jacobi iteration described
//        above 800 times.
	   int bufferInOffset, bufferOutOffset;
	   for(int i=0; i<800; ++i){
		   if (i%2 == 0){
			   bufferInOffset = 0;
			   bufferOutOffset = numRowsSource * numColsSource;
		   }
		   else{
			   bufferInOffset = numRowsSource * numColsSource;
			   bufferOutOffset = 0;
		   }

		   doJacobiIteration<<< blockSize, gridSize >>>(d_isInterior, d_isBorder,
		   		numRowsSource, numColsSource,
		   		&d_bufferRed[bufferInOffset], &d_bufferRed[bufferOutOffset],
		   		d_redChSource, d_redChDest);
		   doJacobiIteration<<< blockSize, gridSize >>>(d_isInterior, d_isBorder,
		   		   		numRowsSource, numColsSource,
		   		   		&d_bufferGreen[bufferInOffset], &d_bufferGreen[bufferOutOffset],
		   		   		d_greenChSource, d_greenChDest);
		   doJacobiIteration<<< blockSize, gridSize >>>(d_isInterior, d_isBorder,
		   		   		numRowsSource, numColsSource,
		   		   		&d_bufferBlue[bufferInOffset], &d_bufferBlue[bufferOutOffset],
		   		   		d_blueChSource, d_blueChDest);

		   //Check during the iteration
//		   if (i==799){
//			   checkCudaErrors( cudaMemcpy(h_bufferRed, &d_bufferRed[bufferOutOffset], numRowsSource*numColsSource*sizeof(float), cudaMemcpyDeviceToHost) );
//			   saveFloatImageBV(h_bufferRed, 1, numRowsSource, numColsSource, "debugImages/h_bufferRed_jacobi.png");
//			   checkCudaErrors( cudaMemcpy(h_bufferGreen, &d_bufferGreen[bufferOutOffset], numRowsSource*numColsSource*sizeof(float), cudaMemcpyDeviceToHost) );
//			   saveFloatImageBV(h_bufferGreen, 1, numRowsSource, numColsSource, "debugImages/h_bufferGreen_jacobi.png");
//			   checkCudaErrors( cudaMemcpy(h_bufferBlue, &d_bufferBlue[bufferOutOffset], numRowsSource*numColsSource*sizeof(float), cudaMemcpyDeviceToHost) );
//			   saveFloatImageBV(h_bufferBlue, 1, numRowsSource, numColsSource, "debugImages/h_bufferBlue_jacobi.png");
//		   }
	   }
	   cudaDeviceSynchronize();
	   checkCudaErrors(cudaGetLastError());

//	   6) Create the output image by replacing all the interior pixels
//	      in the destination image with the result of the Jacobi iterations.
//	      Just cast the floating point values to unsigned chars since we have
//	      already made sure to clamp them to the correct range.

	   combineChannels <<<gridSize_separate,  blockSize_separate >>>(numRowsSource, numColsSource, d_isInterior,
			   &d_bufferRed[bufferOutOffset], &d_bufferGreen[bufferOutOffset], &d_bufferBlue[bufferOutOffset],
	   		d_destImg);
	   cudaDeviceSynchronize();
	   checkCudaErrors(cudaGetLastError());

	   checkCudaErrors( cudaMemcpy(h_blendedImg, d_destImg, numRowsSource*numColsSource*sizeof(uchar4), cudaMemcpyDeviceToHost) );

	   checkCudaErrors( cudaFree(d_sourceImg) );
	   checkCudaErrors( cudaFree(d_destImg) );

	   checkCudaErrors( cudaFree(d_isBorder) );
	   checkCudaErrors( cudaFree(d_isInterior) );

	   checkCudaErrors( cudaFree(d_redChSource) );
	   checkCudaErrors( cudaFree(d_greenChSource) );
	   checkCudaErrors( cudaFree(d_blueChSource ) );

	   checkCudaErrors( cudaFree(d_redChDest) );
	   checkCudaErrors( cudaFree(d_greenChDest) );
	   checkCudaErrors( cudaFree(d_blueChDest ) );

	   checkCudaErrors( cudaFree(d_bufferRed) );
	   checkCudaErrors( cudaFree(d_bufferGreen) );
	   checkCudaErrors( cudaFree(d_bufferBlue) );
}
