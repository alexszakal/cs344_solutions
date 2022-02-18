/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.
*/

#include "utils.h"

__global__
void naiveHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numElems){
	int gid = blockIdx.x*blockDim.x+threadIdx.x;

	if(gid>=numElems)
		return;

	atomicAdd(&histo[vals[gid]], 1);
}

__global__
void sharedHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               unsigned int numBins,
			   unsigned int numElems,
			   unsigned int dataPerThread)
{
	extern __shared__ int localHisto[];

	int gid = blockIdx.x*blockDim.x + threadIdx.x;

	//Zero out values coalesced
	for(int i=threadIdx.x; i<numBins; i+=blockDim.x)
		localHisto[i]=0;

    __syncthreads();

    for(int i=gid; i<numElems; i+=blockDim.x*gridDim.x){
    	atomicAdd(&localHisto[vals[i]], 1);
    }

	__syncthreads();

	//Write result to global memory
	for(int i=threadIdx.x; i<numBins; i+=blockDim.x)
		atomicAdd(&histo[i], localHisto[i]);

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
	//std::cout<< "\nNumber of elements: " << numElems <<'\n';
	//std::cout<< "\nNumber of bins: " << numBins << '\n';

	dim3 blockSize(512);
	dim3 gridSize(numElems/blockSize.x+1);

	//* NAIVE SOLUTION
	//naiveHisto <<< gridSize, blockSize >>>(d_vals, d_histo, numElems);

	//** Using shared memory
	int dataPerThread = 128;
	gridSize.x=numElems/blockSize.x/dataPerThread+1;
	sharedHisto <<< gridSize, blockSize, numBins*sizeof(unsigned int) >>>(d_vals, d_histo, numBins, numElems, dataPerThread);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
