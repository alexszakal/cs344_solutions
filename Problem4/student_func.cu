//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cuda.h>
#include <iostream>

#include <vector>    //for debug!
#include <algorithm> //for debug!!!

#include "cuda_runtime.h"
#include "device_launch_parameters.h"  //This header is only needed to satisfy the indexing in Eclipse...

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void histoCalc(unsigned int* const d_input, const size_t arraySize,
		                  unsigned int shift,  unsigned int* d_histoOut){
	size_t gid=blockIdx.x*blockDim.x + threadIdx.x;

	if(gid>=arraySize)
		return;

	unsigned int mask = 3 << shift;
	unsigned int binNum = (d_input[gid] & mask) >> shift;

	atomicAdd(&d_histoOut[binNum], 1);

}

void checkHistoValidity(unsigned int* const d_inputVals, size_t numElems, unsigned int shift, unsigned int* d_histogram){
	//copy the input for serial debug
	unsigned int * h_input = new unsigned int[numElems];
	checkCudaErrors( cudaMemcpy( h_input, d_inputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	unsigned int histogram[4]={0,0,0,0};
	unsigned int mask = 3 << shift;
	for(int i=0; i<numElems; ++i){
		histogram[(h_input[i]&mask)>>shift]++;
	}

	unsigned int debugHistogram[4];
	checkCudaErrors( cudaMemcpy( debugHistogram, d_histogram, 4*sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	for(int i=0; i<4; ++i){
		if(debugHistogram[i] != histogram[i]){
			std::cout << " PROBLEM in histogram calculation!";
			std::cout<< "\nshift=" << shift << " bin="<<i<<" CPU: "<< histogram[i] << " GPU: " << debugHistogram[i];
			return;
		}
	}
	std::cout << "\nHisto calculation OK, shift = " << shift;

}

__global__ void calculatePredicates(unsigned int radix,
                                    unsigned int shift,
		                            unsigned int* const d_inputVals,
									bool* d_predicates,
		                            const size_t numElems){
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid>=numElems){
		return;
	}
	unsigned int inputValue = d_inputVals[gid];
	inputValue = inputValue & (3<<shift);
	inputValue = inputValue >> shift;
	if(inputValue==radix)
		d_predicates[gid] = 1;
}

void checkCalculatePredicates(unsigned int radix, unsigned int shift,
                              unsigned int* const d_inputVals,
                              bool* d_predicates, const size_t numElems) {
  // get the calculated predicates
  bool* h_predicates = new bool[numElems];
  checkCudaErrors(cudaMemcpy(h_predicates, d_predicates,
                             numElems * sizeof(bool), cudaMemcpyDeviceToHost));
  // get the input values
  unsigned int* h_inputVals = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals,
                             numElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));

  // calculate predicates serially
  for (int i = 0; i < numElems; i++) {
    bool predicate;
    unsigned int inputValue = h_inputVals[i];
    inputValue = inputValue & (3 << shift);
    inputValue = inputValue >> shift;
    predicate = (inputValue == radix);
    if (predicate != h_predicates[i]) {
      std::cout << "\nProblem with parallel predicate calculation";
      return;
    }
    //std::cout << "\nPredicate calculation OK! radix=" << radix
    //          << " shift=" << shift;
  }

  delete[] h_inputVals;
  delete[] h_predicates;
}

__global__ void calculateScanBlock(bool *d_predicates,
		                           const size_t numElems,
		                           unsigned int *d_blockScans,
								   unsigned int *d_blockSums) {
	/**
	 * Calculate the scan of the block and the blockSum value.
	 */
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid = threadIdx.x;

	extern __shared__ unsigned int s_blockVal[];

	if (gid >= numElems) {
		//Write 0 to the corresponding shared memory element
		s_blockVal[tid] = 0;
		return;
	}

	//Copy data from global to shared memory
	if (tid > 0) {
		if (d_predicates[gid - 1])     //Divergent code, NEEDS optimization!
			s_blockVal[tid] = 1;
		else
			s_blockVal[tid] = 0;
	} else
		s_blockVal[0] = 0;

//	if(tid==0)    //Fenti helyett
//		s_blockVal[0]=0
//	else{
//		a_blockVal[tid]=d_predicates[gid-1];
//	}

	__syncthreads();

	//Perform Hillis method for Scan
	unsigned int addVal;
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		addVal = 0;
		if (tid - stride >= 0 ) {
			addVal = s_blockVal[tid - stride];
		}
		__syncthreads();
		s_blockVal[tid] += addVal;
		__syncthreads();
	}

	//Write results to global memory
	d_blockScans[gid] = s_blockVal[tid];

	//Calulate blockSum and write to global memory                 //Divergent code, NEEDS optimization!!!
	if (tid == 0) {
		int lastValueGlob=min(blockDim.x * (blockIdx.x + 1) - 1, static_cast<unsigned int>(numElems-1));
		int lastValueLocal=lastValueGlob - blockIdx.x*blockDim.x;
		//if (d_predicates[blockDim.x * (blockIdx.x + 1) - 1] == 1) {
		if (d_predicates[lastValueGlob] == 1) {
			d_blockSums[blockIdx.x] = s_blockVal[lastValueLocal] + 1;
		} else {
			d_blockSums[blockIdx.x] = s_blockVal[lastValueLocal];
		}
	}
}

__global__ void addBlockSums(unsigned int *d_blockScans,
		                     unsigned int *d_blockSums,
							 const size_t numElems){
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gid>=numElems)
		return;

	if(blockIdx.x == 0)
		return;

	int addVal=0;
	for(int i=0; i<blockIdx.x; ++i){
		addVal+=d_blockSums[i];
	}
	d_blockScans[gid] += addVal;
}

__global__ void scatterValuesAndPos(bool * d_predicates,
		              unsigned int *d_blockScans,
					  unsigned int radixOffset,
		              unsigned int* const d_inputVals,
					  unsigned int* const d_inputPos,
					  unsigned int* const d_outputVals,
					  unsigned int* const d_outputPos,
					  size_t numElems){

	const int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gid>=numElems)
		return;

	if(d_predicates[gid]){
		int targetIndex = d_blockScans[gid]+radixOffset;
		d_outputVals[targetIndex] = d_inputVals[gid];
		d_outputPos[targetIndex] = d_inputPos[gid];
	}
}

void checkBlockScans(bool* d_predicates, int numElems,
                     unsigned int* d_blockScans, unsigned int* d_blockSums,
                     dim3 gridSize, dim3 blockSize) {
  // get the calculated predicates
  bool* h_predicates = new bool[numElems];
  checkCudaErrors(cudaMemcpy(h_predicates, d_predicates,
                             numElems * sizeof(bool), cudaMemcpyDeviceToHost));
  // get the result of blockScans
  unsigned int* h_blockScans = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_blockScans, d_blockScans,
                             numElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));

  // get the result of blockSums
  unsigned int* h_blockSums = new unsigned int[gridSize.x];
  checkCudaErrors(cudaMemcpy(h_blockSums, d_blockSums,
                             gridSize.x * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));

  //Go through blocks:
  for(int blockIdx=0; blockIdx<gridSize.x; ++blockIdx){
	  int blockScanVal=0;
	  if(h_blockScans[blockIdx*blockSize.x]!=0)
		  std::cout << "\nError in blockScan values! blockIdx=" << blockIdx << " tid=" << 0;
	  for(int i=1; i<blockSize.x; ++i){
		int gid=blockIdx*blockSize.x+i;
		if (gid<numElems){
			blockScanVal += h_predicates[gid-1];
			if(blockScanVal != h_blockScans[gid]){
				std::cout << "\nError in blockScan values! blockIdx=" << blockIdx << " tid=" << i;
			}
		}
	  }
	  blockScanVal += h_predicates[(blockIdx+1)*blockSize.x-1];
	  if ( (h_blockSums[blockIdx] != blockScanVal) and (blockIdx != gridSize.x-1)){
		  std::cout << "\nError in blockScanVal blockIdx=" << blockIdx;
	  }

  }
}

void checkAddBlockSums(bool* d_predicates, unsigned int* d_blockScans, int numElems){
  // get the calculated predicates
  bool* h_predicates = new bool[numElems];
  checkCudaErrors(cudaMemcpy(h_predicates, d_predicates,
                             numElems * sizeof(bool), cudaMemcpyDeviceToHost));

  // get the result of blockScans
  unsigned int* h_blockScans = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_blockScans, d_blockScans,
                             numElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));

  int scanVal=0;
  for(int i=1; i<numElems; ++i){
	  scanVal += h_predicates[i-1];
	  if(h_blockScans[i] != scanVal){
		  std::cout << "\nError in addBlockSums! i=" << i << "  CPUval=" << scanVal << "  GPUval=" << h_blockScans[i];
	  }
  }
  std::cout << "\ncheckAddBlockSums ran";
}

void calculateCompact(unsigned int radix,
		              unsigned int shift,
					  unsigned int radixOffset,    //scan of the histogram of bins with radix values
					  unsigned int* const d_inputVals,
					  unsigned int* const d_inputPos,
					  unsigned int* const d_outputVals,
					  unsigned int* const d_outputPos,
					  size_t numElems){
	dim3 blockSize(128,1,1);
	dim3 gridSize(numElems/blockSize.x+1 ,1,1);

	bool *d_predicates;
	checkCudaErrors( cudaMalloc( (void**)&d_predicates, numElems * sizeof(bool)) );
	checkCudaErrors( cudaMemset( d_predicates, 0, numElems*sizeof(bool)) );

	//Calculate predicates
	calculatePredicates<<<gridSize, blockSize>>>(radix, shift, d_inputVals, d_predicates, numElems);
	cudaDeviceSynchronize();
	//DEBUG OK
	//checkCalculatePredicates(radix, shift, d_inputVals, d_predicates, numElems);

	//Calculate scan in blocks then write blocksum
	unsigned int *d_blockScans;
	unsigned int *d_blockSums;
	checkCudaErrors( cudaMalloc( (void**)&d_blockScans, numElems * sizeof(unsigned int)) );
	checkCudaErrors( cudaMalloc( (void**)&d_blockSums,  gridSize.x * sizeof(unsigned int)) );
	calculateScanBlock<<<gridSize, blockSize, blockSize.x*sizeof(unsigned int)>>>(d_predicates,
			                                                                      numElems,
																				  d_blockScans,
																				  d_blockSums);
	cudaDeviceSynchronize();
	//DEBUG OK
	//checkBlockScans(d_predicates, numElems, d_blockScans, d_blockSums, gridSize, blockSize);

	addBlockSums<<<gridSize, blockSize>>>(d_blockScans,
			                              d_blockSums,
								          numElems);
	cudaDeviceSynchronize();
    //DEBUG OK
	//checkAddBlockSums(d_predicates, d_blockScans, numElems);

	scatterValuesAndPos<<<gridSize,blockSize>>>(d_predicates,
                                                d_blockScans,
			                                    radixOffset,
                                                d_inputVals,
			                                    d_inputPos,
			                                    d_outputVals,
			                                    d_outputPos,
			                                    numElems);
	cudaDeviceSynchronize();

	checkCudaErrors( cudaFree( d_predicates ) );
}

void your_sort(unsigned int*  d_inputVals, unsigned int*  d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos, size_t numElems) {              //!!!!levettem a const-ot a debug miatt

  // 4-way radix sort is implemented
  dim3 histoBlockSize(128, 1, 1);
  dim3 histoGridSize(numElems / histoBlockSize.x + 1, 1, 1);

  size_t bitSize = 8 * sizeof(unsigned int);

  unsigned int* d_histogram;
  checkCudaErrors(cudaMalloc((void**)&d_histogram, 4 * sizeof(unsigned int)));

  unsigned int* d_valuePointers[2] = {d_inputVals, d_outputVals};
  unsigned int* d_posiPointers[2] = {d_inputPos, d_outputPos};
  for (unsigned int shift = 0; shift < bitSize; shift += 2) {
    // 1) produce histogram of radix elements (parallel code)
    checkCudaErrors(cudaMemset(d_histogram, 0, 4 * sizeof(unsigned int)));
    histoCalc<<<histoGridSize, histoBlockSize>>>(d_inputVals, numElems, shift,
                                                 d_histogram);
    cudaDeviceSynchronize();

    //DEBUG OK
    //checkHistoValidity(d_inputVals, numElems, shift, d_histogram);

    // 2) Scan of the radixNum length histogram (serial code)
    unsigned int h_histogram[4];
    unsigned int h_radixOffset[4];
    checkCudaErrors(cudaMemcpy(h_histogram, d_histogram,
                               4 * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
    h_radixOffset[0] = 0;
    for (int i = 1; i < 4; ++i) {
      h_radixOffset[i] = h_radixOffset[i - 1] + h_histogram[i - 1];
    }
    // DEBUG OK
    //    std::cout << "\nh_histogram[0]=" << h_histogram[0]
    //              << " h_histoScan[0]=" << h_histoScan[0];
    //    std::cout << "\nh_histogram[1]=" << h_histogram[1]
    //              << "  h_histoScan[0]=" << h_histoScan[1];
    //    std::cout << "\nh_histogram[2]=" << h_histogram[2]
    //              << "  h_histoScan[0]=" << h_histoScan[2];
    //    std::cout << "\nh_histogram[3]=" << h_histogram[3]
    //              << " h_histoScan[0]=" << h_histoScan[3];

    // 3) go through the four possible radixes and compact values
    // corresponding to each
    for (unsigned int radix = 0; radix < 4; radix++) {
      calculateCompact(
          radix, shift, h_radixOffset[radix], d_valuePointers[(shift % 4) / 2],
          d_posiPointers[(shift % 4) / 2], d_valuePointers[1 - (shift % 4) / 2],
          d_posiPointers[1 - (shift % 4) / 2], numElems);
    }
  }

  // Copy the results to the correct place
  checkCudaErrors(cudaMemcpy(d_valuePointers[0], d_valuePointers[1],
                             numElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_posiPointers[0], d_posiPointers[1],
                             numElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();

  // Final check of increasing order
//  unsigned int* valuesResult = new unsigned int[numElems];
//  unsigned int* posiResult = new unsigned int[numElems];
//  checkCudaErrors(cudaMemcpy(valuesResult, d_valuePointers[0],
//                             numElems * sizeof(unsigned int),
//                             cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaMemcpy(posiResult, d_posiPointers[0],
//                             numElems * sizeof(unsigned int),
//                             cudaMemcpyDeviceToHost));
//
//  for (int i = 1; i < numElems; ++i) {
//  //  std::cout << '\n' << i << ": " << valuesResult[i] << " position: " << posiResult[i];
//	  if(valuesResult[i]<valuesResult[i-1])
//		  std::cout << " \norder not increasing i=" << i << "  value=" << valuesResult[i] << "  prevValue=" << valuesResult[i-1];
//  }
}
