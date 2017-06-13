#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "helpers.cuh"
#include <string.h>

#define DATASIZE 192000  //Define Data Size ( Sampling rate values )

#define EQ 10 //Define number of bands in equalizer

__global__  void equalize(cufftComplex *data_FFT, cufftReal *eq_D,int sample_inc,int BATCH);



char* itoa(int val, int base);
char *getFilename(int n,char* str);




int main (int argc,char **argv)
{

int BATCH=atoi(argv[1]);

//Input variables for equalizer,sample values

cufftReal *hostInputData=(cufftReal*)malloc(DATASIZE*BATCH*sizeof(cufftReal));
cufftReal *eq_strength=(cufftReal*)malloc(EQ*BATCH*sizeof(cufftReal));
  
//IO HANDLE ----------------------------------------------------------------- AUDIOIN  

FILE *file,*file_E;

int BATCH_NO=0; 

for(int b=0; b< BATCH; b++){
	
   char name_audio[15]="audioin"; 	
   char name_equalize[15]="equalizer";
   
    	
   char* file_name=getFilename(b,name_audio);	
   char* equ_file=getFilename(b,name_equalize); 	

   
   if( access( file_name, F_OK ) != -1 ) {
   file = fopen(file_name,"r");
   for(int i=0;i<DATASIZE;i++){
        fscanf(file,"%f ",&hostInputData[i + DATASIZE*BATCH_NO]);
   }  
   fclose(file);
 
   }else {
   printf("ERROR : NO SUCH FILE");
   return 0;
   }
   
   if( access( equ_file, F_OK ) != -1 ) {
   file_E = fopen(equ_file,"r");
   for(int i=0;i<EQ;i++){
        fscanf(file_E,"%f ",&eq_strength[i + EQ*BATCH_NO]);
   }  
   fclose(file_E);
   }else {
   printf("ERROR : NO SUCH FILE");
   return 0;
   }
   
   BATCH_NO++ ;
}


printf("%d \n ",BATCH);

//-----------------
	cudaEvent_t start,stop;
	float elapsedtime;
	
	//the moment at which we start measuring the time
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
//---------------------------------------------------------


    // --- Device side input data allocation and initialization
    cufftReal *deviceInputData; cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftReal));
    cufftReal *equ_D; cudaMalloc((void**)&equ_D, EQ * BATCH * sizeof(cufftReal));
    
    cudaMemcpy(deviceInputData, hostInputData, DATASIZE * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);
    cudaMemcpy(equ_D, eq_strength, EQ * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);

    // --- Host side output data allocation
    //cufftComplex *hostOutputData = (cufftComplex*)malloc((DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex));

    // --- Device side output data allocation
    cufftComplex *deviceOutputData; cudaMalloc((void**)&deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex));

    // --- Batched 1D FFTs
    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = BATCH;                      // --- Number of batched executions
    cufftPlanMany(&handle, rank, n, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, batch);

    //cufftPlan1d(&handle, DATASIZE, CUFFT_R2C, BATCH);
    cufftExecR2C(handle,  deviceInputData, deviceOutputData);


//---------------------
    cudaFree(deviceInputData);
    free(hostInputData);
    
    

//EQUALIZE---------------------------------------------------------------------

dim3 block(16,16);
int threads=(int)ceil(pow((DATASIZE*BATCH)/256.0,1/3.0));

dim3 grid(threads,threads,threads);
equalize<<<grid,block>>>(deviceOutputData,equ_D,ceil((DATASIZE/2 + 1) /(float)EQ),BATCH);   checkCudaError();


//INVERSE TRANSFORM--------------------------------------
    cufftHandle inverse;
    if (cufftPlanMany(&inverse, rank,n, 
                  onembed, ostride, odist,
                  inembed, istride, idist,CUFFT_C2R, batch)
     != CUFFT_SUCCESS){ 
		fprintf(stderr, "CUFFT error: Plan creation failed\n");
		return 0;
	}
	
	cufftReal *inversefft; 
	cudaMalloc((void**)&inversefft, DATASIZE * BATCH * sizeof(cufftReal));   checkCudaError();
  
	if (cufftExecC2R(inverse,deviceOutputData,inversefft) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecR2C Forward failed\n");	
		return 0;
	}
 
    cudaFree(deviceOutputData);  checkCudaError();
    cufftDestroy(handle);
    cufftDestroy(inverse);
    
 
    // --- Device->Host copy of the results
    cufftReal *opt = (cufftReal*)malloc(DATASIZE* BATCH * sizeof(cufftReal));
    
    if(opt == NULL){
	   printf("malloc() failed..!");
	   return 0;
	}
    
    cudaMemcpy(opt,inversefft,DATASIZE * (BATCH) * sizeof(cufftReal), cudaMemcpyDeviceToHost);  checkCudaError();



//-----------------write to files

/*
FILE *file_Out;
BATCH_NO=0;

for(int i=0 ;i< BATCH ;i++){
 
    char name_audio_out[15]="audioi_out"; 
    char* file_name_2=getFilename(i,name_audio_out);	
     
    file_Out = fopen(file_name_2,"w");
    for (int j=BATCH_NO*DATASIZE;j<BATCH_NO*DATASIZE+DATASIZE;j++){
        fprintf(file_Out,"%f ",opt[j]/DATASIZE);
    }
    fclose(file_Out);
    BATCH_NO++; 
}
* 
*/

    cufftDestroy(handle);


    
 //--------------------------------
 //the moment at which we stop measuring time 
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	//Find and print the elapsed time
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for operation is %.10f seconds\n",elapsedtime/(float)1000);
	//we get the elapsedtime in milli seconds. Thats why we divide by 1000   
    

}

__global__  void equalize(cufftComplex *data_FFT, cufftReal *eq_D,int sample_inc,int BATCH){


   
int blockId = blockIdx.x+ blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y)
+ (threadIdx.y * blockDim.x) + threadIdx.x;

  
   if(  (0 <= threadId ) && (threadId < (DATASIZE/2 + 1)*BATCH)){
        int batch_id=(threadId/(DATASIZE/2 + 1));
        int rel_id=(threadId%(DATASIZE/2 + 1));
        //handle +1 case
        if(rel_id == (DATASIZE/2 + 1) -1 ){
		   	rel_id--;
		}
        int eq_id=(EQ*batch_id) + (rel_id / sample_inc);
        

        
        data_FFT[threadId].x=eq_D[eq_id]*data_FFT[threadId].x;
        data_FFT[threadId].y=eq_D[eq_id]*data_FFT[threadId].y;
   }
   
   //printf("IN"); 
    	
}

char* itoa(int val, int base){
	if(val == 0){
	   return "0";	
	}
	static char buf[32] = {0};
	int i = 30;
	for(; val && i ; --i, val /= base)
	  buf[i] = "0123456789abcdef"[val % base];
	return &buf[i+1];	
}


char* getFilename(int n,char* str){
   char* val=itoa(n,10);
   strcat(str,val);
   char *ext=".txt";
   strcat(str,ext);

   return str;
}


