#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>

#include "util.h"




void HwachaDepthWiseConv(const size_t batch_size, 
                const size_t group_count, 
                const size_t channels, 
                const size_t in_height, const size_t in_width, 
                const size_t filter_count,
                const size_t kernel_height, const size_t kernel_width,
                const size_t pad_left_height, const size_t pad_left_width,
                const size_t pad_right_height, const size_t pad_right_width, 
                const size_t dilation_height, const size_t dilation_width, 
                const size_t stride_height, const size_t stride_width,
                const size_t out_height, const size_t out_width,
                const  int8_t* input, const int8_t*  filter, const  int32_t* bias, 
                int8_t* output){


  printf("Starting Hwacha Depth Wise Convolution!\n");
  printf("Batch Size: %li\n", batch_size);
  printf("Group Count: %li\n", group_count);
  printf("Channels: %li\n", channels);
  printf("Filter Count: %li\n", filter_count);

  printf("\n");
  printf("input\n");
  for (size_t m = 0; m < in_height; m++) {
    for (size_t k = 0; k < in_width * channels; k++) {
        printf("%i ", input[m * channels * in_width + k]);
    }
    printf("\n");
  }

  printf("\n");
  printf("input\n");
  for(size_t c = 0; c < channels; c++){
    printf("Channel %i\n",c);
    for (size_t y = 0; y < in_height; y+=1) {
      for (size_t x = c; x < in_width * channels; x+=channels) {
          printf("%i ", input[x + in_width * channels * y]);
      }
      printf("\n");
    }
    printf("\n");
  }

  //set zero in hwacha
  for (size_t m = 0; m < out_height; m++) {
    for (size_t k = 0; k < out_width*channels; k++) {
      output[m * channels *  out_width + k] = (int8_t) 0;
    }
  }

  // for (size_t m = 0; m < out_height; m++) {
  //   for (size_t k = 0; k < out_width*channels; k++) {
  //     printf("%i ", output[m * channels * out_width + k]);
  //   }
  //   printf("\n");
  // }

  //hwacha_init();
  
  //asm volatile ("vf 0(t0)");
  


  //printf("\nConsumed: %li\n", consumed);
  //printf("Size: %li\n", sizeof(input[0]));
  //vfadd.w vv0,vv0,vs2


setvcfg(3, 3, 3, 1);
int consumed = setvlen(channels);
printf("\nConsumed: %li\n", consumed);

int input_idx = 0;
int input_idy = 0;

int8_t* input_ptr= (int8_t*) input;
//int8_t* temp_output =  (int8_t*) malloc(channels*out_width*out_height*sizeof(int8_t)); //BufferOutput.GetBuffer(OutputElements);

for(int output_idy=0; output_idy<out_height; output_idy+=1){
  for(int output_idx=0; output_idx<out_width*channels; output_idx+=channels){
  
    asm volatile ("vmca va0, %0" : : "r" (output + output_idx + output_idy*out_width*channels)); //input
    //input_ptr = (int8_t*) input + output_idx + output_idy * in_width*channels; 
    input_idy = output_idy;
    for(int filter_idy=0; filter_idy<kernel_height; filter_idy++){
      input_idx = output_idx;
      for(int filter_idx=0; filter_idx<kernel_width*channels; filter_idx+=channels){
        //printf("Filter_IDX: %i; Filter_IDY:  %i; Input_IDX: %i;  Input_IDY: %i; Output_IDX: %i;  Output_IDY: %i; \n", filter_idx, filter_idy, input_idx, input_idy, output_idx, output_idy);
        //printf("Input Values: %i %i; Filter Values: %i  %i; \n", input[input_idx + input_idy*in_width*channels], input[1 + input_idx + input_idy*in_width*channels], filter[filter_idx + filter_idy*kernel_width*channels], filter[1 + filter_idx + filter_idy*kernel_width*channels]);
        
        asm volatile ("vmca va1, %0" : : "r" (input_ptr + input_idx + input_idy*in_width*channels)); 
        asm volatile ("vmca va2, %0" : : "r" (filter + filter_idx + filter_idy*kernel_width*channels)); 
        //asm volatile ("vmcs vs0, %0" : : "r" (channels)); 
        asm volatile ("la t0, vtest2" : : : "t0");
        asm volatile ("lw t1, 0(t0)");
        asm volatile ("vf 0(t0)");
        input_idx += channels;
      }
      input_idy += 1;
    }
    //printf("\n");
    
  }
}
asm volatile ("fence");

//for loop for as many channels 
// consumed = setvlen(out_width*out_height);
// printf("\nConsumed: %li\n", consumed);
// asm volatile ("vmca va0, %0" : : "r" (temp_output)); 
// asm volatile ("vmca va1, %0" : : "r" (temp_output+1)); 
// asm volatile ("vmca va2, %0" : : "r" (output)); 
// asm volatile ("vmca va3, %0" : : "r" (channels)); 
// asm volatile ("la t0, accumulate_channels" : : : "t0");
// asm volatile ("lw t1, 0(t0)");
// asm volatile ("vf 0(t0)");



   
    // asm volatile ("vmca va3, %0" : : "r" (input+in_width+1)); //bottom right pixel
    //asm volatile ("vmca va1, %0" : : "r" (output));  //output
    // asm volatile ("vmca va4, %0" : : "r" (filter));  //filter
    // asm volatile ("vmca va5, %0" : : "r" (filter+1));  //filter
    // asm volatile ("vmca va6, %0" : : "r" (filter+kernel_width));  //filter
    // asm volatile ("vmca va7, %0" : : "r" (filter+kernel_width+1));  //filter

    // asm volatile ("vmca va8, %0" : : "r" (output));  //output

    // asm volatile ("la t0, vtest2" : : : "t0");
    // asm volatile ("lw t1, 0(t0)");
   
    // asm volatile ("vf 0(t0)");
    // asm volatile ("fence");
    // vlb vv2, va2 #input_image
    // vlb vv3, va3 #input_image
    // vlb vv4, va4 #filter
    // vlb vv5, va5 #filter
    // vlb vv6, va6 #filter
    // vlb vv7, va7 #filter
    // #vmul vv0, vv0, vv4
    // #vmul vv1, vv1, vv5
    // #vmul vv2, vv2, vv6
    // #vmul vv3, vv3, vv7
    // vadd vv0, vv0, vv2
    // vadd vv1, vv1, vv3
    // vadd vv0, vv0, vv1
    // vsb vv0, va8


  

  // printf("\n");
  // printf("temp_output\n");
  // for (size_t m = 0; m < out_height; m++) {
  //   for (size_t k = 0; k < out_width*channels; k++) {
  //       printf("%i ", temp_output[m * channels * out_width + k]);
  //   }
  //   printf("\n");
  // }

  printf("\n");
  printf("output\n");
  for (size_t m = 0; m < out_height; m++) {
    for (size_t k = 0; k < out_width*channels; k++) {
        printf("%i ", output[m * channels * out_width + k]);
    }
    printf("\n");
  }

  // printf("\n");
  // printf("output\n");
  // for(size_t c = 0; c < channels; c++){
  //   printf("Channel %i\n",c);
  //   for (size_t y = c; y < out_height * channels; y+=channels) {
  //     for (size_t x = c; x < out_width * channels; x+=channels) {
  //         printf("%i ", output[x + out_width * y]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  // printf("input\n");
  // for (size_t m = 0; m < in_height; m++) {
  //   for (size_t k = 0; k < in_width; k++) {
  //       printf("%i ", input[m * in_width + k]);
  //   }
  //   printf("\n");
  // }

  // //printf("acc_t: %i\n", bias);

  // printf("params,batch_size: %i\n", params->batch_size);
  // bias +=  0;
  // output = input * weight;
  // output += 0;
  //output = input_arr;

  // printf("input addr: %hhn\n", input);
  // printf("output addr: %hhn\n", output);
  //
  // printf("params addr: %hhn \n", params);




  


  printf("\nFinished Hwacha Depthwise Convolution! \n");




}
