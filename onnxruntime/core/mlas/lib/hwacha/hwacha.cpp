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
    for (size_t k = 0; k < in_width; k++) {
        printf("%i ", input[m * in_width + k]);
    }
    printf("\n");
  }

  //hwacha_init();
  
  //asm volatile ("vf 0(t0)");
  setvcfg(0, 1, 1, 1);
  int a = 10;
  
  int consumed = setvlen(channels);

  printf("\nConsumed: %li\n", consumed);
  printf("Size: %li\n", sizeof(input[0]));
  //vfadd.w vv0,vv0,vs2

  
  asm volatile ("vmca va0, %0" : : "r" (input));
  asm volatile ("vmca va1, %0" : : "r" (output));
  asm volatile ("la t0, vtest_avi" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("vf 0(t0)");
  asm volatile ("fence");

  asm volatile ("la t0, vtest2" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  //asm volatile ("vmcs vs2, %0" : : "r" (input[2]));
  asm volatile ("vf 0(t0)");
  asm volatile ("fence");

  

  printf("\n");
  printf("output\n");
  for (size_t m = 0; m < out_height; m++) {
    for (size_t k = 0; k < out_width; k++) {
        printf("%i ", output[m * out_width + k]);
    }
    printf("\n");
  }

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




  


  printf("Finished Hwacha Depthwise Convolution! \n");




}
