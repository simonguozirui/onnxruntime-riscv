#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>

#include "util.h"

void Hwachaim2col(size_t batch_size, size_t height, size_t width, size_t channels,
                size_t I, size_t K,
                const int8_t* input_arr, const int8_t* output_arr, struct ConvParams * params){


  printf("Starting Hwacha Im2Col!\n");

  printf("Batch Size: %li\n", batch_size);
  printf("Height: %li\n", height);
  printf("Width: %li\n", width);
  printf("Channels: %li\n", channels);

  printf("I: %li\n", I);
  printf("K: %li\n", K);

  input_arr = output_arr;
  output_arr = input_arr;
  params->batch_size = 1;
  // printf("input addr: %hhn\n", input);
  // printf("output addr: %hhn\n", output);
  //
  // printf("params addr: %hhn \n", params);




  hwacha_init();


  printf("Finished Hwacha Im2Col! \n");




}
