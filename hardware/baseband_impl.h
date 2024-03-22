/* -*- c++ -*- */
/*
 * Copyright 2023 zhc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIRADAR_BASEBAND_IMPL_H
#define INCLUDED_WIFIRADAR_BASEBAND_IMPL_H

#include <gnuradio/wifiradar/baseband.h>
#include <fstream>
#include <ctime>
#include <chrono>
#include <iostream>

//+++++ edit by kunzhe
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <thread>
#include <string>
#include <mutex>
#include <chrono>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <condition_variable>
//----- edit by kunzhe


namespace gr {
  namespace wifiradar {

    class baseband_impl : public baseband
    {
     private:
        int num_ant;
        float samp_rate;
        float baud_rate;
        float noise_threshold;
        int option;

        gr_complex signal_avg[10];
        float signal_max[10];
        bool start_flag = true;

        std::chrono::time_point<std::chrono::system_clock> time_start;

        gr_complex in_data_buf[8][100000];
        int in_data_buf_len = 0;

        //+++++ edit by kunzhe
        std::string save_path;
        std::ofstream radio_writer;
        std::mutex buf_mutex;

        long long  shared_timestamp_buf[10000];
		gr_complex shared_complex_buf[10][10000];
		long       shared_buf_len =  0;

		long long  print_shared_timestamp_buf[10000];
		gr_complex print_shared_complex_buf[10][10000];

        void fstream_init();
        void write_thread();
        void camera_init();
        void capture_video();
        //----- edit by kunzhe

		int avg_frame_count = 0;
		gr_complex preamble_arr[10];

     public:
      baseband_impl(int num_ant, float samp_rate, float baud_rate, float noise_threshold, int option);
      ~baseband_impl();

      // Where all the action really happens

      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };
  } // namespace wifiradar
} // namespace gr

#endif /* INCLUDED_WIFIRADAR_BASEBAND_IMPL_H */
