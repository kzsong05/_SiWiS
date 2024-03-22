/* -*- c++ -*- */
/*
 * Copyright 2023 zhc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/io_signature.h>
#include "baseband_impl.h"


namespace gr {
    namespace wifiradar {

    // #pragma message("set the following appropriately and remove this warning")
    // using input_type = float;
    // #pragma message("set the following appropriately and remove this warning")
    // using output_type = float;
    baseband::sptr
    baseband::make(int num_ant, float samp_rate, float baud_rate, float noise_threshold, int option)
    {
      return gnuradio::make_block_sptr<baseband_impl>(
        num_ant, samp_rate, baud_rate, noise_threshold, option);
    }


    /*
     * The private constructor
     */
    baseband_impl::baseband_impl(int num_ant_, float samp_rate_, float baud_rate_, float noise_threshold_, int option_):
    	gr::block("baseband",
        gr::io_signature::make(1 /* min inputs */, 8 /* max inputs */, sizeof(gr_complex)),
        gr::io_signature::make(1 /* min outputs */, 8 /*max outputs */, sizeof(gr_complex)))
    {
        num_ant = num_ant_;
        samp_rate = samp_rate_; // 1e6 Hz
        baud_rate = baud_rate_; // 10e3 Hz
        noise_threshold = noise_threshold_; // 0.005
        option = option_;
        std::cout<<"***************************"<<std::endl;
        std::cout<<"noise_threshold = "<<noise_threshold<<std::endl;
        std::cout<<"***************************"<<std::endl;


        in_data_buf_len = 0;
        set_output_multiple(50);
        for (int n = 0; n < num_ant; n++) signal_max[n] = 0;

        //+++++ edit by kunzhe
        auto now = std::chrono::high_resolution_clock::now();
    	auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        save_path = std::to_string(timestamp);

        writer_buffer.str("");
        buffer_count = 0;

        fstream_init();
        //camera_init(option);
        //----- edit by kunzhe
    }

    /*
     * Our virtual destructor.
     */
    baseband_impl::~baseband_impl()
    {
        //+++++ edit by kunzhe
        if(radio_writer.is_open()){
			radio_writer.close();
		}
        //----- edit by kunzhe
    }

    //+++++ edit by kunzhe
    void
    baseband_impl::fstream_init()
    {
        std::filesystem::path dir_path = "./" + save_path;
        if(!std::filesystem::exists(dir_path)){
            std::filesystem::create_directories(dir_path);
        }
        radio_writer.open("./" + save_path + "/radio.txt");
        if(!radio_writer.is_open()){
			throw std::runtime_error("Fail to open file radio!");
		}

		std::thread t(&baseband_impl::write_thread, this);
        t.detach();
    }

    void
    baseband_impl::write_thread()
    {
        while(true){
            std::unique_lock<std::mutex> lock(buffer_mutex);
            if(buffer_count >= 100){
                std::string str = writer_buffer.str();
                writer_buffer.str("");
                buffer_count = 0;
                lock.unlock();
                radio_writer << str;
            }
            else{
                lock.unlock();
            }
        }
    }

    void
    baseband_impl::camera_init(int option)
    {
        std::filesystem::path dir_path = "./" + save_path + "/video";
        if(!std::filesystem::exists(dir_path)){
            std::filesystem::create_directories(dir_path);
        }

        std::thread t(&baseband_impl::capture_video, this, option);
        t.detach();
    }

    void baseband_impl::capture_video(int option)
    {
    	cv::VideoCapture cap;
    	cap.open(option);
    	if (!cap.isOpened()) {
    	    std::cerr << "Error opening video capture" << std::endl;
    	    return;
    	}

    	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    	cap.set(cv::CAP_PROP_FPS, 30);

    	cv::Mat frame;

    	while (true) {
    	    cap >> frame;
    	    if (frame.empty()) {
    	    	std::cerr << "Failed to capture an image" << std::endl;
    	    	break;
    	    }
    	    auto now = std::chrono::high_resolution_clock::now();
    	    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    	    std::string filename = "./" + save_path + "/video/" + std::to_string(timestamp) + ".jpg";
    	    cv::imwrite(filename, frame);
    	}
    }
    //----- edit by kunzhe

    void
    baseband_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
        ninput_items_required[0] = 2000; //(samp_rate/baud_rate)*noutput_items;
    }

    int
    baseband_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
        //*********************************
        // oversampling rate
        // fs: sampling rate
        // output data rate: 10k Hz
        // WiFi sampling rate: 20MHz
        //*********************************
		const int num_stf = 160*(samp_rate/20.0e6);
		const int num_ltf = 160*(samp_rate/20.0e6);

        //*********************************
        // get input data
        //*********************************
        gr_complex* inn[8];
        gr_complex* out[8];
        int consume_len = 100000;
        for (int n = 0; n < num_ant; n++) {
            inn[n] = (gr_complex*) input_items[n];
            out[n] = (gr_complex*) output_items[n];
            if (consume_len > ninput_items[n]) consume_len = ninput_items[n];
        }

		/*
        // one-time execution to calculate the offset
        if (start_flag == true) {
            for (int n = 0; n < num_ant; n++) {
                signal_avg[n] = 0;
                for (int i = 0; i < consume_len; i++) signal_avg[n] += inn[n][i];
                signal_avg[n] /= float(consume_len);
                std::cout<<signal_avg[n]<<std::endl;
            }
            start_flag = false;
        }
        */


        // copy data to the buffer.
        for (int n = 0; n < num_ant; n++) {
            /*
            for (int i = 0; i < consume_len; i++) {
                // in_data_buf[n][in_data_buf_len+i] = inn[n][i];
                in_data_buf[n][in_data_buf_len+i] = (abs(inn[n][i]) < 0.008 ? 0 : inn[n][i]);
            }
            */
            memcpy(&in_data_buf[n][in_data_buf_len], inn[n], consume_len*sizeof(gr_complex));
        }
        in_data_buf_len += consume_len;
		consume_each(consume_len);



        //*********************************
        // calculate the effective buf len
        //*********************************
        int eff_buf_len = in_data_buf_len - num_stf - num_ltf;

        //*********************************
        // search for frame
        //*********************************
		int produce_len = 0;
        bool is_frame_found = true;
		for (int i = 2; i < eff_buf_len; i++) {
            //if (signal_max[0] < abs(in_data_buf[0][i])) signal_max[0] = abs(in_data_buf[0][i]);
            //noise_threshold = signal_max[0]/4.0;
            if (noise_threshold < 0.01) noise_threshold = 0.01;
            if (noise_threshold > 0.1) noise_threshold = 0.1;
            //noise_threshold = 0.05;

            float coeff = 1;   // from 0.5 to 2
            // Is it a frame?
            is_frame_found = true;
            is_frame_found = is_frame_found & (abs(abs(in_data_buf[0][i])-abs(in_data_buf[0][i-2])) < noise_threshold);
            is_frame_found = is_frame_found & (abs(abs(in_data_buf[0][i])-abs(in_data_buf[0][i+2])) > coeff*noise_threshold);
            if (is_frame_found) {
                //+++++ edit by kunzhe
                std::ostringstream oss;

				auto now = std::chrono::high_resolution_clock::now();
				auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
				oss << timestamp;

                for (int n = 0; n < num_ant; n++) {
                    gr_complex preamble_sum = 0;
                    int count = 0;
                    for (int k = num_stf; k < num_stf+num_ltf; k++) {
                        preamble_sum += in_data_buf[n][i+k];
                        count++;
                    }
                    preamble_sum /= float(count);
                    //send to next block
                    out[n][produce_len] = preamble_sum;
                    produce_len++;
                    //send to file
                    oss << "\t" << preamble_sum.real() << "\t" << preamble_sum.imag();
                }
				oss << "\n";

				std::unique_lock<std::mutex> lock(buffer_mutex);
				writer_buffer << oss.str();
				buffer_count += 1;
				lock.unlock();
                //----- edit by kunzhe

                i += num_stf + num_ltf;
            }
        }

        //*********************************
        // remove the data that have been searched
        //*********************************
        in_data_buf_len -= eff_buf_len;
        for (int n = 0; n < num_ant; n++) {
            memmove(&in_data_buf[n][eff_buf_len], &in_data_buf[n][0], in_data_buf_len*sizeof(gr_complex));
        }

        return produce_len;

    }

  } /* namespace wifiradar */
} /* namespace gr */
