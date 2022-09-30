/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2022, Kangneoung Lee.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   *The names of its  contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Kangneoung Lee
 *********************************************************************/

#ifndef STAIR_DETEC_PRE_PROC_H_
#define STAIR_DETEC_PRE_PROC_H_

#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>
#include <map>
#include <utility>  //pair 
#include <cmath>        // std::abs

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include "stair_detection/stair_detec_config.h"


class STAIR_DETEC_PRE_PROC{
	
	private:
	
	int dummy;
	
	
	public: 
	
	/*constructor and destructor*/
	STAIR_DETEC_PRE_PROC();
	~STAIR_DETEC_PRE_PROC();
	
	void stair_line_extraction(cv::Mat rgb_image,   std::vector<cv::Vec4i> *edge_lines, cv::Mat* resized_rgb_image, cv::Mat* resized_gray_image, bool encoding_rgb_flag = false, unsigned short preproc_resize_height = 800, unsigned short preproc_resize_width = 450,  unsigned short canny_lt=50,
	                                                       unsigned short canny_ht=100,  unsigned short houghp_th = 50,  unsigned short houghp_min_line_len = 50, unsigned short houghp_max_line_gap =5, int noise_rm_pre_proc_index = 1);
														   
	void select_lines_from_hough(const cv::Mat& rgb_input, const cv::Mat& gray_input, std::vector<cv::Vec4i>* org_edge_lines,  std::vector<cv::Vec4i>* final_edge_lines,  std::map<float, std::vector<cv::Vec4i>>*  lines_hist, std::map<int, std::vector<cv::Vec4i>> *lines_hist_top_three, float maxslope_in_pixel = 0.6, float minslope_in_pixel = 0.05);
	
	void grouping_lines_and_define_centers(const cv::Mat& rgb_input, std::map<int, std::vector<cv::Vec4i>> *lines_hist_top_three, cv::Mat* roi_center_point_out, cv::Mat* midp_of_all_lines, int min_numof_lines_4_cluster = 4, int predefined_roi_height=160, int predefined_roi_width=160);
};


#endif
