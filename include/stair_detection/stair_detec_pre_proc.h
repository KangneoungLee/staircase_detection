/*********************************************************************
MIT License

Copyright (c) 2022 Kangneoung Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
