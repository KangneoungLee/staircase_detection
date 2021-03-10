#ifndef STAIR_DETEC_ONLY_RGB_H_
#define STAIR_DETEC_ONLY_RGB_H_

#include <chrono>
#include <vector>
#include <map>
#include <utility>  //pair 
#include <cmath>        // std::abs

#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include "stair_detection/stair_detec_config.h"


class STAIR_DETEC_ONLY_RGB{
	
	private:
	

	
	public: 
	/*constructor and destructor*/
	STAIR_DETEC_ONLY_RGB();
	~STAIR_DETEC_ONLY_RGB();

	bool  stair_case_detec_only_rgb(const cv::Mat& rgb_input, cv::Mat& roi_center_point_out, cv::Mat& midp_of_all_lines,int* final_center_point_col, int* final_center_point_row,int min_numof_lines_4_cluster_rgb_only = 6, 
	                                                int predefined_roi_height=200, int predefined_roi_width=160, unsigned short preproc_resize_height = 800, unsigned short preproc_resize_width = 450);

};



#endif