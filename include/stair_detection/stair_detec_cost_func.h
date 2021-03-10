#ifndef STAIR_DETEC_COST_FUNC_H_
#define STAIR_DETEC_COST_FUNC_H_

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


class STAIR_DETEC_COST_FUNC{
	
	private:
	
	     int dummy;
		 bool svm_offline_training_flag;
		 std::ofstream  in;
	
	public: 
	/*constructor and destructor*/
	STAIR_DETEC_COST_FUNC(bool svm_oft_flag);
	~STAIR_DETEC_COST_FUNC();
	void cal_cost_wrapper_offline_image_true(cv::Mat& depth_img, int x_pixel, int y_start_pixel, int y_end_pixel, float fx, float fy, float px, float py, float dscale, unsigned short preproc_resize_height, unsigned short preproc_resize_width);
	void  cal_cost_wrapper(const cv::Mat& rgb_input, cv::Mat& depth_img, cv::Mat& roi_center_point_out, cv::Mat& midp_of_all_lines, std::vector<std::vector<float>>* vector_set_for_learning,float fx, float fy, float px, float py, float dscale,int min_numof_lines_4_cluster = 4, 
	                                                int predefined_roi_height=200, int predefined_roi_width=160, unsigned short preproc_resize_height = 800, unsigned short preproc_resize_width = 450);
	void least_square_fit(cv::Mat& array_x_coor_final_ls, cv::Mat& array_depth_and_y_coor_final_ls,  float* gradient_out, float* gradient_sub_diff, float* avg_depth_y_error_out, float* x_avg_error_out);												
};



#endif