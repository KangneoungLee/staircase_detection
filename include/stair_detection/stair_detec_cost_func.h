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




#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h" 


class STAIR_DETEC_COST_FUNC{
	
	private:
	
	     int dummy;
		 bool _svm_offline_training_flag;
		 bool _coor_trans_flag;
		 float _small_roi_depth_th1;
		 float _small_roi_depth_th2;
		 float _small_roi_y_th_low;
		 float _small_roi_y_th_high;
		 float _invaild_depth_th;
		 int _invail_depth_count_th;
		 std::ofstream  in;
	
	public: 
	/*constructor and destructor*/
	STAIR_DETEC_COST_FUNC(bool svm_oft_flag, bool coor_trans_flag);
	~STAIR_DETEC_COST_FUNC();
	void cal_cost_wrapper_offline_image_true(cv::Mat& depth_img, int x_pixel, int y_start_pixel, int y_end_pixel, float fx, float fy, float px, float py, float dscale, unsigned short preproc_resize_height = 450, unsigned short preproc_resize_width = 800);
	void  cal_cost_wrapper(const cv::Mat& rgb_input, cv::Mat& depth_img, cv::Mat& roi_center_point_out, cv::Mat& midp_of_all_lines, std::vector<std::vector<float>>* vector_set_for_learning,float fx, float fy, float px, float py, float dscale,int min_numof_lines_4_cluster = 4, 
	                                                int predefined_roi_height=200, int predefined_roi_width=160, unsigned short preproc_resize_height = 800, unsigned short preproc_resize_width = 450);
	void least_square_fit(cv::Mat& array_x_coor_final_ls, cv::Mat& array_depth_and_y_coor_final_ls,  float* gradient_out, float* gradient_sub_diff, float* avg_depth_y_error_out, float* x_avg_error_out);
    void depth_data_collect(cv::Mat& depth_img, cv::Mat& array_depth_y, float fy, float py, float lower_lim, float upper_lim, float width_point, int* data_cnt, bool* invalid_depth_on);	
    void coordinate_transform(geometry_msgs::PointStamped& point_in, geometry_msgs::PointStamped& point_out, float head_down_angle, float height);	
};



#endif