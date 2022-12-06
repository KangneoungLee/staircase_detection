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

#ifndef STAIR_DETEC_ONLINE_FUNC_H_
#define STAIR_DETEC_ONLINE_FUNC_H_

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
#include "stair_detection/stair_detec_cost_func.h"



#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h" 

#define ARRAY_SIZE_FOR_ONLINE_LEARNING 50
class STAIR_DETEC_ONLINE_FUNC{
	
	private:
		 float _online_rule_base_grad_max;
         float _online_rule_base_grad_min;
		 float _online_rule_base_grad_diff_th_y3;
		 float _online_rule_base_grad_diff_th_y2;
		 float _online_rule_base_grad_diff_th_y1;
		 float _online_rule_base_grad_diff_th_x3;
		 float _online_rule_base_grad_diff_th_x2;
		 float _online_rule_base_grad_diff_th_x1;
         float _online_rule_base_depth_y_error_th_y3;
		 float _online_rule_base_depth_y_error_th_y2;
		 float _online_rule_base_depth_y_error_th_y1;
		 float _online_rule_base_depth_y_error_th_x3;
		 float _online_rule_base_depth_y_error_th_x2;
		 float _online_rule_base_depth_y_error_th_x1;
		 float _online_rule_base_depth_y_error_th_min;
		 
		 float _online_init_rule_base_grad_max;
         float _online_init_rule_base_grad_min;
		 float _online_init_rule_base_grad_diff_th_y3;
		 float _online_init_rule_base_grad_diff_th_y2;
		 float _online_init_rule_base_grad_diff_th_y1;
		 float _online_init_rule_base_grad_diff_th_x3;
		 float _online_init_rule_base_grad_diff_th_x2;
		 float _online_init_rule_base_grad_diff_th_x1;
         float _online_init_rule_base_depth_y_error_th_y3;
		 float _online_init_rule_base_depth_y_error_th_y2;
		 float _online_init_rule_base_depth_y_error_th_y1;
		 float _online_init_rule_base_depth_y_error_th_x3;
		 float _online_init_rule_base_depth_y_error_th_x2;
		 float _online_init_rule_base_depth_y_error_th_x1;
		 float _online_init_rule_base_depth_y_error_th_min;
		 
		 float _no_queue_online;
		 float _probability_discard_th;
		 float _probability_scale_del_beta;
		 float _probability_scale_del_gamma;
		 float _probability_scale_del_x;
		 float _probability_scale_del_y;
		 float _online_detect_flag_reset_timer;
		 
		 bool _implementation_on_jetson;
		 
		 int _preproc_resize_height;
		 int _preproc_resize_width;
		 
		 bool _activate_flag;
		 ros::Time _detect_init_time;
		 
		 float _detected_roi_x;
		 float _detected_roi_y; 
		 
		 bool _array_full_flag_bt = false;
		 int _array_indicator_bt = 0;
		 
		 bool _array_full_flag_gmm = false;
		 int _array_indicator_gmm = 0;

         float  _array_del_beta_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING] = {0};
		 float  _array_del_gamma_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
         float  _array_gradient_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};		 
		 float  _array_prob_del_beta_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_prob_del_gamma_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_prob_del_x_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_prob_del_y_bt_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};

         float  _array_del_beta_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_del_gamma_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
         float  _array_gradient_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};		 
		 float  _array_prob_del_beta_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_prob_del_gamma_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_prob_del_x_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 float  _array_prob_del_y_gmm_adp[ARRAY_SIZE_FOR_ONLINE_LEARNING]  = {0};
		 
		 int _detection_count = 0;
		 
		 float _fx;
		 float _fy;
		 float _px;
		 float _py;
		 
		 float _dscale;
		 
	     int dummy;
	
	public: 
	/*constructor and destructor*/
	STAIR_DETEC_ONLINE_FUNC(ros::NodeHandle& param_nh);
	~STAIR_DETEC_ONLINE_FUNC();
	
	bool rule_base_detect(const cv::Mat& rgb_input, const cv::Mat& depth_input, std::vector<std::vector<float>>* vector_set_for_learning, float focal_len_x, float opt_center_x, float focal_len_y, float opt_center_y, float depth_scale);
	void threshold_update(float del_beta, float del_gamma, float del_x, float del_y, float gradient);
	bool reject_false_positive(const cv::Mat& depth_input,float center_x_val ,float center_y_val, float center_depth_val, float center_col, float center_row, float roi_size);
	
	STAIR_DETEC_COST_FUNC* str_det_cost_func;
};



#endif
