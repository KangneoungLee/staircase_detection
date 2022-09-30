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

#include "stair_detection/stair_detec_online_func.h"

#define EXP(X) 1+X+X*X/2+X*X*X/6   /*taylor expansion of exponential function */
/*constructor and destructor*/
STAIR_DETEC_ONLINE_FUNC::STAIR_DETEC_ONLINE_FUNC(ros::NodeHandle& param_nh)
{
	float online_init_rule_base_grad_max = 1.85;
    float online_init_rule_base_grad_min = 0.4;
	float online_init_rule_base_grad_diff_th_y3 = 0.3;
	float online_init_rule_base_grad_diff_th_y2 = 0.2;
	float online_init_rule_base_grad_diff_th_y1 = 0.1;
	float online_init_rule_base_grad_diff_th_x3 = 1.7;
	float online_init_rule_base_grad_diff_th_x2 = 0.8;
	float online_init_rule_base_grad_diff_th_x1 = 0.3;
    float online_init_rule_base_depth_y_error_th_y3 = 2;
	float online_init_rule_base_depth_y_error_th_y2 = 1.4;
	float online_init_rule_base_depth_y_error_th_y1 = 1;
	float online_init_rule_base_depth_y_error_th_x3 = 1.6;
	float online_init_rule_base_depth_y_error_th_x2 = 0.9;
	float online_init_rule_base_depth_y_error_th_x1 = 0.3;
	float online_init_rule_base_depth_y_error_th_min = 0.5;

	float no_queue_online = 5;
	float probability_discard_th = 0.1;
	float probability_scale_del_beta = 1;
	float probability_scale_del_gamma = 1;
	float probability_scale_del_x = 0.05;
	float probability_scale_del_y = 0.01;
	float online_detect_flag_reset_timer = 5;
	
	int preproc_resize_height = 480;
	int preproc_resize_width = 848;
	 
	bool implementation_on_jetson = false; 

	param_nh.getParam("stair_gradient_maximum_threshold",online_init_rule_base_grad_max);
	param_nh.getParam("stair_gradient_minimum_threshold",online_init_rule_base_grad_min);
	param_nh.getParam("online_init_rule_base_grad_diff_th_y3",online_init_rule_base_grad_diff_th_y3);
	param_nh.getParam("online_init_rule_base_grad_diff_th_y2",online_init_rule_base_grad_diff_th_y2);
	param_nh.getParam("online_init_rule_base_grad_diff_th_y1",online_init_rule_base_grad_diff_th_y1);
	param_nh.getParam("online_init_rule_base_grad_diff_th_x3",online_init_rule_base_grad_diff_th_x3);
	param_nh.getParam("online_init_rule_base_grad_diff_th_x2",online_init_rule_base_grad_diff_th_x2);
	param_nh.getParam("online_init_rule_base_grad_diff_th_x1",online_init_rule_base_grad_diff_th_x1);
	param_nh.getParam("online_init_rule_base_depth_y_error_th_y3",online_init_rule_base_depth_y_error_th_y3);
	param_nh.getParam("online_init_rule_base_depth_y_error_th_y2",online_init_rule_base_depth_y_error_th_y2);
	param_nh.getParam("online_init_rule_base_depth_y_error_th_y1",online_init_rule_base_depth_y_error_th_y1);
	param_nh.getParam("online_init_rule_base_depth_y_error_th_x3",online_init_rule_base_depth_y_error_th_x3);
	param_nh.getParam("online_init_rule_base_depth_y_error_th_x2",online_init_rule_base_depth_y_error_th_x2);
	param_nh.getParam("online_init_rule_base_depth_y_error_th_x1",online_init_rule_base_depth_y_error_th_x1);
	param_nh.getParam("online_init_depth_and_y_leastsquare_error_min_th",online_init_rule_base_depth_y_error_th_min);

	param_nh.getParam("number_of_queue_for_online_learning",no_queue_online);
	param_nh.getParam("probability_discard_th",probability_discard_th);
	param_nh.getParam("probability_scale_del_beta",probability_scale_del_beta);
	param_nh.getParam("probability_scale_del_gamma",probability_scale_del_gamma);
	param_nh.getParam("probability_scale_del_x",probability_scale_del_x);
	param_nh.getParam("probability_scale_del_y",probability_scale_del_y);
	param_nh.getParam("online_detect_flag_reset_timer",online_detect_flag_reset_timer);
	
	param_nh.getParam("image_resize_height",preproc_resize_height);
	param_nh.getParam("image_resize_width",preproc_resize_width);
	
	param_nh.getParam("implementation_on_jetson",implementation_on_jetson);
	
	this->_implementation_on_jetson = implementation_on_jetson;

	this->_online_init_rule_base_grad_max = online_init_rule_base_grad_max;
	this->_online_init_rule_base_grad_min = online_init_rule_base_grad_min;
	this->_online_init_rule_base_grad_diff_th_y3 = online_init_rule_base_grad_diff_th_y3;
	this->_online_init_rule_base_grad_diff_th_y2 = online_init_rule_base_grad_diff_th_y2;
	this->_online_init_rule_base_grad_diff_th_y1 = online_init_rule_base_grad_diff_th_y1;
	this->_online_init_rule_base_grad_diff_th_x3 = online_init_rule_base_grad_diff_th_x3;
	this->_online_init_rule_base_grad_diff_th_x2 = online_init_rule_base_grad_diff_th_x2;
	this->_online_init_rule_base_grad_diff_th_x1 = online_init_rule_base_grad_diff_th_x1;
	this->_online_init_rule_base_depth_y_error_th_y3 = online_init_rule_base_depth_y_error_th_y3;
	this->_online_init_rule_base_depth_y_error_th_y2 = online_init_rule_base_depth_y_error_th_y2;
	this->_online_init_rule_base_depth_y_error_th_y1 = online_init_rule_base_depth_y_error_th_y1;
	this->_online_init_rule_base_depth_y_error_th_x3 = online_init_rule_base_depth_y_error_th_x3;
	this->_online_init_rule_base_depth_y_error_th_x2 = online_init_rule_base_depth_y_error_th_x2;
	this->_online_init_rule_base_depth_y_error_th_x1 = online_init_rule_base_depth_y_error_th_x1;
	this->_online_init_rule_base_depth_y_error_th_min = online_init_rule_base_depth_y_error_th_min;
	
	this->_online_rule_base_grad_max = this->_online_init_rule_base_grad_max;
	this->_online_rule_base_grad_min = this->_online_init_rule_base_grad_min;
	this->_online_rule_base_grad_diff_th_y3 = this->_online_init_rule_base_grad_diff_th_y3;
	this->_online_rule_base_grad_diff_th_y2 = this->_online_init_rule_base_grad_diff_th_y2;
	this->_online_rule_base_grad_diff_th_y1 = this->_online_init_rule_base_grad_diff_th_y1;
	this->_online_rule_base_grad_diff_th_x3 = this->_online_init_rule_base_grad_diff_th_x3;
	this->_online_rule_base_grad_diff_th_x2 = this->_online_init_rule_base_grad_diff_th_x2;
	this->_online_rule_base_grad_diff_th_x1 = this->_online_init_rule_base_grad_diff_th_x1;
	this->_online_rule_base_depth_y_error_th_y3 = this->_online_init_rule_base_depth_y_error_th_y3;
	this->_online_rule_base_depth_y_error_th_y2 = this->_online_init_rule_base_depth_y_error_th_y2;
	this->_online_rule_base_depth_y_error_th_y1 = this->_online_init_rule_base_depth_y_error_th_y1;
	this->_online_rule_base_depth_y_error_th_x3 = this->_online_init_rule_base_depth_y_error_th_x3;
	this->_online_rule_base_depth_y_error_th_x2 = this->_online_init_rule_base_depth_y_error_th_x2;
	this->_online_rule_base_depth_y_error_th_x1 = this->_online_init_rule_base_depth_y_error_th_x1;
	this->_online_rule_base_depth_y_error_th_min = this->_online_init_rule_base_depth_y_error_th_min;

	this->_preproc_resize_height = preproc_resize_height;
	this->_preproc_resize_width = preproc_resize_width;	

	if(no_queue_online>ARRAY_SIZE_FOR_ONLINE_LEARNING)
	{
		ROS_ERROR("number_of_queue_for_online_learning should be less than %d",ARRAY_SIZE_FOR_ONLINE_LEARNING);
	}

	this->_no_queue_online = no_queue_online;
	this->_probability_discard_th = probability_discard_th;
	this->_probability_scale_del_beta = probability_scale_del_beta;
	this->_probability_scale_del_gamma = probability_scale_del_gamma;
	this->_probability_scale_del_x = probability_scale_del_x;
	this->_probability_scale_del_y = probability_scale_del_y;
	this->_online_detect_flag_reset_timer = online_detect_flag_reset_timer;
	
	this->str_det_cost_func = new STAIR_DETEC_COST_FUNC( false, false);
}

STAIR_DETEC_ONLINE_FUNC::~STAIR_DETEC_ONLINE_FUNC()
{
	std::cout<<" online_init_rule_base_grad_diff_th_y1 : "<< this->_online_rule_base_grad_diff_th_y1<<std::endl;
	std::cout<<" online_init_rule_base_grad_diff_th_y2 : "<< this->_online_rule_base_grad_diff_th_y2<<std::endl;
	std::cout<<" online_init_rule_base_grad_diff_th_y3 : "<< this->_online_rule_base_grad_diff_th_y3<<std::endl;
	
	std::cout<<" online_init_rule_base_depth_y_error_th_y1 : "<< this->_online_rule_base_depth_y_error_th_y1<<std::endl;
	std::cout<<" online_init_rule_base_depth_y_error_th_y2 : "<< this->_online_rule_base_depth_y_error_th_y2<<std::endl;
	std::cout<<" online_init_rule_base_depth_y_error_th_y3 : "<< this->_online_rule_base_depth_y_error_th_y3<<std::endl;
	


}

bool STAIR_DETEC_ONLINE_FUNC::rule_base_detect(const cv::Mat& rgb_input, const cv::Mat& depth_input, std::vector<std::vector<float>>* vector_set_for_learning, float focal_len_x, float opt_center_x, float focal_len_y, float opt_center_y, float depth_scale)
{
	this->_fx = focal_len_x;
	this->_px = opt_center_x;
	this->_fy = focal_len_y;
	this->_py = opt_center_y;
	this->_dscale = depth_scale;
	
	std::vector<std::vector<float>>::iterator it;

	float gradient;
	float gradient_diff;
	float depth_y_error;
	float x_avg_error;
	float x_coor_cam_frame_tmp;
	float y_coor_cam_frame_tmp;
	float z_coor_cam_frame_tmp;
	float center_depth;
    float center_x_coor;
	float center_y_coor;
	float x_center_pixel;
	float y_center_pixel;
	float roi_size;

	float del_beta;
	float del_gamma;
	float del_x;
	float del_y;

	float continuity_factor_th_interp;
	float deviation_cost_th_interp;

	bool gradient_range_ok = false;
	bool gradient_condition_ok = false;
	bool depth_y_error_condition_ok = false;
	bool avg_x_error_condition_ok = false;
	bool stair_case_detc_flag = false;
	
	
	 bool falsepos_diag_ = true;

	int i=0;
	/*vector_for_learning include 8 elements*/
	/* stair gradient , least square error(x coordinate of map frame (depth) and z coordinate of map frame(rows)) per line,  average error (y coordinate of map frame (cols)), center point x (cam frame), center point y(cam frame), depth, center point x pixel, center point y pixel*/

	if(this->_activate_flag == true)
	{
	   ros::Duration elapse_time = ros::Time::now() - this->_detect_init_time;

	   if(elapse_time>ros::Duration(this->_online_detect_flag_reset_timer))
	   {
		   this->_activate_flag = false;
		   this->_array_full_flag_bt = false;
		   this->_array_full_flag_gmm = false;
		   this->_array_indicator_bt = 0;
		   this->_array_indicator_gmm = 0;
		   
		   int j;
		   
		   for(j=0;j<this->_no_queue_online;j++)
		   {
			   _array_del_beta_bt_adp[i] = 0;
			   _array_del_gamma_bt_adp[i] = 0;
			   _array_prob_del_beta_bt_adp[i] = 0;
			   _array_prob_del_gamma_bt_adp[i] = 0;
			   _array_prob_del_x_bt_adp[i] = 0;
			   _array_prob_del_y_bt_adp[i] = 0;
			   
			   _array_del_beta_gmm_adp[i] = 0;
			   _array_del_gamma_gmm_adp[i] = 0;
			   _array_prob_del_beta_gmm_adp[i] = 0;
			   _array_prob_del_gamma_gmm_adp[i] = 0;
			   _array_prob_del_x_gmm_adp[i] = 0;
			   _array_prob_del_y_gmm_adp[i] = 0;
		   }
		   
		   std::cout<<"activate flag reset "<<std::endl;
		   std::cout<<"activate flag reset "<<std::endl;
		   std::cout<<"activate flag reset "<<std::endl;
	   }
	}


	for(it=vector_set_for_learning->begin(); it!=vector_set_for_learning->end(); it++)
	{
		gradient = it->at(0);
		gradient_diff = it->at(1);
		//depth_y_error = (it->at(2))*10;
		depth_y_error = it->at(2);
        center_x_coor = it->at(4);
		center_y_coor = it->at(5);
		center_depth = it->at(6);
		x_center_pixel = it->at(7);
		y_center_pixel = it->at(8);
		roi_size = it ->at(9);

		if((gradient>3)||(std::abs(gradient_diff)>2)||(depth_y_error>10))
		{
			continue;
		}

		continuity_factor_th_interp=interpol3(gradient,this->_online_rule_base_grad_diff_th_x1,this->_online_rule_base_grad_diff_th_x2,this->_online_rule_base_grad_diff_th_x3,
		                                                                       this->_online_rule_base_grad_diff_th_y1,this->_online_rule_base_grad_diff_th_y2,this->_online_rule_base_grad_diff_th_y3);
		deviation_cost_th_interp=interpol3(gradient, this->_online_rule_base_depth_y_error_th_x1, this->_online_rule_base_depth_y_error_th_x2, this->_online_rule_base_depth_y_error_th_x3,
		                                                                                               this->_online_rule_base_depth_y_error_th_y1, this->_online_rule_base_depth_y_error_th_y2, this->_online_rule_base_depth_y_error_th_y3);

		if((gradient<this->_online_rule_base_grad_max)&&(gradient>this->_online_rule_base_grad_min))
		{
			gradient_range_ok = true;
		}
		else
		{
			gradient_range_ok = false;
		}


		if((gradient_range_ok == true)&&(gradient_diff<continuity_factor_th_interp))
		{
			gradient_condition_ok = true;
		}
		else
		{
			gradient_condition_ok = false;
		}

         if((gradient_range_ok==true)&&(depth_y_error<deviation_cost_th_interp)&&(depth_y_error>_online_rule_base_depth_y_error_th_min))
		{
			depth_y_error_condition_ok = true;
		}
        else
		{
            depth_y_error_condition_ok = false;
		}
		
		if((gradient_range_ok == true)&&(gradient_condition_ok == false))
		{
			del_gamma = gradient_diff - continuity_factor_th_interp;
		}
		else if(gradient_range_ok == false)
		{
			del_gamma = 5;
		}
		else /*gradient_range_ok == true  and gradient_condition_ok == true */
		{
			del_gamma = 0;
		}
		
		
		if((gradient_range_ok == true)&&(depth_y_error_condition_ok == false)&&(depth_y_error>_online_rule_base_depth_y_error_th_min))
		{
			del_beta = depth_y_error - deviation_cost_th_interp;
		}
		else if((gradient_range_ok == false)||(depth_y_error<_online_rule_base_depth_y_error_th_min))
		{
			del_beta = 5;
		}
		else /*gradient_range_ok == true and depth_y_error_condition_ok == true and depth_y_error>_online_rule_base_depth_y_error_th_min*/
		{
			del_beta = 0;
		}


		if((gradient_condition_ok==true)&&(depth_y_error_condition_ok==true))
		{
			
			
			bool false_pos_flag = reject_false_positive(depth_input, center_x_coor, center_y_coor, center_depth, x_center_pixel, y_center_pixel, roi_size);
			

           

			if(falsepos_diag_)
			{
			   cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;

			   rectangle_tmp1_pt1.x =x_center_pixel-30;
		       rectangle_tmp1_pt1.y =y_center_pixel+30;

		       rectangle_tmp1_pt2.x =x_center_pixel+30;
		       rectangle_tmp1_pt2.y =y_center_pixel-30;

			   cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);


		       cv::imshow("stair case ROI", rgb_input);
	           cv::waitKey(0);
			}
			
			
			if(1/*false_pos_flag == false*/)
			{
			   stair_case_detc_flag = true;
			
			   x_coor_cam_frame_tmp = it->at(4);
			   y_coor_cam_frame_tmp = it->at(5);
			   z_coor_cam_frame_tmp = it->at(6);

			   this->_detected_roi_x = x_center_pixel;
			   this->_detected_roi_y = y_center_pixel;
			
			   this->_detect_init_time = ros::Time::now();
			   //this->x_coor_cam_frame_vec.push_back(x_coor_cam_frame_tmp);
			   //this->y_coor_cam_frame_vec.push_back(y_coor_cam_frame_tmp);
			   //this->z_coor_cam_frame_vec.push_back(z_coor_cam_frame_tmp);
			   //this->slope_n_pixel.push_back(0);

			   //this->row_center_vec.push_back(x_center_pixel);
			   //this->column_center_vec.push_back(y_center_pixel);

			   // if(this->_use_only_rgb_to_detect_staircase == true)
			   // {
			   //	this->depth_info_vec.push_back(0);
			   // }
			   // else
               //{
			   //	this->depth_info_vec.push_back(z_coor_cam_frame_tmp);
			   // }

			   //this->debug_gradient.push_back(gradient);
			   //this->debug_gradient_diff.push_back(gradient_diff);
			   //this->debug_depth_y_error.push_back(depth_y_error);

               //if(roi_size <101)
			   // {
		 	   //   _detected_data_write_roi_100<<gradient<<" "<<gradient_diff<<" "<<depth_y_error<<" "<<x_center_pixel<<" "<<y_center_pixel<<std::endl;
			   // }
			   //else
			   // {
			   //    _detected_data_write_roi_200<<gradient<<" "<<gradient_diff<<" "<<depth_y_error<<" "<<x_center_pixel<<" "<<y_center_pixel<<std::endl;
			   // }

			   //break;
						
			this->_activate_flag = stair_case_detc_flag;
			}
		}
		else
		{
			stair_case_detc_flag = false;

			if((this->_activate_flag == true)&&(gradient_range_ok == true))
			{
                del_x = std::abs(this->_detected_roi_x  - x_center_pixel);
				del_y = std::abs(this->_detected_roi_y  - y_center_pixel);
				
				//std::cout<<"gradient_range_ok : "<<gradient_range_ok<<"  gradient_condition_ok : "<<gradient_condition_ok<<"  del_gamma : "<<del_gamma<<std::endl;
				//std::cout<<"gradient_range_ok : "<<gradient_range_ok<<"  depth_y_error_condition_ok : "<<depth_y_error_condition_ok<<"  del_beta : "<<del_beta<<std::endl;
				//std::cout<<"this->_detected_roi_x : "<<this->_detected_roi_x<<"  x_center_pixel : "<<x_center_pixel<<"  del_x : "<<del_x<<std::endl;
				//std::cout<<"this->_detected_roi_y : "<<this->_detected_roi_y<<"  y_center_pixel : "<<y_center_pixel<<"  del_y : "<<del_y<<std::endl;
				
				if(falsepos_diag_ == false)
				{
			       threshold_update(del_beta,del_gamma,del_x,del_y,gradient);
				}
			}

		}

	   //if(stair_case_detc_flag==true)
	   //{
		   
	   //	cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;

	   //	rectangle_tmp1_pt1.x =x_center_pixel-30;
	   //    rectangle_tmp1_pt1.y =y_center_pixel+30;

	   //    rectangle_tmp1_pt2.x =x_center_pixel+30;
	   //    rectangle_tmp1_pt2.y =y_center_pixel-30;

	   //	 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);

	   //}

		i++;
	}
	
	
	if(stair_case_detc_flag == true)
	{
	   cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;

	   rectangle_tmp1_pt1.x =x_center_pixel-30;
	   rectangle_tmp1_pt1.y =y_center_pixel+30;

	   rectangle_tmp1_pt2.x =x_center_pixel+30;
	   rectangle_tmp1_pt2.y =y_center_pixel-30;

	   cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);


	   cv::imshow("stair case ROI", rgb_input);
	   cv::waitKey(0);
	}
	
	
		 
	 return stair_case_detc_flag;


}


void STAIR_DETEC_ONLINE_FUNC::threshold_update(float del_beta, float del_gamma, float del_x, float del_y, float gradient)
{
     float del_beta_prob_input = -this->_probability_scale_del_beta*del_beta;
	 float del_gamma_prob_input = -this->_probability_scale_del_gamma*del_gamma;
	 float del_x_prob_input = -this->_probability_scale_del_x*del_x;
	 float del_y_prob_input = -this->_probability_scale_del_y*del_y;
	 
	 float del_beta_prob = std::exp(del_beta_prob_input);
	 float del_gamma_prob = std::exp(del_gamma_prob_input);
	 float del_x_prob = std::exp(del_x_prob_input);
	 float del_y_prob = std::exp(del_y_prob_input);
	 
	 //std::cout<<"  del_beta : "<<del_beta<<"  this->_probability_scale_del_beta : "<<this->_probability_scale_del_beta<<"  del_beta_prob_input : "<<del_beta_prob_input<<"  del_beta_prob : "<<del_beta_prob<<std::endl;
	 //std::cout<<"  del_gamma : "<<del_gamma<<"  this->_probability_scale_del_gamma : "<<this->_probability_scale_del_gamma<<"  del_gamma_prob_input : "<<del_gamma_prob_input<<"  del_gamma_prob : "<<del_gamma_prob<<std::endl;
	 //std::cout<<"  del_x : "<<del_x<<"  this->_probability_scale_del_x : "<<this->_probability_scale_del_x<<"  del_x_prob_input : "<<del_x_prob_input<<"  del_x_prob : "<<del_x_prob<<std::endl;
	// std::cout<<"  del_y : "<<del_y<<"  this->_probability_scale_del_y : "<<this->_probability_scale_del_y<<"  del_y_prob_input : "<<del_y_prob_input<<"  del_y_prob : "<<del_y_prob<<std::endl;
	 
	 if((del_beta_prob<this->_probability_discard_th)||(del_gamma_prob<this->_probability_discard_th)||(del_x_prob<this->_probability_discard_th)||(del_y_prob<this->_probability_discard_th))
	 {
		 return;
	 }
	 else
	 {
		 if(del_beta_prob<1)
		 {
			 //std::cout<<"**************** del beta inputted to array *********************************"<<std::endl;
			 //std::cout<<"_array_indicator_bt : "<<this->_array_indicator_bt<<std::endl;
			 //std::cout<<"gradient : "<<gradient<<std::endl;
			 //std::cout<<"  del_beta : "<<del_beta<<"  this->_probability_scale_del_beta : "<<this->_probability_scale_del_beta<<"  del_beta_prob_input : "<<del_beta_prob_input<<"  del_beta_prob : "<<del_beta_prob<<std::endl;
	         //std::cout<<"  del_gamma : "<<del_gamma<<"  this->_probability_scale_del_gamma : "<<this->_probability_scale_del_gamma<<"  del_gamma_prob_input : "<<del_gamma_prob_input<<"  del_gamma_prob : "<<del_gamma_prob<<std::endl;
	         //std::cout<<"  del_x : "<<del_x<<"  this->_probability_scale_del_x : "<<this->_probability_scale_del_x<<"  del_x_prob_input : "<<del_x_prob_input<<"  del_x_prob : "<<del_x_prob<<std::endl;
	         //std::cout<<"  del_y : "<<del_y<<"  this->_probability_scale_del_y : "<<this->_probability_scale_del_y<<"  del_y_prob_input : "<<del_y_prob_input<<"  del_y_prob : "<<del_y_prob<<std::endl;
			 
			 
			 this->_array_del_beta_bt_adp[this->_array_indicator_bt] = del_beta;
			 this->_array_del_gamma_bt_adp[this->_array_indicator_bt] = del_gamma;
			 this->_array_gradient_bt_adp[this->_array_indicator_bt] = gradient;
			 this->_array_prob_del_beta_bt_adp[this->_array_indicator_bt] = del_beta_prob;
			 this->_array_prob_del_gamma_bt_adp[this->_array_indicator_bt] = del_gamma_prob;
			 this->_array_prob_del_x_bt_adp[this->_array_indicator_bt] = del_x_prob;
			 this->_array_prob_del_y_bt_adp[this->_array_indicator_bt] = del_y_prob;
			 
			 this->_array_indicator_bt = this->_array_indicator_bt + 1;
			 
			 if(this->_array_indicator_bt == this->_no_queue_online)
			 {
				 this->_array_indicator_bt = 0;
				 this->_array_full_flag_bt = true;
			 }
			 
		 }
		 
		 if(del_gamma_prob<1)
		 {
			 
			 //std::cout<<"**************** del gamma inputted to array *********************************"<<std::endl;
			 //std::cout<<"_array_indicator_gmm : "<<this->_array_indicator_gmm<<std::endl;
			 //std::cout<<"gradient : "<<gradient<<std::endl;
			 //std::cout<<"  del_beta : "<<del_beta<<"  this->_probability_scale_del_beta : "<<this->_probability_scale_del_beta<<"  del_beta_prob_input : "<<del_beta_prob_input<<"  del_beta_prob : "<<del_beta_prob<<std::endl;
	         //std::cout<<"  del_gamma : "<<del_gamma<<"  this->_probability_scale_del_gamma : "<<this->_probability_scale_del_gamma<<"  del_gamma_prob_input : "<<del_gamma_prob_input<<"  del_gamma_prob : "<<del_gamma_prob<<std::endl;
	         //std::cout<<"  del_x : "<<del_x<<"  this->_probability_scale_del_x : "<<this->_probability_scale_del_x<<"  del_x_prob_input : "<<del_x_prob_input<<"  del_x_prob : "<<del_x_prob<<std::endl;
	         //std::cout<<"  del_y : "<<del_y<<"  this->_probability_scale_del_y : "<<this->_probability_scale_del_y<<"  del_y_prob_input : "<<del_y_prob_input<<"  del_y_prob : "<<del_y_prob<<std::endl;
			 
			 this->_array_del_beta_gmm_adp[this->_array_indicator_gmm] = del_beta;
			 this->_array_del_gamma_gmm_adp[this->_array_indicator_gmm] = del_gamma;
			 this->_array_gradient_gmm_adp[this->_array_indicator_gmm] = gradient;
			 this->_array_prob_del_beta_gmm_adp[this->_array_indicator_gmm] = del_beta_prob;
			 this->_array_prob_del_gamma_gmm_adp[this->_array_indicator_gmm] = del_gamma_prob;
			 this->_array_prob_del_x_gmm_adp[this->_array_indicator_gmm] = del_x_prob;
			 this->_array_prob_del_y_gmm_adp[this->_array_indicator_gmm] = del_y_prob;
			 
			 
			 this->_array_indicator_gmm = this->_array_indicator_gmm + 1;
			 
			 if(this->_array_indicator_gmm == this->_no_queue_online)
			 {
				 this->_array_indicator_gmm = 0;
				 this->_array_full_flag_gmm = true;
			 }
		 }
	 }

	 

	 
	 if(this->_array_full_flag_bt == true)
	 {
		 //std::cout<<" array check for beta threshold adjustment " << std::endl; 
	     //std::cout<<"  _array_del_beta_bt_adp -->> "<<" index 0 : "<<this->_array_del_beta_bt_adp[0]<<" index 1 : "<<this->_array_del_beta_bt_adp[1]<<" index 2 : "<<this->_array_del_beta_bt_adp[2]<<" index 3 : "<<this->_array_del_beta_bt_adp[3]<<" index 4 : "<<this->_array_del_beta_bt_adp[4]<<" index 5 : "<<this->_array_del_beta_bt_adp[5]<<std::endl;
	     //std::cout<<"  _array_del_gamma_bt_adp -->> "<<" index 0 : "<<this->_array_del_gamma_bt_adp[0]<<" index 1 : "<<this->_array_del_gamma_bt_adp[1]<<" index 2 : "<<this->_array_del_gamma_bt_adp[2]<<" index 3 : "<<this->_array_del_gamma_bt_adp[3]<<" index 4 : "<<this->_array_del_gamma_bt_adp[4]<<" index 5 : "<<this->_array_del_gamma_bt_adp[5]<<std::endl;
	     //std::cout<<"  _array_gradient_bt_adp -->> "<<" index 0 : "<<this->_array_gradient_bt_adp[0]<<" index 1 : "<<this->_array_gradient_bt_adp[1]<<" index 2 : "<<this->_array_gradient_bt_adp[2]<<" index 3 : "<<this->_array_gradient_bt_adp[3]<<" index 4 : "<<this->_array_gradient_bt_adp[4]<<" index 5 : "<<this->_array_gradient_bt_adp[5]<<std::endl;
	     //std::cout<<"  _array_prob_del_beta_bt_adp -->> "<<" index 0 : "<<this->_array_prob_del_beta_bt_adp[0]<<" index 1 : "<<this->_array_prob_del_beta_bt_adp[1]<<" index 2 : "<<this->_array_prob_del_beta_bt_adp[2]<<" index 3 : "<<this->_array_prob_del_beta_bt_adp[3]<<" index 4 : "<<this->_array_prob_del_beta_bt_adp[4]<<" index 5 : "<<this->_array_prob_del_beta_bt_adp[5]<<std::endl;
         //std::cout<<"  _array_prob_del_gamma_bt_adp -->> "<<" index 0 : "<<this->_array_prob_del_gamma_bt_adp[0]<<" index 1 : "<<this->_array_prob_del_gamma_bt_adp[1]<<" index 2 : "<<this->_array_prob_del_gamma_bt_adp[2]<<" index 3 : "<<this->_array_prob_del_gamma_bt_adp[3]<<" index 4 : "<<this->_array_prob_del_gamma_bt_adp[4]<<" index 5 : "<<this->_array_prob_del_gamma_bt_adp[5]<<std::endl;
	     //std::cout<<"  _array_prob_del_x_bt_adp -->> "<<" index 0 : "<<this->_array_prob_del_x_bt_adp[0]<<" index 1 : "<<this->_array_prob_del_x_bt_adp[1]<<" index 2 : "<<this->_array_prob_del_x_bt_adp[2]<<" index 3 : "<<this->_array_prob_del_x_bt_adp[3]<<" index 4 : "<<this->_array_prob_del_x_bt_adp[4]<<" index 5 : "<<this->_array_prob_del_x_bt_adp[5]<<std::endl;
	     //std::cout<<"  _array_prob_del_y_bt_adp -->> "<<" index 0 : "<<this->_array_prob_del_y_bt_adp[0]<<" index 1 : "<<this->_array_prob_del_y_bt_adp[1]<<" index 2 : "<<this->_array_prob_del_y_bt_adp[2]<<" index 3 : "<<this->_array_prob_del_y_bt_adp[3]<<" index 4 : "<<this->_array_prob_del_y_bt_adp[4]<<" index 5 : "<<this->_array_prob_del_y_bt_adp[5]<<std::endl;
	     //std::cout<<" array check for beta threshold adjustment finished " << std::endl;
		 
		 
		 std::cout<<"********************************** beta threshold update start ****************************** " << std::endl;
		 float denom_for_wbt = 0;
		 float beta_inc = 0;
		 float gradient_for_beta_inc = 0;
		 int i = 0;
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 denom_for_wbt = denom_for_wbt + this->_array_prob_del_beta_bt_adp[i] + this->_array_prob_del_gamma_bt_adp[i];
		 }
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 beta_inc = beta_inc + (this->_array_prob_del_beta_bt_adp[i] + this->_array_prob_del_gamma_bt_adp[i])/denom_for_wbt*this->_array_del_beta_bt_adp[i];
			 gradient_for_beta_inc = gradient_for_beta_inc + (this->_array_prob_del_beta_bt_adp[i] + this->_array_prob_del_gamma_bt_adp[i])/denom_for_wbt*this->_array_gradient_bt_adp[i];
		 }
		 
		 std::cout<<"beta_inc"<<beta_inc << std::endl;
		 std::cout<<"gradient_for_beta_inc"<<gradient_for_beta_inc << std::endl;
		 /*threshold update code should be inserted at here*/
		 
		 float  left_point_for_inter, right_point_for_inter;
		 
		 if(gradient_for_beta_inc<=this->_online_rule_base_depth_y_error_th_x1)
		 {
			 this->_online_rule_base_depth_y_error_th_y1 = this->_online_rule_base_depth_y_error_th_y1 + beta_inc;
		 }
		 else if((gradient_for_beta_inc>this->_online_rule_base_depth_y_error_th_x1)&&(gradient_for_beta_inc<=this->_online_rule_base_depth_y_error_th_x2))
		 {
			 left_point_for_inter = this->_online_rule_base_depth_y_error_th_x1;
			 right_point_for_inter = this->_online_rule_base_depth_y_error_th_x2;
			 
			 this->_online_rule_base_depth_y_error_th_y1 = this->_online_rule_base_depth_y_error_th_y1 + (gradient_for_beta_inc - left_point_for_inter)/(right_point_for_inter - left_point_for_inter)*beta_inc;
			 this->_online_rule_base_depth_y_error_th_y2 = this->_online_rule_base_depth_y_error_th_y2 + (right_point_for_inter - gradient_for_beta_inc)/(right_point_for_inter - left_point_for_inter)*beta_inc;
		 }
		 else if((gradient_for_beta_inc>this->_online_rule_base_depth_y_error_th_x2)&&(gradient_for_beta_inc<=this->_online_rule_base_depth_y_error_th_x3))
		 {
			 left_point_for_inter = this->_online_rule_base_depth_y_error_th_x2;
			 right_point_for_inter = this->_online_rule_base_depth_y_error_th_x3;

			 this->_online_rule_base_depth_y_error_th_y2 = this->_online_rule_base_depth_y_error_th_y2 + (gradient_for_beta_inc - left_point_for_inter)/(right_point_for_inter - left_point_for_inter)*beta_inc;
			 this->_online_rule_base_depth_y_error_th_y3 = this->_online_rule_base_depth_y_error_th_y3 + (right_point_for_inter - gradient_for_beta_inc)/(right_point_for_inter - left_point_for_inter)*beta_inc;			 
		 }
		 else /*gradient_for_beta_inc>this->_online_rule_base_depth_y_error_th_x3*/
		 {
			 this->_online_rule_base_depth_y_error_th_y3 = this->_online_rule_base_depth_y_error_th_y3 + beta_inc;
		 }
		 
		 /*limitation of the beta (depth y error) threshold */
		 
		 if(this->_online_rule_base_depth_y_error_th_y1>2)
		 {
			 this->_online_rule_base_depth_y_error_th_y1 = 2;
		 }
		 if(this->_online_rule_base_depth_y_error_th_y2>3)
		 {
			 this->_online_rule_base_depth_y_error_th_y2 = 3;
		 }
		 if(this->_online_rule_base_depth_y_error_th_y3>4)
		 {
			 this->_online_rule_base_depth_y_error_th_y3 = 4;
		 }		 
		 
		 
		 
		 /* update the del beta and beta probability queue with new beta threshold for online learning */
		  for(i=0;i<this->_no_queue_online;i++)
		  {
			  float del_beta_prob_input_update;
			  float del_beta_prob_update;
			  
			 this->_array_del_beta_bt_adp[i] = this->_array_del_beta_bt_adp[i] - 0.7*beta_inc;
			 
			 del_beta_prob_input_update = -this->_probability_scale_del_beta*this->_array_del_beta_bt_adp[i];
			 del_beta_prob_update = std::exp(del_beta_prob_input_update);
			 
			 this->_array_prob_del_beta_bt_adp[i] = del_beta_prob_update;
			 
			 if(this->_array_del_beta_bt_adp[i] < 0)
			 {
				 this->_array_del_beta_bt_adp[i] = 0;
				 this->_array_prob_del_beta_bt_adp[i] = 0;
			 }
			 
			 this->_array_del_beta_gmm_adp[i] = this->_array_del_beta_gmm_adp[i] - 0.7*beta_inc;
			 
			 del_beta_prob_input_update = -this->_probability_scale_del_beta*this->_array_del_beta_gmm_adp[i];
			 del_beta_prob_update = std::exp(del_beta_prob_input_update);
			 
			 this->_array_prob_del_beta_gmm_adp[i] = del_beta_prob_update;

			 if(this->_array_del_beta_gmm_adp[i] < 0)
			 {
				 this->_array_del_beta_gmm_adp[i] = 0;
				 this->_array_prob_del_beta_gmm_adp[i] = 0;
			 }
			 
		  }
		 
		 this->_array_full_flag_bt = false;
		 
	 }

	 if(this->_array_full_flag_gmm == true)
	 {
		 //std::cout<<" array check for gamma threshold adjustment " << std::endl; 
	     //std::cout<<"  _array_del_beta_gmm_adp -->> "<<" index 0 : "<<this->_array_del_beta_gmm_adp[0]<<" index 1 : "<<this->_array_del_beta_gmm_adp[1]<<" index 2 : "<<this->_array_del_beta_gmm_adp[2]<<" index 3 : "<<this->_array_del_beta_gmm_adp[3]<<" index 4 : "<<this->_array_del_beta_gmm_adp[4]<<" index 5 : "<<this->_array_del_beta_gmm_adp[5]<<std::endl;
	     //std::cout<<"  _array_del_gamma_gmm_adp -->> "<<" index 0 : "<<this->_array_del_gamma_gmm_adp[0]<<" index 1 : "<<this->_array_del_gamma_gmm_adp[1]<<" index 2 : "<<this->_array_del_gamma_gmm_adp[2]<<" index 3 : "<<this->_array_del_gamma_gmm_adp[3]<<" index 4 : "<<this->_array_del_gamma_gmm_adp[4]<<" index 5 : "<<this->_array_del_gamma_gmm_adp[5]<<std::endl;
	     //std::cout<<"  _array_gradient_gmm_adp -->> "<<" index 0 : "<<this->_array_gradient_gmm_adp[0]<<" index 1 : "<<this->_array_gradient_gmm_adp[1]<<" index 2 : "<<this->_array_gradient_gmm_adp[2]<<" index 3 : "<<this->_array_gradient_gmm_adp[3]<<" index 4 : "<<this->_array_gradient_gmm_adp[4]<<" index 5 : "<<this->_array_gradient_gmm_adp[5]<<std::endl;
	     //std::cout<<"  _array_prob_del_beta_gmm_adp -->> "<<" index 0 : "<<this->_array_prob_del_beta_gmm_adp[0]<<" index 1 : "<<this->_array_prob_del_beta_gmm_adp[1]<<" index 2 : "<<this->_array_prob_del_beta_gmm_adp[2]<<" index 3 : "<<this->_array_prob_del_beta_gmm_adp[3]<<" index 4 : "<<this->_array_prob_del_beta_gmm_adp[4]<<" index 5 : "<<this->_array_prob_del_beta_gmm_adp[5]<<std::endl;
         //std::cout<<"  _array_prob_del_gamma_gmm_adp -->> "<<" index 0 : "<<this->_array_prob_del_gamma_gmm_adp[0]<<" index 1 : "<<this->_array_prob_del_gamma_gmm_adp[1]<<" index 2 : "<<this->_array_prob_del_gamma_gmm_adp[2]<<" index 3 : "<<this->_array_prob_del_gamma_gmm_adp[3]<<" index 4 : "<<this->_array_prob_del_gamma_gmm_adp[4]<<" index 5 : "<<this->_array_prob_del_gamma_gmm_adp[5]<<std::endl;
	     //std::cout<<"  _array_prob_del_x_gmm_adp -->> "<<" index 0 : "<<this->_array_prob_del_x_gmm_adp[0]<<" index 1 : "<<this->_array_prob_del_x_gmm_adp[1]<<" index 2 : "<<this->_array_prob_del_x_gmm_adp[2]<<" index 3 : "<<this->_array_prob_del_x_gmm_adp[3]<<" index 4 : "<<this->_array_prob_del_x_gmm_adp[4]<<" index 5 : "<<this->_array_prob_del_x_gmm_adp[5]<<std::endl;
	     //std::cout<<"  _array_prob_del_y_gmm_adp -->> "<<" index 0 : "<<this->_array_prob_del_y_gmm_adp[0]<<" index 1 : "<<this->_array_prob_del_y_gmm_adp[1]<<" index 2 : "<<this->_array_prob_del_y_gmm_adp[2]<<" index 3 : "<<this->_array_prob_del_y_gmm_adp[3]<<" index 4 : "<<this->_array_prob_del_y_gmm_adp[4]<<" index 5 : "<<this->_array_prob_del_y_gmm_adp[5]<<std::endl;
	     //std::cout<<" array check for gamma threshold adjustment finished " << std::endl;
		 
		 
		 std::cout<<"********************************** gaama threshold update start ****************************** " << std::endl;
		 float denom_for_wgt = 0;
		 float gamma_inc = 0;
		 float gradient_for_gamma_inc = 0;
		 int i = 0;
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 denom_for_wgt = denom_for_wgt + this->_array_prob_del_beta_gmm_adp[i] + this->_array_prob_del_gamma_gmm_adp[i];
		 }
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 gamma_inc = gamma_inc + (this->_array_prob_del_beta_gmm_adp[i] + this->_array_prob_del_gamma_gmm_adp[i])/denom_for_wgt*this->_array_del_beta_gmm_adp[i];
			 gradient_for_gamma_inc = gradient_for_gamma_inc + (this->_array_prob_del_beta_gmm_adp[i] + this->_array_prob_del_gamma_gmm_adp[i])/denom_for_wgt*this->_array_gradient_gmm_adp[i]; 
		 }
		 
		 std::cout<<"gamma_inc"<<gamma_inc << std::endl;
		 std::cout<<"gradient_for_gamma_inc"<<gradient_for_gamma_inc << std::endl;
		 
		 float  left_point_for_inter, right_point_for_inter;
		 
		 /*threshold update code should be inserted at here*/
		 if(gradient_for_gamma_inc<=this->_online_rule_base_grad_diff_th_x1)
		 {
			 this->_online_rule_base_grad_diff_th_y1 = this->_online_rule_base_grad_diff_th_y1 + gamma_inc;
		 }
		 else if((gradient_for_gamma_inc>this->_online_rule_base_grad_diff_th_x1)&&(gradient_for_gamma_inc<=this->_online_rule_base_grad_diff_th_x2))
		 {
			 left_point_for_inter = this->_online_rule_base_grad_diff_th_x1;
			 right_point_for_inter = this->_online_rule_base_grad_diff_th_x2;
			 
			 this->_online_rule_base_grad_diff_th_y1 = this->_online_rule_base_grad_diff_th_y1 + (gradient_for_gamma_inc - left_point_for_inter)/(right_point_for_inter - left_point_for_inter)*gamma_inc;
			 this->_online_rule_base_grad_diff_th_y2 = this->_online_rule_base_grad_diff_th_y2 + (right_point_for_inter - gradient_for_gamma_inc)/(right_point_for_inter - left_point_for_inter)*gamma_inc;
		 }
		 else if((gradient_for_gamma_inc>this->_online_rule_base_grad_diff_th_x2)&&(gradient_for_gamma_inc<=this->_online_rule_base_grad_diff_th_x3))
		 {
			 left_point_for_inter = this->_online_rule_base_grad_diff_th_x2;
			 right_point_for_inter = this->_online_rule_base_grad_diff_th_x3;

			 this->_online_rule_base_grad_diff_th_y2 = this->_online_rule_base_grad_diff_th_y2 + (gradient_for_gamma_inc - left_point_for_inter)/(right_point_for_inter - left_point_for_inter)*gamma_inc;
			 this->_online_rule_base_grad_diff_th_y3 = this->_online_rule_base_grad_diff_th_y3 + (right_point_for_inter - gradient_for_gamma_inc)/(right_point_for_inter - left_point_for_inter)*gamma_inc;			 
		 }
		 else /*gradient_for_gamma_inc>this->_online_rule_base_grad_diff_th_x3*/
		 {
			 this->_online_rule_base_grad_diff_th_y3 = this->_online_rule_base_grad_diff_th_y3 + gamma_inc;
		 }
		 
		 /*limitation of the gamma (gradient diff) threshold */
		 if(this->_online_rule_base_grad_diff_th_y1 >this->_online_rule_base_grad_diff_th_x1)
		 {
			 this->_online_rule_base_grad_diff_th_y1 = this->_online_rule_base_grad_diff_th_x1;
		 }
		 if(this->_online_rule_base_grad_diff_th_y2 >this->_online_rule_base_grad_diff_th_x2)
		 {
			 this->_online_rule_base_grad_diff_th_y2 = this->_online_rule_base_grad_diff_th_x2;
		 }
		 if(this->_online_rule_base_grad_diff_th_y3 >this->_online_rule_base_grad_diff_th_x3)
		 {
			 this->_online_rule_base_grad_diff_th_y3 = this->_online_rule_base_grad_diff_th_x3;
		 }		 
		 		 /* update the del gamma and gamma probability queue with new gamma threshold for online learning */
		  for(i=0;i<this->_no_queue_online;i++)
		  {
			  float del_gamma_prob_input_update;
			  float del_gamma_prob_update;
			  
			 this->_array_del_gamma_bt_adp[i] = this->_array_del_gamma_bt_adp[i] - 0.7*gamma_inc;
			 
			 del_gamma_prob_input_update = -this->_probability_scale_del_gamma*this->_array_del_gamma_bt_adp[i];
			 del_gamma_prob_update = std::exp(del_gamma_prob_input_update);
			 
			 this->_array_prob_del_gamma_bt_adp[i] = del_gamma_prob_update;
			 
			 if(this->_array_del_gamma_bt_adp[i] < 0)
			 {
				 this->_array_del_gamma_bt_adp[i] = 0;
				 this->_array_prob_del_gamma_bt_adp[i] = 0;
			 }
			 
			 this->_array_del_gamma_gmm_adp[i] = this->_array_del_gamma_gmm_adp[i] - 0.7*gamma_inc;
			 
			 del_gamma_prob_input_update = -this->_probability_scale_del_gamma*this->_array_del_gamma_gmm_adp[i];
			 del_gamma_prob_update = std::exp(del_gamma_prob_input_update);
			 
			 this->_array_prob_del_gamma_gmm_adp[i] = del_gamma_prob_update;

			 if(this->_array_del_gamma_gmm_adp[i] < 0)
			 {
				 this->_array_del_gamma_gmm_adp[i] = 0;
				 this->_array_prob_del_gamma_gmm_adp[i] = 0;
			 } 
			 
		  }
		  
		  this->_array_full_flag_gmm = false;
	
	 }	 
	 
}

bool STAIR_DETEC_ONLINE_FUNC::reject_false_positive(const cv::Mat& depth_input, float center_x_val, float center_y_val, float center_depth_val, float center_col, float center_row, float roi_size)
{
	float fp_del_beta_array[3];
	float fp_del_beta_prob_array[3];
	float fp_del_gamma_array[3];
	float fp_del_gamma_prob_array[3];
	
	float continuity_factor_th_interp;
	float deviation_cost_th_interp;
	
	cv::Mat resized_depth_input, resized_depth_input_32f;
	
	cv::resize(depth_input, resized_depth_input, cv::Size(this->_preproc_resize_width,this->_preproc_resize_height), cv::INTER_AREA);
	
	if(resized_depth_input.type() == CV_16UC1)   resized_depth_input.convertTo(resized_depth_input_32f, CV_32F, this->_dscale);
    else if(resized_depth_input.type() == CV_32F)   resized_depth_input_32f = resized_depth_input.clone();
	
	
	
	int validation_count = 0;
	
	float center_y_val_wcoor = -center_y_val;   /*center_y_val hs (+) sign when the cooridnate is toward the ground so (-) sign is required for world coordinate sign convention */
	float y_coorthresh = 0.5*this->_py*center_depth_val / this->_fy;  /*threshod to determine if the detected staircase position (y position) is high enough or not,  0.5*this->_py = (this->py + 0.5*this->py)(centroid of half upper region) - this->py*/
	
	/***********************************if the detected staircase position (y position) is high enough then add one special validation set*************************************************/
	if((roi_size<150)&&(center_y_val_wcoor>y_coorthresh))
	{
		
		ROS_INFO("Additional validation set may be collected");
		
		cv::Mat array_depth_and_y_coor = cv::Mat::zeros(100, 2, CV_32F);
		int data_count = 0;
		bool invalid_depth_flag = false;
		
		float lower_limit = std::ceil(center_row);   /*actually lower_limit = center_row - roi_size/2  but additionally give a + roi_size/2 offset*/
		if(lower_limit<15)   lower_limit = 15; 
		
		float upper_limit = std::floor(center_row + roi_size); /*actually upper_limit = center_row + roi_size/2  but additionally give a + roi_size/2 offset*/
		if(upper_limit>(this->_preproc_resize_height-15))   upper_limit =  this->_preproc_resize_height-15;	
		
		
		this->str_det_cost_func->depth_data_collect(resized_depth_input_32f,array_depth_and_y_coor, this->_fy, this->_py, lower_limit, upper_limit, center_col, &data_count, &invalid_depth_flag);
		
		ROS_INFO("focal length y : %f, image center of y : %f, lower limt : %f, upper limit : %f, center_x : %f, data_count : %f, invalid depth flag %d",this->_fy,this->_py,lower_limit,upper_limit,center_col,data_count,invalid_depth_flag);
		
		float gradient_out;
		float deviation_cost_out;
		float x_avg_error_out;
		float continuity_factor_out;
		cv::Mat array_x_coor_final = cv::Mat::zeros(100, 1, CV_32F);  /*ths variable is dummy variable for matching the input argument of least_square_fit function */
		cv::Mat  array_depth_and_y_coor_final = array_depth_and_y_coor(cv::Range(0, data_count), cv::Range::all());		
		
		this->str_det_cost_func->least_square_fit(array_x_coor_final, array_depth_and_y_coor_final, &gradient_out, &continuity_factor_out, &deviation_cost_out, &x_avg_error_out);
				
		//deviation_cost_out = deviation_cost_out*10;  /*scale up */
		
		ROS_INFO("estimated gradient : %f, continuity_factor_out : %f, deviation_cost_out : %f, depth : %f", gradient_out, continuity_factor_out, deviation_cost_out, center_depth_val);
		
		/* if the gradient is out of range, then the beta and gamma are considered as outliers so large value  (e.g., 100) is set for the del beta and del gamma*/
		if((gradient_out > this->_online_rule_base_grad_max)||(gradient_out<this->_online_rule_base_grad_min))
		{
			fp_del_beta_array[validation_count] = 100;
			fp_del_gamma_array[validation_count] = 100;
		}
		else
		{
			continuity_factor_th_interp=interpol3(gradient_out,this->_online_rule_base_grad_diff_th_x1,this->_online_rule_base_grad_diff_th_x2,this->_online_rule_base_grad_diff_th_x3,
		                                                                       this->_online_rule_base_grad_diff_th_y1,this->_online_rule_base_grad_diff_th_y2,this->_online_rule_base_grad_diff_th_y3);
			deviation_cost_th_interp=interpol3(gradient_out, this->_online_rule_base_depth_y_error_th_x1, this->_online_rule_base_depth_y_error_th_x2, this->_online_rule_base_depth_y_error_th_x3,
		                                                                                               this->_online_rule_base_depth_y_error_th_y1, this->_online_rule_base_depth_y_error_th_y2, this->_online_rule_base_depth_y_error_th_y3);
			
			float del_beta = deviation_cost_out - deviation_cost_th_interp;
			if(del_beta<0) del_beta = 0;  /* del_beta < 0 means that deviation cost  is under true region */
			
			float del_gamma = continuity_factor_out - continuity_factor_th_interp;
			if(del_gamma<0) del_gamma = 0;    /* del_gamma < 0 means that continuitdy factor is under true region */

			fp_del_beta_array[validation_count] = del_beta;
			fp_del_gamma_array[validation_count] = del_gamma;
		}
		
		validation_count++;
		
		ROS_INFO("del_beta : %f, del_gamma : %f ", fp_del_beta_array[0], fp_del_gamma_array[0]);
		ROS_INFO("**********Additional validation set collection end******");
	}
	
	
    /**********************************************collect of validation set of left and right side *********************************************************************************8*/
	int loop_num = 2;
	int j =0;
	float x_val_offset[2] = {0.20,-0.20}; /*unit : m */
	
	for (j=0;j<loop_num;j++)
	{
		ROS_INFO("Loop number : ****** %d %d %d ****** validation set may be collected (left and right)", j+1, j+1, j+1);
	    cv::Mat array_depth_and_y_coor = cv::Mat::zeros(100, 2, CV_32F);
		int data_count = 0;
		bool invalid_depth_flag = false;
		
		
		float lower_limit = std::ceil(center_row - roi_size/2);
		if(lower_limit<15)   lower_limit = 15;
		
		float upper_limit = std::floor(center_row + roi_size/2);
		if(upper_limit>(this->_preproc_resize_height-15))   upper_limit =  this->_preproc_resize_height-15;
		
		float input_col = (center_x_val + x_val_offset[j])*this->_fx/center_depth_val + this->_px;
		if (input_col<5) input_col = 15;
		if (input_col>(this->_preproc_resize_width-15))  input_col = this->_preproc_resize_width - 15;
		
		input_col =  std::round(input_col);
		
		this->str_det_cost_func->depth_data_collect(resized_depth_input_32f,array_depth_and_y_coor, this->_fy, this->_py, lower_limit, upper_limit, input_col, &data_count, &invalid_depth_flag);

        ROS_INFO("focal length y : %f, image center of y : %f, lower limt : %f, upper limit : %f, center_x : %f, data_count : %d, invalid depth flag : %d", this->_fy, this->_py, lower_limit, upper_limit, input_col, data_count, invalid_depth_flag);
		
		float gradient_out;
		float deviation_cost_out;
		float x_avg_error_out;
		float continuity_factor_out;
		cv::Mat array_x_coor_final = cv::Mat::zeros(100, 1, CV_32F);  /*ths variable is dummy variable for matching the input argument of least_square_fit function */
		cv::Mat  array_depth_and_y_coor_final = array_depth_and_y_coor(cv::Range(0, data_count), cv::Range::all());
		
		this->str_det_cost_func->least_square_fit(array_x_coor_final, array_depth_and_y_coor_final, &gradient_out, &continuity_factor_out, &deviation_cost_out, &x_avg_error_out);
		
		//deviation_cost_out = deviation_cost_out*10;  /*scale up */
		
		ROS_INFO("estimated gradient : %f, continuity_factor_out : %f, deviation_cost_out : %f, depth : %f", gradient_out, continuity_factor_out, deviation_cost_out, center_depth_val);
		
		/* if the gradient is out of range, then the beta and gamma are considered as outliers so large value  (e.g., 100) is set for the del beta and del gamma*/
		if((gradient_out > this->_online_rule_base_grad_max)||(gradient_out<this->_online_rule_base_grad_min))
		{
			fp_del_beta_array[validation_count] = 100;
			fp_del_gamma_array[validation_count] = 100;
		}
		else
		{
			continuity_factor_th_interp=interpol3(gradient_out,this->_online_rule_base_grad_diff_th_x1,this->_online_rule_base_grad_diff_th_x2,this->_online_rule_base_grad_diff_th_x3,
		                                                                       this->_online_rule_base_grad_diff_th_y1,this->_online_rule_base_grad_diff_th_y2,this->_online_rule_base_grad_diff_th_y3);
			deviation_cost_th_interp=interpol3(gradient_out, this->_online_rule_base_depth_y_error_th_x1, this->_online_rule_base_depth_y_error_th_x2, this->_online_rule_base_depth_y_error_th_x3,
		                                                                                               this->_online_rule_base_depth_y_error_th_y1, this->_online_rule_base_depth_y_error_th_y2, this->_online_rule_base_depth_y_error_th_y3);
			
			float del_beta = deviation_cost_out - deviation_cost_th_interp;
			if(del_beta<0) del_beta = 0;  /* del_beta < 0 means that deviation cost  is under true region */
			
			float del_gamma = continuity_factor_out - continuity_factor_th_interp;
			if(del_gamma<0) del_gamma = 0;    /* del_gamma < 0 means that continuitdy factor is under true region */

			fp_del_beta_array[validation_count] = del_beta;
			fp_del_gamma_array[validation_count] = del_gamma;
		}
		
		ROS_INFO("validation count : %d del_beta : %f, del_gamma : %f ",validation_count, fp_del_beta_array[validation_count], fp_del_gamma_array[validation_count]);
		ROS_INFO("**********validation set collection end******");
		
		validation_count++;
	}
	
	 
	//center_point_x_coor = ((float)center_point_col - px) * depth_center / fx;
}