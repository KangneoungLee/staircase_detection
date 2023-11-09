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


#include "stair_detection/stair_detec_online_func.h"


//using namespace sciplot;


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
	float probability_discard_th = 0.4;
	float probability_scale_del_beta = 1;
	float probability_scale_del_gamma = 1;
	float probability_scale_del_x = 0.05;
	float probability_scale_del_y = 0.01;
	float online_detect_flag_reset_timer = 5;
	
	float false_detect_joint_probability_th = 0.3;
	
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
	param_nh.getParam("false_detect_joint_probability_th",false_detect_joint_probability_th);
	
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
	
    this->_false_detect_joint_probability_th = false_detect_joint_probability_th;
	
	this->str_det_cost_func = new STAIR_DETEC_COST_FUNC(false, false);
	
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
	float center_depth_final = 1000;
    float center_x_coor;
	float center_y_coor;
	float x_center_pixel;
	float y_center_pixel;
	float x_center_pixel_final = 0;
	float y_center_pixel_final = 0;
	float roi_size;

	float del_x;
	float del_y;

	float continuity_factor_th_interp;
	float deviation_cost_th_interp;

	bool gradient_range_ok = false;
	bool gradient_condition_ok = false;
	bool depth_y_error_condition_ok = false;
	bool avg_x_error_condition_ok = false;
	bool stair_case_detc_flag = false;
	
	
	 bool falsepos_diag_ = false;

    ROS_INFO("Enter rule based detection for online learning");
	int i=0;
	/*vector_for_learning include 8 elements*/
	/* stair gradient , least square error(x coordinate of map frame (depth) and z coordinate of map frame(rows)) per line,  average error (y coordinate of map frame (cols)), center point x (cam frame), center point y(cam frame), depth, center point x pixel, center point y pixel*/
    this->_detect_frame_cnt++;
	
	if(this->_activate_flag == true)
	{
	   ros::Duration elapse_time = ros::Time::now() - this->_detect_init_time;

	   if(this->_detect_frame_cnt > 50)//if(elapse_time>ros::Duration(this->_online_detect_flag_reset_timer))
	   {
		   this->_activate_flag = false;
		   this->isBetaArrayFull_ = false;
		   this->isGammaArrayFull_ = false;
		   this->BetaArrayIndex_ = 0;
		   this->GammaArrayIndex_ = 0;
		   
		   int j;
		   
		   for(j=0;j<this->_no_queue_online;j++)
		   {
			   DelBetaArray4BetaUp[i] = 0;
			   DelGammaArray4BetaUp[i] = 0;
			   DelBetaProbArray4BetaUp[i] = 0;
			   DelGammaProbArray4BetaUp[i] = 0;
			   DelX_PArray4BetaUp[i] = 0;
			   DelX_YArray4BetaUp[i] = 0;
			   
			   DelBetaArray4GammaUp[i] = 0;
			   DelGammaArray4GammaUp[i] = 0;
			   DelBetaProbArray4GammaUp[i] = 0;
			   DelGammaProbArray4GammaUp[i] = 0;
			   DelX_PArray4GammaUp[i] = 0;
			   DelY_PArray4GammaUp[i] = 0;
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

		if((gradient<this->_online_rule_base_grad_max)&&(gradient>this->_online_rule_base_grad_min))      gradient_range_ok = true;
		else      gradient_range_ok = false;

		if((gradient_range_ok == true)&&(gradient_diff<continuity_factor_th_interp))			gradient_condition_ok = true;
		else			gradient_condition_ok = false;

         if((gradient_range_ok==true)&&(depth_y_error<deviation_cost_th_interp)&&(depth_y_error>_online_rule_base_depth_y_error_th_min))			depth_y_error_condition_ok = true;
        else           depth_y_error_condition_ok = false;
	
		if((gradient_condition_ok==true)&&(depth_y_error_condition_ok==true))
		{
			bool false_pos_flag = reject_false_positive(depth_input, center_x_coor, center_y_coor, center_depth, x_center_pixel, y_center_pixel, continuity_factor_th_interp, deviation_cost_th_interp, roi_size, gradient, depth_y_error, gradient_diff);

			if(false_pos_flag == false)
			{
			   stair_case_detc_flag = true;
			   
			   if(center_depth<center_depth_final)
			   {
				   x_center_pixel_final = x_center_pixel;
				   y_center_pixel_final = y_center_pixel;
				   
				   center_depth_final = center_depth;
			   }
			
			   x_coor_cam_frame_tmp = it->at(4);
			   y_coor_cam_frame_tmp = it->at(5);
			   z_coor_cam_frame_tmp = it->at(6);

			   this->_detected_roi_x = x_center_pixel_final;
			   this->_detected_roi_y = y_center_pixel_final;
			
			   this->_detect_init_time = ros::Time::now();
			   this->_detect_frame_cnt = 0;
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
			
			if((this->_activate_flag == true)&&(gradient_range_ok == true))
			{
                del_x = std::abs(this->_detected_roi_x  - x_center_pixel);
				del_y = std::abs(this->_detected_roi_y  - y_center_pixel);
				
				//std::cout<<"gradient_range_ok : "<<gradient_range_ok<<"  gradient_condition_ok : "<<gradient_condition_ok<<"  del_gamma : "<<del_gamma<<std::endl;
				//std::cout<<"gradient_range_ok : "<<gradient_range_ok<<"  depth_y_error_condition_ok : "<<depth_y_error_condition_ok<<"  del_beta : "<<del_beta<<std::endl;
				//std::cout<<"this->_detected_roi_x : "<<this->_detected_roi_x<<"  x_center_pixel : "<<x_center_pixel<<"  del_x : "<<del_x<<std::endl;
				//std::cout<<"this->_detected_roi_y : "<<this->_detected_roi_y<<"  y_center_pixel : "<<y_center_pixel<<"  del_y : "<<del_y<<std::endl;

			    threshold_update(gradient_range_ok, continuity_factor_th_interp, deviation_cost_th_interp, _online_rule_base_depth_y_error_th_min, depth_y_error, gradient_diff ,del_x, del_y, gradient); 
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

	   rectangle_tmp1_pt1.x =x_center_pixel_final-30;
	   rectangle_tmp1_pt1.y =y_center_pixel_final+30;

	   rectangle_tmp1_pt2.x =x_center_pixel_final+30;
	   rectangle_tmp1_pt2.y =y_center_pixel_final-30;

	   cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);



	}
	
    visualize_threshold(rgb_input);

	 return stair_case_detc_flag;


}


void STAIR_DETEC_ONLINE_FUNC::threshold_update(bool gradient_range_ok, float continuity_factor_th_interp, float deviation_cost_th_interp, float deviation_cost_th_interp_min, float beta, float gamma, float del_x, float del_y, float gradient)
{
	 float DelBeta;
	 float DelGamma;
	
	 if(gradient_range_ok == true)
     {
		 if(gamma > continuity_factor_th_interp)
		 {
		     DelGamma = gamma - continuity_factor_th_interp;
		 }
		 else 
		 {
			 DelGamma = 0;
		 }
	 }
	 else
	 {
		 DelGamma = 5;
	 }
	
	 if((gradient_range_ok == true)&&(beta > deviation_cost_th_interp_min ))
     {
		 if(beta > deviation_cost_th_interp)
		 {
		     DelBeta = beta - deviation_cost_th_interp;
		 }
		 else 
		 {
			 DelBeta = 0;
		 }
	 }
	 else
	 {
		 DelBeta = 5;
	 }
	
	
	
     float InputDelBetaP = -this->_probability_scale_del_beta*DelBeta;
	 float InputDelGammaP = -this->_probability_scale_del_gamma*DelGamma;
	 float InputDelX_P = -this->_probability_scale_del_x*del_x;
	 float InputDelY_P = -this->_probability_scale_del_y*del_y;
	 
	 float DelBetaP = std::exp(InputDelBetaP);
	 float DelGammaP = std::exp(InputDelGammaP);
	 float DelX_P = std::exp(InputDelX_P);
	 float DelY_P = std::exp(InputDelY_P);
	 
	 //ROS_INFO("DelBeta : %f   DelGamma :  %f   del_x :  %f    del_y :   %f   DelBetaP  : %f    DelGammaP :  %f   DelX_P : %f   DelY_P : %f  ", DelBeta, DelGamma, del_x, del_y, DelBetaP, DelGammaP, DelX_P, DelY_P);
	 
	 //std::cout<<"  DelBeta : "<<DelBeta<<"  this->_probability_scale_del_beta : "<<this->_probability_scale_del_beta<<"  InputDelBetaP : "<<InputDelBetaP<<"  DelBetaP : "<<DelBetaP<<std::endl;
	 //std::cout<<"  DelGamma : "<<DelGamma<<"  this->_probability_scale_del_gamma : "<<this->_probability_scale_del_gamma<<"  InputDelGammaP : "<<InputDelGammaP<<"  DelGammaP : "<<DelGammaP<<std::endl;
	 //std::cout<<"  del_x : "<<del_x<<"  this->_probability_scale_del_x : "<<this->_probability_scale_del_x<<"  InputDelX_P : "<<InputDelX_P<<"  DelX_P : "<<DelX_P<<std::endl;
	// std::cout<<"  del_y : "<<del_y<<"  this->_probability_scale_del_y : "<<this->_probability_scale_del_y<<"  InputDelY_P : "<<InputDelY_P<<"  DelY_P : "<<DelY_P<<std::endl;
	 
	 if((DelBetaP<this->_probability_discard_th)||(DelGammaP<this->_probability_discard_th)||(DelX_P<this->_probability_discard_th)||(DelY_P<this->_probability_discard_th))
	 {
		 return;
	 }
	 else
	 {
		 if(DelBetaP<1)
		 {
			 //std::cout<<"**************** del beta inputted to array *********************************"<<std::endl;
			 //std::cout<<"BetaArrayIndex_ : "<<this->BetaArrayIndex_<<std::endl;
			 //std::cout<<"gradient : "<<gradient<<std::endl;
			 //std::cout<<"  DelBeta : "<<DelBeta<<"  this->_probability_scale_del_beta : "<<this->_probability_scale_del_beta<<"  InputDelBetaP : "<<InputDelBetaP<<"  DelBetaP : "<<DelBetaP<<std::endl;
	         //std::cout<<"  DelGamma : "<<DelGamma<<"  this->_probability_scale_del_gamma : "<<this->_probability_scale_del_gamma<<"  InputDelGammaP : "<<InputDelGammaP<<"  DelGammaP : "<<DelGammaP<<std::endl;
	         //std::cout<<"  del_x : "<<del_x<<"  this->_probability_scale_del_x : "<<this->_probability_scale_del_x<<"  InputDelX_P : "<<InputDelX_P<<"  DelX_P : "<<DelX_P<<std::endl;
	         //std::cout<<"  del_y : "<<del_y<<"  this->_probability_scale_del_y : "<<this->_probability_scale_del_y<<"  InputDelY_P : "<<InputDelY_P<<"  DelY_P : "<<DelY_P<<std::endl;
			 if(InitBetaFullFlag_ == true)
			 { 
				 PlotGradBeta4OnLng_.erase(PlotGradBeta4OnLng_.begin());
			     PlotBeta4OnLng_.erase(PlotBeta4OnLng_.begin());		 
			     PlotGradBeta4OnLng_.push_back(gamma);
			     PlotBeta4OnLng_.push_back(gradient);
				 
				 PlotDelBetaP4BetaOnLng_.erase(PlotDelBetaP4BetaOnLng_.begin());
				 PlotDelGammaP4BetaOnLng_.erase(PlotDelGammaP4BetaOnLng_.begin());				 
				 PlotDelBetaP4BetaOnLng_.push_back(DelBetaP);
				 PlotDelGammaP4BetaOnLng_.push_back(DelGammaP);
			 }
			 else
			 {
				 PlotGradBeta4OnLng_.push_back(gamma);
			     PlotBeta4OnLng_.push_back(gradient);
				 PlotDelBetaP4BetaOnLng_.push_back(DelBetaP);
				 PlotDelGammaP4BetaOnLng_.push_back(DelGammaP);
			 }

			 
			 this->DelBetaArray4BetaUp[this->BetaArrayIndex_] = DelBeta;
			 this->DelGammaArray4BetaUp[this->BetaArrayIndex_] = DelGamma;
			 this->GradArray4BetaUp[this->BetaArrayIndex_] = gradient;
			 this->DelBetaProbArray4BetaUp[this->BetaArrayIndex_] = DelBetaP;
			 this->DelGammaProbArray4BetaUp[this->BetaArrayIndex_] = DelGammaP;
			 this->DelX_PArray4BetaUp[this->BetaArrayIndex_] = DelX_P;
			 this->DelX_YArray4BetaUp[this->BetaArrayIndex_] = DelY_P;
			 
			 this->BetaArrayIndex_ = this->BetaArrayIndex_ + 1;
			 
			 if(this->BetaArrayIndex_ == this->_no_queue_online)
			 {
				 this->BetaArrayIndex_ = 0;
				 this->isBetaArrayFull_ = true;
				 
				 InitBetaFullFlag_ = true;
			 }
			 
		 }
		 
		 if(DelGammaP<1)
		 {
			 
			 //std::cout<<"**************** del gamma inputted to array *********************************"<<std::endl;
			 //std::cout<<"GammaArrayIndex_ : "<<this->GammaArrayIndex_<<std::endl;
			 //std::cout<<"gradient : "<<gradient<<std::endl;
			 //std::cout<<"  DelBeta : "<<DelBeta<<"  this->_probability_scale_del_beta : "<<this->_probability_scale_del_beta<<"  InputDelBetaP : "<<InputDelBetaP<<"  DelBetaP : "<<DelBetaP<<std::endl;
	         //std::cout<<"  DelGamma : "<<DelGamma<<"  this->_probability_scale_del_gamma : "<<this->_probability_scale_del_gamma<<"  InputDelGammaP : "<<InputDelGammaP<<"  DelGammaP : "<<DelGammaP<<std::endl;
	         //std::cout<<"  del_x : "<<del_x<<"  this->_probability_scale_del_x : "<<this->_probability_scale_del_x<<"  InputDelX_P : "<<InputDelX_P<<"  DelX_P : "<<DelX_P<<std::endl;
	         //std::cout<<"  del_y : "<<del_y<<"  this->_probability_scale_del_y : "<<this->_probability_scale_del_y<<"  InputDelY_P : "<<InputDelY_P<<"  DelY_P : "<<DelY_P<<std::endl;
			 
			 if(InitGammaFullFlag_ == true)
			 {
				 PlotGradGamma4OnLng_.erase(PlotGradGamma4OnLng_.begin());
			     PlotGamma4OnLng_.erase(PlotGamma4OnLng_.begin());		 
			     PlotGradGamma4OnLng_.push_back(gamma);
			     PlotGamma4OnLng_.push_back(gradient);

				 PlotDelBetaP4GammaOnLng_.erase(PlotDelBetaP4GammaOnLng_.begin());
				 PlotDelGammaP4GammaOnLng_.erase(PlotDelGammaP4GammaOnLng_.begin());				 
				 PlotDelBetaP4GammaOnLng_.push_back(DelBetaP);
				 PlotDelGammaP4GammaOnLng_.push_back(DelGammaP);
				 
			 }
			 else
			 {
				 PlotGradGamma4OnLng_.push_back(gamma);
			     PlotGamma4OnLng_.push_back(gradient);
				 PlotDelBetaP4GammaOnLng_.push_back(DelBetaP);
				 PlotDelGammaP4GammaOnLng_.push_back(DelGammaP);
			 }
			 
			 this->DelBetaArray4GammaUp[this->GammaArrayIndex_] = DelBeta;
			 this->DelGammaArray4GammaUp[this->GammaArrayIndex_] = DelGamma;
			 this->GradArray4GammaUp[this->GammaArrayIndex_] = gradient;
			 this->DelBetaProbArray4GammaUp[this->GammaArrayIndex_] = DelBetaP;
			 this->DelGammaProbArray4GammaUp[this->GammaArrayIndex_] = DelGammaP;
			 this->DelX_PArray4GammaUp[this->GammaArrayIndex_] = DelX_P;
			 this->DelY_PArray4GammaUp[this->GammaArrayIndex_] = DelY_P;
			 
			 
			 this->GammaArrayIndex_ = this->GammaArrayIndex_ + 1;
			 
			 if(this->GammaArrayIndex_ == this->_no_queue_online)
			 {
				 this->GammaArrayIndex_ = 0;
				 this->isGammaArrayFull_ = true;
				 
				 InitGammaFullFlag_ = true;

			 }
		 }
	 }

	 

	 
	 if(this->isBetaArrayFull_ == true)
	 {
		 //std::cout<<" array check for beta threshold adjustment " << std::endl; 
	     //std::cout<<"  DelBetaArray4BetaUp -->> "<<" index 0 : "<<this->DelBetaArray4BetaUp[0]<<" index 1 : "<<this->DelBetaArray4BetaUp[1]<<" index 2 : "<<this->DelBetaArray4BetaUp[2]<<" index 3 : "<<this->DelBetaArray4BetaUp[3]<<" index 4 : "<<this->DelBetaArray4BetaUp[4]<<" index 5 : "<<this->DelBetaArray4BetaUp[5]<<std::endl;
	     //std::cout<<"  DelGammaArray4BetaUp -->> "<<" index 0 : "<<this->DelGammaArray4BetaUp[0]<<" index 1 : "<<this->DelGammaArray4BetaUp[1]<<" index 2 : "<<this->DelGammaArray4BetaUp[2]<<" index 3 : "<<this->DelGammaArray4BetaUp[3]<<" index 4 : "<<this->DelGammaArray4BetaUp[4]<<" index 5 : "<<this->DelGammaArray4BetaUp[5]<<std::endl;
	     //std::cout<<"  GradArray4BetaUp -->> "<<" index 0 : "<<this->GradArray4BetaUp[0]<<" index 1 : "<<this->GradArray4BetaUp[1]<<" index 2 : "<<this->GradArray4BetaUp[2]<<" index 3 : "<<this->GradArray4BetaUp[3]<<" index 4 : "<<this->GradArray4BetaUp[4]<<" index 5 : "<<this->GradArray4BetaUp[5]<<std::endl;
	     //std::cout<<"  DelBetaProbArray4BetaUp -->> "<<" index 0 : "<<this->DelBetaProbArray4BetaUp[0]<<" index 1 : "<<this->DelBetaProbArray4BetaUp[1]<<" index 2 : "<<this->DelBetaProbArray4BetaUp[2]<<" index 3 : "<<this->DelBetaProbArray4BetaUp[3]<<" index 4 : "<<this->DelBetaProbArray4BetaUp[4]<<" index 5 : "<<this->DelBetaProbArray4BetaUp[5]<<std::endl;
         //std::cout<<"  DelGammaProbArray4BetaUp -->> "<<" index 0 : "<<this->DelGammaProbArray4BetaUp[0]<<" index 1 : "<<this->DelGammaProbArray4BetaUp[1]<<" index 2 : "<<this->DelGammaProbArray4BetaUp[2]<<" index 3 : "<<this->DelGammaProbArray4BetaUp[3]<<" index 4 : "<<this->DelGammaProbArray4BetaUp[4]<<" index 5 : "<<this->DelGammaProbArray4BetaUp[5]<<std::endl;
	     //std::cout<<"  DelX_PArray4BetaUp -->> "<<" index 0 : "<<this->DelX_PArray4BetaUp[0]<<" index 1 : "<<this->DelX_PArray4BetaUp[1]<<" index 2 : "<<this->DelX_PArray4BetaUp[2]<<" index 3 : "<<this->DelX_PArray4BetaUp[3]<<" index 4 : "<<this->DelX_PArray4BetaUp[4]<<" index 5 : "<<this->DelX_PArray4BetaUp[5]<<std::endl;
	     //std::cout<<"  DelX_YArray4BetaUp -->> "<<" index 0 : "<<this->DelX_YArray4BetaUp[0]<<" index 1 : "<<this->DelX_YArray4BetaUp[1]<<" index 2 : "<<this->DelX_YArray4BetaUp[2]<<" index 3 : "<<this->DelX_YArray4BetaUp[3]<<" index 4 : "<<this->DelX_YArray4BetaUp[4]<<" index 5 : "<<this->DelX_YArray4BetaUp[5]<<std::endl;
	     //std::cout<<" array check for beta threshold adjustment finished " << std::endl;
		 

		 
		 //std::cout<<"********************************** beta threshold update start ****************************** " << std::endl;
		 float BetaUpDateDiv = 0;
		 float BetaUpdateVal = 0;
		 float Grad4BetaUpateVal = 0;
		 int i = 0;
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 BetaUpDateDiv = BetaUpDateDiv + this->DelBetaProbArray4BetaUp[i] + this->DelGammaProbArray4BetaUp[i];
		 }
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 BetaUpdateVal = BetaUpdateVal + (this->DelBetaProbArray4BetaUp[i] + this->DelGammaProbArray4BetaUp[i])/BetaUpDateDiv*this->DelBetaArray4BetaUp[i];
			 Grad4BetaUpateVal = Grad4BetaUpateVal + (this->DelBetaProbArray4BetaUp[i] + this->DelGammaProbArray4BetaUp[i])/BetaUpDateDiv*this->GradArray4BetaUp[i];
		 }
		 		 
		 /*Beta threshold update */
		 this->beta_thresh_update(Grad4BetaUpateVal, BetaUpdateVal);
		 
		 /* update the del beta and beta probability queue with new beta threshold for online learning */
		  for(i=0;i<this->_no_queue_online;i++)
		  {
			  float del_beta_prob_input_update;
			  float del_beta_prob_update;
			  
			 this->DelBetaArray4BetaUp[i] = this->DelBetaArray4BetaUp[i] - 0.7*BetaUpdateVal;
			 
			 del_beta_prob_input_update = -this->_probability_scale_del_beta*this->DelBetaArray4BetaUp[i];
			 del_beta_prob_update = std::exp(del_beta_prob_input_update);
			 
			 this->DelBetaProbArray4BetaUp[i] = del_beta_prob_update;
			 
			 if(this->DelBetaArray4BetaUp[i] < 0)
			 {
				 this->DelBetaArray4BetaUp[i] = 0;
				 this->DelBetaProbArray4BetaUp[i] = 0;
			 }
			 
			 this->DelBetaArray4GammaUp[i] = this->DelBetaArray4GammaUp[i] - 0.7*BetaUpdateVal;
			 
			 del_beta_prob_input_update = -this->_probability_scale_del_beta*this->DelBetaArray4GammaUp[i];
			 del_beta_prob_update = std::exp(del_beta_prob_input_update);
			 
			 this->DelBetaProbArray4GammaUp[i] = del_beta_prob_update;

			 if(this->DelBetaArray4GammaUp[i] < 0)
			 {
				 this->DelBetaArray4GammaUp[i] = 0;
				 this->DelBetaProbArray4GammaUp[i] = 0;
			 }
			 
		  }
		 
		 this->isBetaArrayFull_ = false;
		 
	 }

	 if(this->isGammaArrayFull_ == true)
	 {
		 //std::cout<<" array check for gamma threshold adjustment " << std::endl; 
	     //std::cout<<"  DelBetaArray4GammaUp -->> "<<" index 0 : "<<this->DelBetaArray4GammaUp[0]<<" index 1 : "<<this->DelBetaArray4GammaUp[1]<<" index 2 : "<<this->DelBetaArray4GammaUp[2]<<" index 3 : "<<this->DelBetaArray4GammaUp[3]<<" index 4 : "<<this->DelBetaArray4GammaUp[4]<<" index 5 : "<<this->DelBetaArray4GammaUp[5]<<std::endl;
	     //std::cout<<"  DelGammaArray4GammaUp -->> "<<" index 0 : "<<this->DelGammaArray4GammaUp[0]<<" index 1 : "<<this->DelGammaArray4GammaUp[1]<<" index 2 : "<<this->DelGammaArray4GammaUp[2]<<" index 3 : "<<this->DelGammaArray4GammaUp[3]<<" index 4 : "<<this->DelGammaArray4GammaUp[4]<<" index 5 : "<<this->DelGammaArray4GammaUp[5]<<std::endl;
	     //std::cout<<"  GradArray4GammaUp -->> "<<" index 0 : "<<this->GradArray4GammaUp[0]<<" index 1 : "<<this->GradArray4GammaUp[1]<<" index 2 : "<<this->GradArray4GammaUp[2]<<" index 3 : "<<this->GradArray4GammaUp[3]<<" index 4 : "<<this->GradArray4GammaUp[4]<<" index 5 : "<<this->GradArray4GammaUp[5]<<std::endl;
	     //std::cout<<"  DelBetaProbArray4GammaUp -->> "<<" index 0 : "<<this->DelBetaProbArray4GammaUp[0]<<" index 1 : "<<this->DelBetaProbArray4GammaUp[1]<<" index 2 : "<<this->DelBetaProbArray4GammaUp[2]<<" index 3 : "<<this->DelBetaProbArray4GammaUp[3]<<" index 4 : "<<this->DelBetaProbArray4GammaUp[4]<<" index 5 : "<<this->DelBetaProbArray4GammaUp[5]<<std::endl;
         //std::cout<<"  DelGammaProbArray4GammaUp -->> "<<" index 0 : "<<this->DelGammaProbArray4GammaUp[0]<<" index 1 : "<<this->DelGammaProbArray4GammaUp[1]<<" index 2 : "<<this->DelGammaProbArray4GammaUp[2]<<" index 3 : "<<this->DelGammaProbArray4GammaUp[3]<<" index 4 : "<<this->DelGammaProbArray4GammaUp[4]<<" index 5 : "<<this->DelGammaProbArray4GammaUp[5]<<std::endl;
	     //std::cout<<"  DelX_PArray4GammaUp -->> "<<" index 0 : "<<this->DelX_PArray4GammaUp[0]<<" index 1 : "<<this->DelX_PArray4GammaUp[1]<<" index 2 : "<<this->DelX_PArray4GammaUp[2]<<" index 3 : "<<this->DelX_PArray4GammaUp[3]<<" index 4 : "<<this->DelX_PArray4GammaUp[4]<<" index 5 : "<<this->DelX_PArray4GammaUp[5]<<std::endl;
	     //std::cout<<"  DelY_PArray4GammaUp -->> "<<" index 0 : "<<this->DelY_PArray4GammaUp[0]<<" index 1 : "<<this->DelY_PArray4GammaUp[1]<<" index 2 : "<<this->DelY_PArray4GammaUp[2]<<" index 3 : "<<this->DelY_PArray4GammaUp[3]<<" index 4 : "<<this->DelY_PArray4GammaUp[4]<<" index 5 : "<<this->DelY_PArray4GammaUp[5]<<std::endl;
	     //std::cout<<" array check for gamma threshold adjustment finished " << std::endl;
		 
		 
		 //std::cout<<"********************************** gaama threshold update start ****************************** " << std::endl;
		 float GammaUpDateDiv = 0;
		 float GammaUpdateVal = 0;
		 float Grad4GammaUpateVal = 0;
		 int i = 0;
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 GammaUpDateDiv = GammaUpDateDiv + this->DelBetaProbArray4GammaUp[i] + this->DelGammaProbArray4GammaUp[i];
		 }
		 
		 for(i=0;i<this->_no_queue_online;i++)
		 {
			 GammaUpdateVal = GammaUpdateVal + (this->DelBetaProbArray4GammaUp[i] + this->DelGammaProbArray4GammaUp[i])/GammaUpDateDiv*this->DelBetaArray4GammaUp[i];
			 Grad4GammaUpateVal = Grad4GammaUpateVal + (this->DelBetaProbArray4GammaUp[i] + this->DelGammaProbArray4GammaUp[i])/GammaUpDateDiv*this->GradArray4GammaUp[i]; 
		 }
		 
		 /* Gamma threshold update*/
         this->gamma_thresh_update(Grad4GammaUpateVal, GammaUpdateVal);		 
	 
		 /* update the del gamma and gamma probability queue with new gamma threshold for online learning */
		  for(i=0;i<this->_no_queue_online;i++)
		  {
			  float del_gamma_prob_input_update;
			  float del_gamma_prob_update;
			  
			 this->DelGammaArray4BetaUp[i] = this->DelGammaArray4BetaUp[i] - 0.7*GammaUpdateVal;
			 
			 del_gamma_prob_input_update = -this->_probability_scale_del_gamma*this->DelGammaArray4BetaUp[i];
			 del_gamma_prob_update = std::exp(del_gamma_prob_input_update);
			 
			 this->DelGammaProbArray4BetaUp[i] = del_gamma_prob_update;
			 
			 if(this->DelGammaArray4BetaUp[i] < 0)
			 {
				 this->DelGammaArray4BetaUp[i] = 0;
				 this->DelGammaProbArray4BetaUp[i] = 0;
			 }
			 
			 this->DelGammaArray4GammaUp[i] = this->DelGammaArray4GammaUp[i] - 0.7*GammaUpdateVal;
			 
			 del_gamma_prob_input_update = -this->_probability_scale_del_gamma*this->DelGammaArray4GammaUp[i];
			 del_gamma_prob_update = std::exp(del_gamma_prob_input_update);
			 
			 this->DelGammaProbArray4GammaUp[i] = del_gamma_prob_update;

			 if(this->DelGammaArray4GammaUp[i] < 0)
			 {
				 this->DelGammaArray4GammaUp[i] = 0;
				 this->DelGammaProbArray4GammaUp[i] = 0;
			 } 
			 
		  }
		  
		  this->isGammaArrayFull_ = false;
	
	 }	 
	 
}

void STAIR_DETEC_ONLINE_FUNC::gamma_thresh_update(float Grad4GammaUpateVal, float GammaUpdateVal)
{
		 ROS_INFO("gamma_thresh_update :::: Grad4GammaUpateVal : %f   GammaUpdateVal : %f ", Grad4GammaUpateVal, GammaUpdateVal);
		 
		 float  LeftPivot, RightPivot;
		 
		 /*threshold update code should be inserted at here*/
		 if(Grad4GammaUpateVal<=this->_online_rule_base_grad_diff_th_x1)
		 {
			 this->_online_rule_base_grad_diff_th_y1 = this->_online_rule_base_grad_diff_th_y1 + GammaUpdateVal;
		 }
		 else if((Grad4GammaUpateVal>this->_online_rule_base_grad_diff_th_x1)&&(Grad4GammaUpateVal<=this->_online_rule_base_grad_diff_th_x2))
		 {
			 LeftPivot = this->_online_rule_base_grad_diff_th_x1;
			 RightPivot = this->_online_rule_base_grad_diff_th_x2;
			 
			 this->_online_rule_base_grad_diff_th_y1 = this->_online_rule_base_grad_diff_th_y1 + (Grad4GammaUpateVal - LeftPivot)/(RightPivot - LeftPivot)*GammaUpdateVal;
			 this->_online_rule_base_grad_diff_th_y2 = this->_online_rule_base_grad_diff_th_y2 + (RightPivot - Grad4GammaUpateVal)/(RightPivot - LeftPivot)*GammaUpdateVal;
		 }
		 else if((Grad4GammaUpateVal>this->_online_rule_base_grad_diff_th_x2)&&(Grad4GammaUpateVal<=this->_online_rule_base_grad_diff_th_x3))
		 {
			 LeftPivot = this->_online_rule_base_grad_diff_th_x2;
			 RightPivot = this->_online_rule_base_grad_diff_th_x3;

			 this->_online_rule_base_grad_diff_th_y2 = this->_online_rule_base_grad_diff_th_y2 + (Grad4GammaUpateVal - LeftPivot)/(RightPivot - LeftPivot)*GammaUpdateVal;
			 this->_online_rule_base_grad_diff_th_y3 = this->_online_rule_base_grad_diff_th_y3 + (RightPivot - Grad4GammaUpateVal)/(RightPivot - LeftPivot)*GammaUpdateVal;			 
		 }
		 else /*Grad4GammaUpateVal>this->_online_rule_base_grad_diff_th_x3*/
		 {
			 this->_online_rule_base_grad_diff_th_y3 = this->_online_rule_base_grad_diff_th_y3 + GammaUpdateVal;
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

		 if(this->_online_rule_base_grad_diff_th_y1<=0)
		 {
			 this->_online_rule_base_grad_diff_th_y1 = 0.01;
		 }
		 if(this->_online_rule_base_grad_diff_th_y2<=0)
		 {
			 this->_online_rule_base_grad_diff_th_y2 = 0.01;
		 }
		 if(this->_online_rule_base_grad_diff_th_y3<=0)
		 {
			 this->_online_rule_base_grad_diff_th_y3 = 0.01;
		 }	
	

}


void STAIR_DETEC_ONLINE_FUNC::beta_thresh_update(float Grad4BetaUpateVal, float BetaUpdateVal)
{
	    ROS_INFO("beta_thresh_update :::: Grad4BetaUpateVal : %f   BetaUpdateVal : %f ", Grad4BetaUpateVal, BetaUpdateVal);
	
		 float  LeftPivot, RightPivot;
		 
		 if(Grad4BetaUpateVal<=this->_online_rule_base_depth_y_error_th_x1)
		 {
			 this->_online_rule_base_depth_y_error_th_y1 = this->_online_rule_base_depth_y_error_th_y1 + BetaUpdateVal;
		 }
		 else if((Grad4BetaUpateVal>this->_online_rule_base_depth_y_error_th_x1)&&(Grad4BetaUpateVal<=this->_online_rule_base_depth_y_error_th_x2))
		 {
			 LeftPivot = this->_online_rule_base_depth_y_error_th_x1;
			 RightPivot = this->_online_rule_base_depth_y_error_th_x2;
			 
			 this->_online_rule_base_depth_y_error_th_y1 = this->_online_rule_base_depth_y_error_th_y1 + (Grad4BetaUpateVal - LeftPivot)/(RightPivot - LeftPivot)*BetaUpdateVal;
			 this->_online_rule_base_depth_y_error_th_y2 = this->_online_rule_base_depth_y_error_th_y2 + (RightPivot - Grad4BetaUpateVal)/(RightPivot - LeftPivot)*BetaUpdateVal;
		 }
		 else if((Grad4BetaUpateVal>this->_online_rule_base_depth_y_error_th_x2)&&(Grad4BetaUpateVal<=this->_online_rule_base_depth_y_error_th_x3))
		 {
			 LeftPivot = this->_online_rule_base_depth_y_error_th_x2;
			 RightPivot = this->_online_rule_base_depth_y_error_th_x3;

			 this->_online_rule_base_depth_y_error_th_y2 = this->_online_rule_base_depth_y_error_th_y2 + (Grad4BetaUpateVal - LeftPivot)/(RightPivot - LeftPivot)*BetaUpdateVal;
			 this->_online_rule_base_depth_y_error_th_y3 = this->_online_rule_base_depth_y_error_th_y3 + (RightPivot - Grad4BetaUpateVal)/(RightPivot - LeftPivot)*BetaUpdateVal;			 
		 }
		 else /*Grad4BetaUpateVal>this->_online_rule_base_depth_y_error_th_x3*/
		 {
			 this->_online_rule_base_depth_y_error_th_y3 = this->_online_rule_base_depth_y_error_th_y3 + BetaUpdateVal;
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
		 
		 
		 if(this->_online_rule_base_depth_y_error_th_y1<=0)
		 {
			 this->_online_rule_base_depth_y_error_th_y1 = 0.01;
		 }
		 if(this->_online_rule_base_depth_y_error_th_y2<=0)
		 {
			 this->_online_rule_base_depth_y_error_th_y2 = 0.01;
		 }
		 if(this->_online_rule_base_depth_y_error_th_y3<=0)
		 {
			 this->_online_rule_base_depth_y_error_th_y3 = 0.01;
		 }	
	
}



bool STAIR_DETEC_ONLINE_FUNC::reject_false_positive(const cv::Mat& depth_input, float center_x_val, float center_y_val, float center_depth_val, float center_col, float center_row, float continuity_factor_th_interp, 
                                                                                           float deviation_cost_th_interp, float roi_size, float GradOrg, float BetaOrg, float GammaOrg)
{
	float FalRejDelBetaArray[3];
	float FalRejDelBetaProbArray[3];
	float FalRejDelGammaArray[3];
	float FalRejDelGammaProbArray[3];
	float DelX_Array[3];
	float DelY_Array[3];
	float GradArray[3];
	float BetaArray[3];
	float GammaArray[3];
	bool gradient_range_ok_Array[3];
	
	//float continuity_factor_th_interp;
	//float deviation_cost_th_interp;
	
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
		
		cv::Mat array_depth_and_y_coor = cv::Mat::zeros(200, 2, CV_32F);
		int data_count = 0;
		bool invalid_depth_flag = false;
		
		float lower_limit = std::ceil(center_row);   /*actually lower_limit = center_row - roi_size/2  but additionally give a + roi_size/2 offset*/
		if(lower_limit<15)   lower_limit = 15; 
		
		float upper_limit = std::floor(center_row + roi_size); /*actually upper_limit = center_row + roi_size/2  but additionally give a + roi_size/2 offset*/
		if(upper_limit>(this->_preproc_resize_height-15))   upper_limit =  this->_preproc_resize_height-15;	
		
		float input_col = center_col;

		this->str_det_cost_func->depth_data_collect(resized_depth_input_32f,array_depth_and_y_coor, this->_fy, this->_py, lower_limit, upper_limit, input_col, &data_count, &invalid_depth_flag);
		
		ROS_INFO("focal length y : %f, image center of y : %f, lower limt : %f, upper limit : %f, center_x : %f, data_count : %f, invalid depth flag %d",this->_fy,this->_py,lower_limit,upper_limit,center_col,data_count,invalid_depth_flag);
		if(invalid_depth_flag== false){
		   float Grad;
		   float Beta;
		   float x_avg_error_out;
		   float Gamma;

		   cv::Mat array_x_coor_final = cv::Mat::zeros(200, 1, CV_32F);  /*ths variable is dummy variable for matching the input argument of least_square_fit function */
		   cv::Mat  array_depth_and_y_coor_final = array_depth_and_y_coor(cv::Range(0, data_count), cv::Range::all());		

		   this->str_det_cost_func->least_square_fit(array_x_coor_final, array_depth_and_y_coor_final, &Grad, &Gamma, &Beta, &x_avg_error_out);
				
		   //Beta = Beta*10;  /*scale up */
		
		   GradArray[validation_count] = Grad;
		   BetaArray[validation_count] = Beta;
		   GammaArray[validation_count] = Gamma;
		   DelX_Array[validation_count] = std::abs(center_col - input_col);
		   DelY_Array[validation_count] = std::abs(center_row - (lower_limit + upper_limit)/2);
		
		   ROS_INFO("estimated gradient : %f, Gamma : %f, Beta : %f, depth : %f", Grad, Gamma, Beta, center_depth_val);
		
		   /* if the gradient is out of range, then the beta and gamma are considered as outliers so large value  (e.g., 100) is set for the del beta and del gamma*/
		   if((Grad > (this->_online_rule_base_grad_max + 0.2))||(Grad<(this->_online_rule_base_grad_min - 0.1)))
		   {
			   FalRejDelBetaArray[validation_count] = 100;
			   FalRejDelGammaArray[validation_count] = 100;
			   gradient_range_ok_Array[validation_count] = false;
		   }
		   else
		   {
			   //continuity_factor_th_interp=interpol3(Grad,this->_online_rule_base_grad_diff_th_x1,this->_online_rule_base_grad_diff_th_x2,this->_online_rule_base_grad_diff_th_x3,
		       //                                                                   this->_online_rule_base_grad_diff_th_y1,this->_online_rule_base_grad_diff_th_y2,this->_online_rule_base_grad_diff_th_y3);
			   //deviation_cost_th_interp=interpol3(Grad, this->_online_rule_base_depth_y_error_th_x1, this->_online_rule_base_depth_y_error_th_x2, this->_online_rule_base_depth_y_error_th_x3,
		       //                                                                                           this->_online_rule_base_depth_y_error_th_y1, this->_online_rule_base_depth_y_error_th_y2, this->_online_rule_base_depth_y_error_th_y3);

			   gradient_range_ok_Array[validation_count] = true;
			   //float DelBeta = Beta - deviation_cost_th_interp;
			   //if(DelBeta<0) DelBeta = 0;  /* del_beta < 0 means that deviation cost  is under true region */
			
			   float DelBeta = std::abs(Beta - BetaOrg);
			
			   //float DelGamma = Gamma - continuity_factor_th_interp;
			   //if(DelGamma<0) DelGamma = 0;    /* DelGamma < 0 means that continuitdy factor is under true region */

               float DelGamma = std::abs(Gamma - GammaOrg);

			   FalRejDelBetaArray[validation_count] = DelBeta;
			   FalRejDelGammaArray[validation_count] = DelGamma;
		   }
		
		   validation_count++;
		
		   //ROS_INFO("DelBeta : %f, DelGamma : %f ", FalRejDelBetaArray[0], FalRejDelGammaArray[0]);
		   ROS_INFO("**********Additional validation set collection end******"); 
		}
	}
	
	
    /**********************************************collect of validation set of left and right side *********************************************************************************8*/
	int loop_num = 2;
	int j =0;
	float x_val_offset[2] = {0.20,-0.20}; /*unit : m */
	
	for (j=0;j<loop_num;j++)
	{
		ROS_INFO("Loop number : ****** %d %d %d ****** validation set may be collected (left and right)", j+1, j+1, j+1);
	    cv::Mat array_depth_and_y_coor = cv::Mat::zeros(200, 2, CV_32F);
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
        if(invalid_depth_flag==true)   continue;
        //ROS_INFO("focal length y : %f, image center of y : %f, lower limt : %f, upper limit : %f, center_x : %f, data_count : %d, invalid depth flag : %d", this->_fy, this->_py, lower_limit, upper_limit, input_col, data_count, invalid_depth_flag);
		
		float Grad;
		float Beta;
		float x_avg_error_out;
		float Gamma;

		cv::Mat array_x_coor_final = cv::Mat::zeros(200, 1, CV_32F);  /*ths variable is dummy variable for matching the input argument of least_square_fit function */
		cv::Mat  array_depth_and_y_coor_final = array_depth_and_y_coor(cv::Range(0, data_count), cv::Range::all());

		this->str_det_cost_func->least_square_fit(array_x_coor_final, array_depth_and_y_coor_final, &Grad, &Gamma, &Beta, &x_avg_error_out);

		//Beta = Beta*10;  /*scale up */
		
		GradArray[validation_count] = Grad;
		BetaArray[validation_count] = Beta;
		GammaArray[validation_count] = Gamma;
		DelX_Array[validation_count] = std::abs(center_col - input_col);
		DelY_Array[validation_count] = std::abs(center_row - (lower_limit + upper_limit)/2);
		
		//ROS_INFO("estimated gradient : %f, Gamma : %f, Beta : %f, depth : %f", Grad, Gamma, Beta, center_depth_val);
		
		/* if the gradient is out of range, then the beta and gamma are considered as outliers so large value  (e.g., 100) is set for the del beta and del gamma*/
		if((Grad > this->_online_rule_base_grad_max)||(Grad<this->_online_rule_base_grad_min))
		{
			FalRejDelBetaArray[validation_count] = 1;
			FalRejDelGammaArray[validation_count] = 1;
			gradient_range_ok_Array[validation_count] = false;
		}
		else
		{
			//continuity_factor_th_interp=interpol3(Grad,this->_online_rule_base_grad_diff_th_x1,this->_online_rule_base_grad_diff_th_x2,this->_online_rule_base_grad_diff_th_x3,
		    //                                                                   this->_online_rule_base_grad_diff_th_y1,this->_online_rule_base_grad_diff_th_y2,this->_online_rule_base_grad_diff_th_y3);
			//deviation_cost_th_interp=interpol3(Grad, this->_online_rule_base_depth_y_error_th_x1, this->_online_rule_base_depth_y_error_th_x2, this->_online_rule_base_depth_y_error_th_x3,
		    //                                                                                          this->_online_rule_base_depth_y_error_th_y1, this->_online_rule_base_depth_y_error_th_y2, this->_online_rule_base_depth_y_error_th_y3);
			gradient_range_ok_Array[validation_count] = true;
			//float DelBeta = Beta - deviation_cost_th_interp;
			//if(DelBeta<0) DelBeta = 0;  /* DelBeta < 0 means that deviation cost  is under true region */
			
			float DelBeta = std::abs(Beta - BetaOrg);
			
			//float DelGamma = Gamma - continuity_factor_th_interp;
			//if(DelGamma<0) DelGamma = 0;    /* DelGamma < 0 means that continuitdy factor is under true region */

            float DelGamma = std::abs(Gamma - GammaOrg);

			FalRejDelBetaArray[validation_count] = DelBeta;
			FalRejDelGammaArray[validation_count] = DelGamma;
				
		}
		
		//ROS_INFO("validation count : %d DelBeta : %f, DelGamma : %f ",validation_count, FalRejDelBetaArray[validation_count], FalRejDelGammaArray[validation_count]);
		ROS_INFO("**********validation set collection end******");
		
		validation_count++;
	}
	

	//center_point_x_coor = ((float)center_point_col - px) * depth_center / fx;

    float JointP_BetaGamma = 1;

	for(int k=0; k<validation_count; k++)
	{
		FalRejDelBetaProbArray[k] = std::exp(-this->_probability_scale_del_beta*FalRejDelBetaArray[k]);
		FalRejDelGammaProbArray[k] = std::exp(-this->_probability_scale_del_gamma*FalRejDelGammaArray[k]);
		
		JointP_BetaGamma = JointP_BetaGamma*FalRejDelBetaProbArray[k]*FalRejDelGammaProbArray[k];
	}
		

	
	if(JointP_BetaGamma >= this->_false_detect_joint_probability_th)
	{
		ROS_INFO("No false positive detection ::: JointP_BetaGamma : %f   Threshold : %f ", JointP_BetaGamma, this->_false_detect_joint_probability_th);
		for(int k=0; k<validation_count; k++)
	    {
	      threshold_update(gradient_range_ok_Array[k], continuity_factor_th_interp, deviation_cost_th_interp, this->_online_rule_base_depth_y_error_th_min, BetaArray[k], GammaArray[k], DelX_Array[k], DelY_Array[k], GradArray[k]);
	    }		
		
		return false; // false means that there is no false positive
	}
	else
	{
		ROS_INFO("False positive detected ::: JointP_BetaGamma : %f   Threshold : %f ", JointP_BetaGamma, this->_false_detect_joint_probability_th);
        float BetaUpdateVal = BetaOrg - deviation_cost_th_interp;
		float GammaUpdateVal = GammaOrg - continuity_factor_th_interp;
		
		float FalsePosUpW = 0.2; //1.05;
		
		bool SkipFalPosThUpd = false;
		
		for(int k=0; k<validation_count; k++)
		{
			if(gradient_range_ok_Array[k] == false)
			{
				SkipFalPosThUpd = true;
				break;
			}
			
		}
		
		if(SkipFalPosThUpd == false)
		{
		    float BetaUpdateR  = std::abs(BetaUpdateVal)/(deviation_cost_th_interp + 0.001);
		    float GammaUpdateR  = std::abs(GammaUpdateVal)/(continuity_factor_th_interp + 0.001);
		
		    if(BetaUpdateR < GammaUpdateR)   /* Beta is closer to the threshold than gamma*/
		    {
			    BetaUpdateVal = BetaUpdateVal*FalsePosUpW;
			    this->beta_thresh_update(GradOrg, BetaUpdateVal);	
		    }
		    else /* Gamma is closer to the threshold than beta*/
		    {
		        GammaUpdateVal = GammaUpdateVal*FalsePosUpW;
		        this->gamma_thresh_update(GradOrg, GammaUpdateVal);			
		    }
		}

		return true; // true means that  false positive case is detected
		
	}		

	return false; // false means that there is no false positive
}


void STAIR_DETEC_ONLINE_FUNC::visualize_threshold(const cv::Mat& rgb_input)
{
    // Create values for your x-axis
    sciplot::Vec x;
	
	std::valarray<float> x_beta(5), y_beta(5);
	
	x_beta[0] = 0;
	x_beta[1] = this->_online_rule_base_depth_y_error_th_x1;
	x_beta[2] = this->_online_rule_base_depth_y_error_th_x2;
	x_beta[3] = this->_online_rule_base_depth_y_error_th_x3;
	x_beta[4] = this->_online_rule_base_depth_y_error_th_x3 +0.5;
	
	
	y_beta[0] = this->_online_rule_base_depth_y_error_th_y1;
	y_beta[1] = this->_online_rule_base_depth_y_error_th_y1;
	y_beta[2] = this->_online_rule_base_depth_y_error_th_y2;
	y_beta[3] = this->_online_rule_base_depth_y_error_th_y3;
	y_beta[4] = this->_online_rule_base_depth_y_error_th_y3;
	
	float max_depth_y_error = std::max({this->_online_rule_base_depth_y_error_th_y1, this->_online_rule_base_depth_y_error_th_y2, this->_online_rule_base_depth_y_error_th_y3});  //#include <initializer_list> is required

    // Create a Plot object
    sciplot::Plot2D plot1;
    // Set color palette for first Plot
    //plot1.palette("paired");
	plot1.xrange(0.0,this->_online_rule_base_depth_y_error_th_x3 +0.5);
	plot1.yrange(0.0,max_depth_y_error*2);
	
	plot1.xlabel("gradient m");
	plot1.ylabel("beta threshold");
	
	plot1.size(360, 200);
	
	sciplot::LegendSpecs& plot1_legend = plot1.legend();
	plot1_legend.atOutsideTopLeft();
    
    plot1.drawCurve(x_beta, y_beta).label("beta Th").lineWidth(2);
    
	plot1.drawPoints(this->PlotGradBeta4OnLng_, this->PlotBeta4OnLng_).label("beta points").pointSize(2).pointType(1);

	std::valarray<float> x_gamma(5), y_gamma(5);
	
	x_gamma[0] = 0;
	x_gamma[1] = this->_online_rule_base_grad_diff_th_x1;
	x_gamma[2] = this->_online_rule_base_grad_diff_th_x2;
	x_gamma[3] = this->_online_rule_base_grad_diff_th_x3;
	x_gamma[4] = this->_online_rule_base_grad_diff_th_x3 +0.5;
	
	
	y_gamma[0] = this->_online_rule_base_grad_diff_th_y1;
	y_gamma[1] = this->_online_rule_base_grad_diff_th_y1;
	y_gamma[2] = this->_online_rule_base_grad_diff_th_y2;
	y_gamma[3] = this->_online_rule_base_grad_diff_th_y3;
	y_gamma[4] = this->_online_rule_base_grad_diff_th_y3;	

	float max_grad_diff = std::max({this->_online_rule_base_grad_diff_th_y1, this->_online_rule_base_grad_diff_th_y2, this->_online_rule_base_grad_diff_th_y3});  //#include <initializer_list> is required

    // Create a second Plot object
    sciplot::Plot2D plot2;
	plot2.xlabel("gradient m");
	plot2.ylabel("gamma threshold");
	
	plot2.size(360, 200);

	plot2.xrange(0.0,this->_online_rule_base_grad_diff_th_x3 +0.5);
	plot2.yrange(0.0,max_grad_diff*2);	
	
	sciplot::LegendSpecs& plot2_legend = plot2.legend();
	plot2_legend.atOutsideTopLeft();
    // Draw a tangent graph putting x on the x-axis and tan(x) on the y-axis
    plot2.drawCurve(x_gamma, y_gamma).label("gamma Th").lineWidth(2);
	
	plot2.drawPoints(this->PlotGradGamma4OnLng_, this->PlotGamma4OnLng_).label("gamma points").pointSize(1).pointType(1);
    
	
	/******************* plot 3 ************************************/
	
	std::vector<float> PlotDelBetaP4BetaOnLngInd_;
	
	for (int i = 0; i< PlotDelBetaP4BetaOnLng_.size(); i++)
	{
		PlotDelBetaP4BetaOnLngInd_.push_back(i+1);
	}
	
	sciplot::Plot2D plot3;
	
	plot3.palette("paired"); // setup color for plot 3
	
    plot3.xlabel("Index");
	plot3.ylabel("Probability of delta beta for beta Thresh");
	
	plot3.xrange(0.0,PlotDelBetaP4BetaOnLng_.size() + 1);
	plot3.yrange(-0.2,1.2);
	
	sciplot::LegendSpecs& plot3_legend = plot3.legend();
	plot3_legend.atOutsideTopLeft();
	
	plot3.drawPoints(PlotDelBetaP4BetaOnLngInd_, this->PlotDelBetaP4BetaOnLng_).label("Probability of Delta Beta for Beta").pointSize(1).pointType(5);
	

	/******************* plot 4 ************************************/
	
	std::vector<float> PlotDelGammaP4BetaOnLngInd_;
	
	for (int i = 0; i< PlotDelGammaP4BetaOnLng_.size(); i++)
	{
		PlotDelGammaP4BetaOnLngInd_.push_back(i+1);
	}
	
	sciplot::Plot2D plot4;
	
	plot4.palette("paired"); // setup color for plot 4
	
    plot4.xlabel("Index");
	plot4.ylabel("Probability of delta gamma for beta Thresh");
	
	plot4.xrange(0.0,PlotDelGammaP4BetaOnLng_.size() + 1);
	plot4.yrange(-0.2,1.2);	

	sciplot::LegendSpecs& plot4_legend = plot4.legend();
	plot4_legend.atOutsideTopLeft();
	
	plot4.drawPoints(PlotDelGammaP4BetaOnLngInd_, this->PlotDelGammaP4BetaOnLng_).label("Probability of Delta Gamma for Beta").pointSize(1).pointType(5);	


	/******************* plot 5 ************************************/
	
	std::vector<float> PlotDelBetaP4GammaOnLngInd_;
	
	for (int i = 0; i< PlotDelBetaP4GammaOnLng_.size(); i++)
	{
		PlotDelBetaP4GammaOnLngInd_.push_back(i+1);
	}
	
	sciplot::Plot2D plot5;
	
    plot5.palette("set2"); // setup color for plot 5
	
    plot5.xlabel("Index");
	plot5.ylabel("Probability of delta beta for gamma Thresh");
	
	plot5.xrange(0.0,PlotDelBetaP4GammaOnLng_.size() + 1);
	plot5.yrange(-0.2,1.2);	
	
	sciplot::LegendSpecs& plot5_legend = plot5.legend();
	plot5_legend.atOutsideTopLeft();
	
	plot5.drawPoints(PlotDelBetaP4GammaOnLngInd_, this->PlotDelBetaP4GammaOnLng_).label("Probability of Delta Beta for Gamma").pointSize(1).pointType(5);	


	/******************* plot 6 ************************************/

	std::vector<float> PlotDelGammaP4GammaOnLngInd_;
	
	for (int i = 0; i< PlotDelGammaP4GammaOnLng_.size(); i++)
	{
		PlotDelGammaP4GammaOnLngInd_.push_back(i+1);
	}
	
	sciplot::Plot2D plot6;
	
	plot6.palette("set2"); // setup color for plot 6
	
    plot6.xlabel("Index");
	plot6.ylabel("Probability of delta gamma for gamma Thresh");
	
	plot6.xrange(0.0,PlotDelGammaP4GammaOnLng_.size() + 1);
	plot6.yrange(-0.2,1.2);
	
	sciplot::LegendSpecs& plot6_legend = plot6.legend();
	plot6_legend.atOutsideTopLeft();
	
	plot6.drawPoints(PlotDelGammaP4GammaOnLngInd_, this->PlotDelGammaP4GammaOnLng_).label("Probability of Delta Gamma for Gamma").pointSize(1).pointType(5);	
	
	
    // Put both plots in a "figure" 2*3 sections
    sciplot::Figure figure = {{plot1, plot2},{plot3, plot4},{plot5, plot6}};
	
	figure.palette("dark2");

    // Create a canvas / drawing area to hold figure and plots
    sciplot::Canvas canvas = {{figure}};
    // Set color palette for all Plots that do not have a palette set (plot2) / the default palette
    //canvas.defaultPalette("set1");
	
	int width = 800;
	int height = 1200;
	
	canvas.size(width, height);

    // Show the canvas in a pop-up window
    //canvas.show();
	
	
	cv::imshow("stair case ROI", rgb_input);
	cv::waitKey(5);
	
	//canvas.windowshutdown();

}
