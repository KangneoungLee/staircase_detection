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

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include<sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>   /*atof*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

float num1;
float num2;
float num3;


enum training_element
{
	GRADIENT, //0
	GRADIENT_DIFF,//1
	AVG_DEPTH_Y_ERROR, //2
	
	//AVG_X_ERROR, //2
	
	NUMBER_OF_ELEMENT //3
};

float gradient_range[2] = {0.4, 1.8};
float gradient_diff_range[2] = {0.15, 0.5};
float depth_y_error_sum_range[2]={1, 4}; 


class SAMPLE_COR_SVM_ROS{

	private:
	 	 
		 ros::NodeHandle main_nh;
         ros::NodeHandle param_nh;
	     ros::Rate* _loop_rate;
		
		 std:: ofstream _sample_in;
		 
		
		 cv::Ptr<cv::ml::SVM> classifier;
		 
		 std::string  _file_name;
		 std::string  _result_save_dir;
		 
		 float _delta_gradient;
		 float _delta_gradient_diff;
		 float _delta_depth_y_error_sum;
		 float _gradient_range_min;
		 float _gradient_range_max;
		 float _gradient_diff_range_min;
		 float _gradient_diff_range_max;
		 float _depth_y_error_sum_range_min;
		 float _depth_y_error_sum_range_max;
		 
		 int _anchor_factor_index;
			 
	public:
	    
		 void run();
		 void sample_generation();
		 float predict_wraper(float gradient, float gradient_diff, float depth_y_error);
		 
		 /*constructor and destructor*/
	    SAMPLE_COR_SVM_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh);
	    ~SAMPLE_COR_SVM_ROS();

};

SAMPLE_COR_SVM_ROS::SAMPLE_COR_SVM_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh):main_nh(m_nh),param_nh(p_nh)
{     
     
	 std::string result_save_dir = "/home/kangneoung/stair_detection/src/stair_detection/sample_correlation";
	 
	 float delta_gradient = 0.1;
	 float delta_gradient_diff = 0.025;
	 float delta_depth_y_error_sum = 0.2;
	 int anchor_factor_index = 1;    /*1: gradient anchor  2 : gradient diff anchor  3: depth y error sum anchor*/
	 float gradient_range_min = 0.3;
	 float gradient_range_max = 1.8;
	 float gradient_diff_range_min = 0.15;
	 float gradient_diff_range_max = 0.5;
	 float depth_y_error_sum_range_min = 0.2;
	 float depth_y_error_sum_range_max = 4;
	 
	 param_nh.getParam("result_save_dir",result_save_dir);
	 param_nh.getParam("delta_gradient",delta_gradient);
	 param_nh.getParam("delta_gradient_diff",delta_gradient_diff);
	 param_nh.getParam("delta_depth_y_error_sum",delta_depth_y_error_sum);
	 param_nh.getParam("anchor_factor_index",anchor_factor_index); /*1: gradient anchor  2 : gradient diff anchor  3: depth y error sum anchor*/
	 param_nh.getParam("gradient_range_min",gradient_range_min);
	 param_nh.getParam("gradient_range_max",gradient_range_max);
	 param_nh.getParam("gradient_diff_range_min",gradient_diff_range_min);
	 param_nh.getParam("gradient_diff_range_max",gradient_diff_range_max);
	 param_nh.getParam("depth_y_error_sum_range_min",depth_y_error_sum_range_min);
	 param_nh.getParam("depth_y_error_sum_range_max",depth_y_error_sum_range_max);
	 
	 this->_result_save_dir = result_save_dir;
	 this->_delta_gradient = delta_gradient;
	 this->_delta_gradient_diff = delta_gradient_diff;
	 this->_delta_depth_y_error_sum = delta_depth_y_error_sum;
	 this->_anchor_factor_index = anchor_factor_index; /*1: gradient anchor  2 : gradient diff anchor  3: depth y error sum anchor*/
	 this->_gradient_range_min = gradient_range_min;
	 this->_gradient_range_max = gradient_range_max;
	 this->_gradient_diff_range_min = gradient_diff_range_min;
	 this->_gradient_diff_range_max = gradient_diff_range_max;
	 this->_depth_y_error_sum_range_min = depth_y_error_sum_range_min;
	 this->_depth_y_error_sum_range_max = depth_y_error_sum_range_max;
	 
	 gradient_range[0] = this->_gradient_range_min;
	 gradient_range[1] = this->_gradient_range_max;
	 gradient_diff_range[0] = this->_gradient_diff_range_min;
	 gradient_diff_range[1] = this->_gradient_diff_range_max;
	 depth_y_error_sum_range[0] = this->_depth_y_error_sum_range_min;
	 depth_y_error_sum_range[1] = this->_depth_y_error_sum_range_max;
	 
	 
	 if(this->_anchor_factor_index == 1)
	 {
		 this->_file_name = "gradient_anchor_correlation.txt";
	 }
	 else if(this->_anchor_factor_index == 2)
	 {
		 this->_file_name = "gradient_diff_anchor_correlation.txt";
	 }
	 else if(this->_anchor_factor_index == 3)
	 {
		 this->_file_name = "depth_y_error_sum_anchor_correlation.txt";
	 }
	 
	 
	 std::string full_txt_dir = this->_result_save_dir + "/" + this->_file_name;
	
	 this->_sample_in.open(full_txt_dir,std::ofstream::app);
	
	 /*1: gradient anchor  2 : gradient diff anchor  3: depth y error sum anchor*/
     
	 this->_sample_in<<"gradient / "<<"gradient diff / "<<"depth y error sum  "<<"\n"<<std::endl;


	 int update_rate = 50;
	   
	 this->_loop_rate = new ros::Rate(update_rate);
          
	 try
	 {
		 classifier = cv::Algorithm::load<cv::ml::SVM>("/home/kangneoung/stair_detection/src/stair_detection/svm_train.xml");
	 }
	 catch(int e)
	 {
		ROS_ERROR("check the directory for svm parameters ");
	 }
	   
}

SAMPLE_COR_SVM_ROS::~SAMPLE_COR_SVM_ROS()
{   
    _sample_in.close();

	delete this->_loop_rate;

}


void SAMPLE_COR_SVM_ROS::sample_generation()
{	   

	 float response = 0, response_old =0;
	
	 float gradient_incremental, gradient_diff_incremental, delta_depth_y_error_sum_incremental;
	 
	 float gradient_incremental_old = 0, gradient_diff_incremental_old =0, delta_depth_y_error_sum_incremental_old =0;
	 
     float anchor, anchor_min, anchor_max, delta_anchor;	
     
	 float factor1, factor1_min,  factor1_max, delta_factor1, factor1_greedy; 
     float factor2, factor2_min,  factor2_max, delta_factor2, factor2_greedy; 	 
	

	 /*1: gradient anchor  2 : gradient diff anchor  3: depth y error sum anchor*/	
	 if(this->_anchor_factor_index == 1)
	 {
		anchor_min =  gradient_range[0];
		anchor_max =  gradient_range[1];
		delta_anchor = this->_delta_gradient;
		
		factor1_min = gradient_diff_range[0];
		factor1_max = gradient_diff_range[1];
		delta_factor1 =  this->_delta_gradient_diff;
		factor1_greedy = 0.1;
		
		factor2_min = depth_y_error_sum_range[0];
		factor2_max = depth_y_error_sum_range[1];
		delta_factor2 = this->_delta_depth_y_error_sum;
		factor2_greedy = 0.5;
		
	 }
	 else if(this->_anchor_factor_index == 2)
	 {
		anchor_min =  gradient_diff_range[0];
		anchor_max =  gradient_diff_range[1];
		delta_anchor = this->_delta_gradient_diff;
		
		factor1_min = gradient_range[0];
		factor1_max = gradient_range[1];
		delta_factor1 =  this->_delta_gradient;
		factor1_greedy = 1;
		
		factor2_min = depth_y_error_sum_range[0];
		factor2_max = depth_y_error_sum_range[1];
		delta_factor2 = this->_delta_depth_y_error_sum;
		factor2_greedy = 0.5;
	 }
	 else if(this->_anchor_factor_index == 3)
	 {
		anchor_min = depth_y_error_sum_range[0];
		anchor_max = depth_y_error_sum_range[1];
		delta_anchor = this->_delta_depth_y_error_sum;
		
		factor1_min = gradient_range[0];
		factor1_max = gradient_range[1];
		delta_factor1 =  this->_delta_gradient;
		factor1_greedy = 1;
		
		factor2_min =  gradient_diff_range[0];
		factor2_max =  gradient_diff_range[1];
		delta_factor2 = this->_delta_gradient_diff;
		factor2_greedy = 0.1;
	 }
	
	 float i,j,k;
	
	std::cout<<"anchor_min : "<<anchor_min<<"anchor_max : "<<anchor_max<<"delta_anchor : "<<delta_anchor<<std::endl ;
	std::cout<<"factor1_min : "<<factor1_min<<"factor1_max : "<<factor1_max<<"delta_factor1 : "<<delta_factor1<<std::endl ;
	std::cout<<"factor2_min : "<<factor2_min<<"factor2_max : "<<factor2_max<<"delta_factor2 : "<<delta_factor2<<std::endl ;
	
	int anchor_num = std::round((anchor_max-anchor_min)/delta_anchor);
	int factor1_num = std::round((factor1_max+factor1_greedy-factor1_min)/delta_factor1);
	int factor2_num = std::round((factor2_max+factor2_greedy-factor2_min)/delta_factor2);
	
		std::cout<<"anchor_num : "<<anchor_num<<"factor1_num : "<<factor1_num<<"factor2_num : "<<factor2_num<<std::endl ;
	
     for(i=0/*anchor_min*/;i<=anchor_num/*anchor_max*/;i=i+1/*delta_anchor*/)
	 {
        for(j=0/*factor1_min*/;j<=factor1_num/*(factor1_max+factor1_greedy)*/;j=j+1/*delta_factor1*/)
		{
			for(k=0/*factor2_min*/;k<=factor2_num/*(factor2_max+factor2_greedy)*/;k=k+1/*delta_factor2*/)
			{

				 if(this->_anchor_factor_index == 1)
				 {  
			         gradient_incremental =anchor_min + i*delta_anchor;
					 gradient_diff_incremental = factor1_min +j*delta_factor1;
					 delta_depth_y_error_sum_incremental = factor2_min + k*delta_factor2;
				 }
				 else if(this->_anchor_factor_index == 2)
				 {
					 gradient_incremental =factor1_min +j*delta_factor1;
					 gradient_diff_incremental = anchor_min + i*delta_anchor;
					 delta_depth_y_error_sum_incremental = factor2_min + k*delta_factor2;
				 }
				 else if(this->_anchor_factor_index == 3)
				 {
					 
					 gradient_incremental =  factor1_min +j*delta_factor1 ;
					 gradient_diff_incremental = factor2_min + k*delta_factor2;
					 delta_depth_y_error_sum_incremental =anchor_min + i*delta_anchor;
				 }
				
				 response = this->predict_wraper(gradient_incremental,gradient_diff_incremental,delta_depth_y_error_sum_incremental);
				 //response = this->predict(gradient_incremental,gradient_diff_incremental,delta_depth_y_error_sum_incremental);
				 
				  
				 
				 std::cout<<"gradient_incremental   :"<<gradient_incremental<<"gradient_diff_incremental   :"<<gradient_diff_incremental<<"delta_depth_y_error_sum_incremental   :"<<delta_depth_y_error_sum_incremental<<"\n"<<std::endl;
				 std::cout<<"response   :"<<response<<std::endl;
				 
				 if((response_old == 1)&&(response != 1))
				 {
					  std::cout<<"ok   :"<<std::endl;
					 this->_sample_in<<gradient_incremental_old<<"/"<<gradient_diff_incremental_old<< "/"<<delta_depth_y_error_sum_incremental_old<<std::endl;
				 }
				 
				 response_old = response;
				 
				 gradient_incremental_old = gradient_incremental;	
                 gradient_diff_incremental_old =  gradient_diff_incremental;
	             delta_depth_y_error_sum_incremental_old = delta_depth_y_error_sum_incremental;
				 
				 
			}
		}
	 }		 
		
}


float SAMPLE_COR_SVM_ROS::predict_wraper(float gradient, float gradient_diff, float depth_y_error)
{
		 float response;
		 float training_array_temp[1][3] = {gradient,gradient_diff, depth_y_error};
		 cv::Mat test_mat(1, 3, CV_32F,training_array_temp);/*training_array_temp*/
		 response = classifier->predict(test_mat);
		 
		 return response;
}




void SAMPLE_COR_SVM_ROS::run()
{
	while(ros::ok())
	{ 
	 
       this->sample_generation(); 
		
	   ros::spin();
	   //ros::spinOnce();
	   this->_loop_rate->sleep();
	}
	
	
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "svm_training");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");
  
  
  SAMPLE_COR_SVM_ROS  smaple_cor_svm_ros(nh,_nh);
  
  smaple_cor_svm_ros.run();
 
  //image_transport::ImageTransport it(nh);
  //image_transport::Subscriber sub = it.subscribe("front_cam/camera/depth/image_rect_raw", 1, imageCallback);
  //ros::spin();
   return 0;
}
