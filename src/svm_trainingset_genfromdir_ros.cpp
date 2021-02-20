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

#include "stair_detection/stair_detec_cost_func.h"

#include <sys/stat.h> /*directory check*/
//#include <windows.h>  /*directory check*/


/*directory check function*/

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}


//bool dirExists(const std::string& dirName_in)
//{
// DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
//  if (ftyp == INVALID_FILE_ATTRIBUTES)
//   return false;  //something is wrong with your path!

//  if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
//    return true;   // this is a directory!

//  return false;    // this is not a directory!
//}

struct SelectedLine {
  // on off
  int init;
  int end;

  //initial coordination based on EVENT_LBUTTONDOWN
  int initX;
  int initY;

  // actual coordination 
  int actualX;
  int actualY;

}SelectedLine;

 static void CallBackF(int event, int x, int y, int flags, void* img) {
//Mouse Right button down
    SelectedLine.end = 0;
  if (event == cv::EVENT_RBUTTONDOWN) {
     SelectedLine.end = 1;
	std::cout << "right button " << std::endl;
    return;
  }
//Mouse Left button down
  if (event == cv::EVENT_LBUTTONDOWN) {
    SelectedLine.initX = x;
    SelectedLine.initY = y;
    SelectedLine.init = 1;
    std::cout << "left button DOWN" << std::endl; 
    return;
  }
//Mouse Left button up
  if (event == cv::EVENT_LBUTTONUP) {
    SelectedLine.actualX = x;
    SelectedLine.actualY = y;
    std::cout << "left button UP" << std::endl;
    return;
  }
 }

unsigned int string_read_count = 0;

class SVM_TRAINING_ROS{

	private:
	 	 
		 ros::NodeHandle main_nh;
         ros::NodeHandle param_nh;
	     ros::Rate* _loop_rate;
		 
		 std::string  _training_set_dir;
		 std::string  _training_set_rgb_image_list_file;
		 std::string  _training_set_depth_image_list_file;
		 
		 std:: ifstream rgb_in;
		 std:: ifstream depth_in;
		 
		 std::string _rgb_image_file;
		 std::string _depth_image_file;
		 
		 cv::Mat _rgb_image;
		 cv::Mat _depth_image;
		 
		 
		 float _dscale;
		 float _fx, _fy, _px, _py;
		 
		 bool _offline_svm_training = true;  /*write down text file for svm training*/
		 bool _image_open_ok = false; 		 
		 unsigned char reading_status;   /*0 : reading bad, 1 : reading (process), 2 : reading (success)*/
		 
		 bool _true_image_training = true; /*true : generate training set for true image,  false : generate training set for false image*/
		 
	public:
	    
		 void run();
		 void read_rgb_depth_file_list();
		 
		 STAIR_DETEC_COST_FUNC* str_det_cost_func;
		 
		 /*constructor and destructor*/
	    SVM_TRAINING_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh);
	    ~SVM_TRAINING_ROS();

};

SVM_TRAINING_ROS::SVM_TRAINING_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh):main_nh(m_nh),param_nh(p_nh)
{     
       
	   this->_training_set_dir = "/home/kangneoung/stair_detection/src/stair_detection/image_set/training";
	   this->_training_set_rgb_image_list_file = "training_rgb_img_file_list.txt";
	   this->_training_set_depth_image_list_file = "training_depth_img_file_list.txt";
	   
	   std::string full_dir_rgb_list = this->_training_set_dir +"/" +  this->_training_set_rgb_image_list_file ;
	   std::string full_dir_depth_list = this->_training_set_dir +"/" +  this->_training_set_depth_image_list_file ;
       
       try
	   {
	        rgb_in.open(full_dir_rgb_list);
			depth_in.open(full_dir_depth_list);
	    }
		catch(int e)
		{
			ROS_ERROR("check the directory for training data ");
		}
		
	   this->_fx = 425;
       this->_fy = 425;
       this->_px = 423;
       this->_py = 239;
	   this->_dscale = 0.001;
		
		
	   int update_rate = 10;
	   
	   this->str_det_cost_func = new STAIR_DETEC_COST_FUNC( this->_offline_svm_training);
	   
	   this->_loop_rate = new ros::Rate(update_rate);
       

}

SVM_TRAINING_ROS::~SVM_TRAINING_ROS()
{   
    rgb_in.close();
	depth_in.close();
     
	delete  this->str_det_cost_func;
	delete this->_loop_rate;

}

void SVM_TRAINING_ROS::read_rgb_depth_file_list()
{
    
	std::string rgb_s;
	std::string depth_s;
	
	std::getline(rgb_in,rgb_s);
	std::getline(depth_in,depth_s);
	
	 bool rgb_directory_check_flag = false;
	 bool depth_directory_check_flag = false;
	
	if((rgb_in.bad())||(depth_in.bad()))
	{
		reading_status = 0;
	}
    else if((rgb_in.eof())||(depth_in.eof()))
	{
		reading_status = 2;
	}
	else
	{
		reading_status = 1;
	}
	std::cout<<"reading_status   :"<<reading_status<<"\n"<<std::endl;

    std::istringstream rgb_ss(rgb_s);
	std::istringstream depth_ss(depth_s);
	
	std::string stringBuffer_rgb;
	std::string stringBuffer_depth;
	
     if(reading_status==1)    /*ignore first line  of text file*/ 
	 {  
	    std::getline(rgb_ss, stringBuffer_rgb);
		std::getline(depth_ss, stringBuffer_depth);
		
	    this->_rgb_image_file = this->_training_set_dir + "/" + stringBuffer_rgb;
	    this->_depth_image_file = this->_training_set_dir + "/" + stringBuffer_depth;
	 
	    std::cout<<" this->_rgb_image_file :"<<  this->_rgb_image_file<<"\n"<<std::endl;
	    std::cout<<" this->_depth_image_file :"<< this->_depth_image_file<<"\n"<<std::endl;
	 
	 
	    rgb_directory_check_flag = IsPathExist(this->_rgb_image_file);
	    depth_directory_check_flag = IsPathExist(this->_depth_image_file);
	 
	    std::cout<<" rgb_directory_check_flag :"<< rgb_directory_check_flag<<"\n"<<std::endl;
	    std::cout<<" depth_directory_check_flag :"<< depth_directory_check_flag<<"\n"<<std::endl;
		
	 }
	 

	 
	 this->_image_open_ok = false;
	 
	 if((rgb_directory_check_flag==true)&&(depth_directory_check_flag==true))
	 {
		 this->_image_open_ok = true;
	 }
	 
	  std::cout<<" this->_image_open_ok :"<< this->_image_open_ok<<"\n"<<std::endl;
	 
	 if(reading_status==1)
	 {
	    string_read_count++;
	 }
	
	 std::cout<<"string_read_count :"<<string_read_count<<"\n"<<std::endl;
	
	 if(reading_status!=1)
	 {
		 this->_image_open_ok = false;
	 }	
}


void SVM_TRAINING_ROS::run()
{
	while(ros::ok())
	{ 
	
		if(reading_status<2)
	    {
            this->read_rgb_depth_file_list();
	     }
		   
		 if(this->_image_open_ok)
		 {
			 std::cout<<" this->_image_open_ok :"<< "\n"<<std::endl;
			 this->_rgb_image = cv::imread(this->_rgb_image_file,cv::IMREAD_COLOR);
			 this->_depth_image = cv::imread(this->_depth_image_file,cv::IMREAD_ANYDEPTH);
			   	
			 SelectedLine.initX = 0;
			 SelectedLine.initY = 0;
			 SelectedLine.actualX = 0;
			 SelectedLine.actualY = 0;
			   
			   
			   
			 //cv::namedWindow("RGB Image");
			 cv::imshow("RGB Image",this->_rgb_image);
			 cv::setMouseCallback("RGB Image",CallBackF);
			 //cv::waitKey(20);
			   
             while(char(cv::waitKey(1)!='q')) //waiting for the 'q' key to finish the execution
             {
     
             }
			 
			 cv::destroyWindow("RGB Image");
			  
			 std::cout<<" SelectedLine.initX :" <<SelectedLine.initX<< "\n"<<std::endl;
			 std::cout<<" SelectedLine.initY :" <<SelectedLine.initY<< "\n"<<std::endl;
			 std::cout<<" SelectedLine.actualX :" <<SelectedLine.actualX<< "\n"<<std::endl;
			 std::cout<<" SelectedLine.actualY :" <<SelectedLine.actualY<< "\n"<<std::endl;
			  
			 int avg_X = std::round((SelectedLine.initX +  SelectedLine.actualX)/2 );
			  
			 std::cout<<" avg_X :" <<avg_X<< "\n"<<std::endl;
			  
			 //std::cout<<" this->_image_open_ok :"<< "\n"<<std::endl;
			   
			 cv::Mat rgb_image_confirm = this->_rgb_image.clone();
			 cv::line(rgb_image_confirm, cv::Point(avg_X, SelectedLine.initY), cv::Point(avg_X, SelectedLine.actualY), cv::Scalar(0, 0, 255));
			   
			 cv::imshow("RGB Image line_confirm",rgb_image_confirm);
			 
			 while(char(cv::waitKey(1)!='w')) //waiting for the 'w' key to finish the execution
             {
     
             }
			 
			 cv::destroyWindow("RGB Image line_confirm");
			 
			 if(this->_true_image_training)
             {			   
		         this->str_det_cost_func->cal_cost_wrapper_offline_image_true(this->_depth_image, avg_X, SelectedLine.initY, SelectedLine.actualY, this->_fx, this->_fy, this->_px, this->_py,  this->_dscale, 480, 848);
		     }
		     else
		     {
			     /*this->str_det_cost_func->cal_cost_wrapper_offline_image_false will be implemented*/
		     }   
		 } 
	}
	
	 ros::spinOnce();
	 this->_loop_rate->sleep();
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "svm_training");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");
    
  SVM_TRAINING_ROS  svm_training_ros(nh,_nh);
  
  svm_training_ros.run();
 
  //image_transport::ImageTransport it(nh);
  //image_transport::Subscriber sub = it.subscribe("front_cam/camera/depth/image_rect_raw", 1, imageCallback);
  //ros::spin();
   return 0;
}