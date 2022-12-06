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
#include <tf2_ros/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <sensor_msgs/CameraInfo.h>
#include <boost/bind.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


std::mutex _lock;

bool testing_set_end = false;
bool training_set_end = false;

cv::Mat image_read_test_rgb;
cv::Mat image_read_test_depth;

class TRAINING_TEST_SET_ROS{
	
	private:
	 
	     int dummy;
		 
		 ros::NodeHandle main_nh;
         ros::NodeHandle param_nh;
	     ros::Rate* _loop_rate;
		 image_transport::ImageTransport  _it;
		 image_transport::Subscriber _rgb_sub;
		 image_transport::Subscriber _depth_sub;
		
         ros::Subscriber _rgb_cam_info_sub;
		 ros::Subscriber _depth_cam_info_sub;
		 
		 message_filters::Subscriber<sensor_msgs::Image> _rgb_msg_filter_sub;
		 message_filters::Subscriber<sensor_msgs::Image> _depth_msg_filter_sub;
		 
		 cv::Mat _rgb_image;
		 cv::Mat _depth_image;
		 cv::Mat _stair_case_output_image;
		 
		 std::string  _training_set_dir;
		 std::string  _training_set_rgb_dir;
		 std::string  _training_set_depth_dir;
		 std::string  _test_set_dir;
		 std::string  _test_set_rgb_dir;
		 std::string  _test_set_depth_dir;
		 std::string  _rgb_image_file_name;
		 std::string  _depth_image_file_name;
		 std::string  _rgb_image_extension;
		 std::string  _depth_image_extension;
		 
		 std::ofstream _training_file_rgb_list;
		 std::ofstream _training_file_depth_list;
		 std::ofstream _test_file_rgb_list;
		 std::ofstream _test_file_depth_list;
		 
		 int _rgb_count = 0;
		 int _depth_count = 0;
		 int _training_set_save_count = 0;
		 int _test_set_save_count = 0;
		 
		 float _num_of_training_set;
		 float _num_of_test_set;
		 
		 float _dscale;
		 float _fx, _fy, _px, _py;
		
		 ros::Time _start_time;
		 
		 int _update_rate;
		 float  _collection_time;
		 float  _percent_of_test_set;
		 
		 
	
	
	public:
	    
		bool _save_rgb_depth_image_file;
		
	    void run();
		void pre_proc_run();
		void rgb_depth_image_callback(const sensor_msgs::Image::ConstPtr& msgRGB, const sensor_msgs::Image::ConstPtr& msgD); 
		void rgb_image_callback(const sensor_msgs::Image::ConstPtr& msg);
		void depth_image_callback(const sensor_msgs::Image::ConstPtr& msg);
		void rgb_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg, int value);
		void depth_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg, int value);
		void imread_test_func();
	 
		/*constructor and destructor*/
	    TRAINING_TEST_SET_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh);
	    ~TRAINING_TEST_SET_ROS();

};

TRAINING_TEST_SET_ROS::TRAINING_TEST_SET_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh):main_nh(m_nh),param_nh(p_nh),_it(m_nh)
{

	
	 std::string  rgb_image_topic = "front_cam/camera/color/image_raw";
	 std::string  depth_image_topic = "front_cam/camera/depth/image_rect_raw";
	 std::string camera_rgb_info_topic = "front_cam/camera/color/camera_info";
	 std::string camera_depth_info_topic = "front_cam/camera/depth/camera_info";
	 
	 std::string  training_set_dir = "/home/kangneoung/stair_detection/src/stair_detection/image_set/training";
	 std::string  test_set_dir ="/home/kangneoung/stair_detection/src/stair_detection/image_set/testing";
	 
	 
	 std::string rgb_image_file_name ="stair1_rgb";
	 std::string depth_image_file_name ="stair1_depth";
	 
	 std::string rgb_image_extension =".png";
	 std::string depth_image_extension =".png";
	 
	 param_nh.getParam("depth_image_topic",depth_image_topic);
	 param_nh.getParam("rgb_image_topic",rgb_image_topic);
	 param_nh.getParam("camera_depth_info_topic",camera_depth_info_topic);
	 param_nh.getParam("camera_rgb_info_topic",camera_rgb_info_topic);
	 param_nh.getParam("training_set_dir",training_set_dir);
	 param_nh.getParam("test_set_dir",test_set_dir);
	 
	 param_nh.getParam("rgb_image_file_name",rgb_image_file_name);
	 param_nh.getParam("depth_image_file_name",depth_image_file_name);
	 param_nh.getParam("rgb_image_extension",rgb_image_extension);
	 param_nh.getParam("depth_image_extension",depth_image_extension);
	 
	 
	 this->_training_set_dir = training_set_dir;
	 this->_training_set_rgb_dir = this->_training_set_dir + "/" + "rgb";
	 this->_training_set_depth_dir = this->_training_set_dir + "/" + "depth";
	 this->_test_set_dir = test_set_dir;
	 this->_test_set_rgb_dir = this->_test_set_dir + "/" + "rgb"; 
	 this->_test_set_depth_dir = this->_test_set_dir + "/" + "depth";
	 this->_rgb_image_file_name = rgb_image_file_name;
	 this->_depth_image_file_name = depth_image_file_name;
	 this->_rgb_image_extension = rgb_image_extension;
	 this->_depth_image_extension = depth_image_extension;
	 
	 int update_rate = 10;
     float collection_time = 30;
	 float  percent_of_test_set = 0.2;
	 
	 bool save_rgb_depth_image_file = true;
	  
	 param_nh.getParam("update_rate",update_rate);
	 param_nh.getParam("collection_time",collection_time);
	 param_nh.getParam("percent_of_test_set",percent_of_test_set);
	 param_nh.getParam("save_rgb_depth_image_file",save_rgb_depth_image_file);
     
	 this->_update_rate = update_rate;
	 this->_collection_time = collection_time;
	 this->_percent_of_test_set = percent_of_test_set;
	 
	 this->_num_of_training_set = (1-percent_of_test_set)*10;
	 this->_num_of_test_set = percent_of_test_set*10;
	 
	 this->_save_rgb_depth_image_file = save_rgb_depth_image_file;

	 //this->_rgb_sub =  _it.subscribe(rgb_image_topic, 1,&TRAINING_TEST_SET_ROS::rgb_image_callback,this);
	 //this->_depth_sub =  _it.subscribe(depth_image_topic, 1,&TRAINING_TEST_SET_ROS::depth_image_callback,this);
	
	
	// typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
	
	 //this->_rgb_msg_filter_sub.subscribe(main_nh,rgb_image_topic,1);
	 //this->_depth_msg_filter_sub.subscribe(main_nh,depth_image_topic,1);
	 
	 // ApproximateTime takes a queue size as its constructor argument, hence sync_pol(10)
	// message_filters::Synchronizer<sync_pol> sync(sync_pol(10), this->_rgb_msg_filter_sub,this->_depth_msg_filter_sub);
	
	 //sync.registerCallback(boost::bind(&TRAINING_TEST_SET_ROS::rgb_depth_image_callback,this,_1,_2));
	 
	 this->_rgb_cam_info_sub = main_nh.subscribe<sensor_msgs::CameraInfo>(camera_rgb_info_topic, 1, boost::bind(&TRAINING_TEST_SET_ROS::rgb_cam_info_callback, this,_1,0));
	 this->_depth_cam_info_sub = main_nh.subscribe<sensor_msgs::CameraInfo>(camera_depth_info_topic, 1, boost::bind(&TRAINING_TEST_SET_ROS::depth_cam_info_callback, this,_1,0));
	 

     this->_loop_rate = new ros::Rate(update_rate);	
	
	
	 std::string training_rgb_txt_file_dir =  this->_training_set_rgb_dir + "/" + "training_rgb_img_file_list.txt";
	 std::string training_depth_txt_file_dir =  this->_training_set_depth_dir + "/" + "training_depth_img_file_list.txt";
	 std::string  test_rgb_txt_file_dir = this->_test_set_rgb_dir + "/" + "test_rgb_img_file_list.txt";
	 std::string  test_depth_txt_file_dir = this->_test_set_depth_dir + "/" + "test_depth_img_file_list.txt";
	
     this->_training_file_rgb_list.open(training_rgb_txt_file_dir, std::ofstream::app);
	 this->_training_file_depth_list.open(training_depth_txt_file_dir, std::ofstream::app);
     this->_test_file_rgb_list.open(test_rgb_txt_file_dir, std::ofstream::app);
	 this->_test_file_depth_list.open(test_depth_txt_file_dir, std::ofstream::app);
	
	 
}

TRAINING_TEST_SET_ROS::~TRAINING_TEST_SET_ROS()
{
	 this->_training_file_rgb_list.close();
	 this->_training_file_depth_list.close();
     this->_test_file_rgb_list.close();
	 this->_test_file_depth_list.close();
	
	delete this->_loop_rate;
}


void TRAINING_TEST_SET_ROS::rgb_depth_image_callback(const sensor_msgs::Image::ConstPtr& msgRGB, const sensor_msgs::Image::ConstPtr& msgD)
{
	std::string str_rgb_index;
	std::string str_depth_index;
	
	std::string full_rgb_training_data_dir;
	std::string full_depth_training_data_dir;
	
	std::string full_rgb_test_data_dir;
	std::string full_depth_test_data_dir;
	
	std::string rgb_data_file_name;
	std::string depth_data_file_name;
   
	
	bool initial_start = false;
	bool skip_once = false;
	
	cv::Mat rgb_image;
	rgb_image = cv_bridge::toCvShare(msgRGB, "bgr8")->image;
	this->_rgb_image = rgb_image.clone();
	
	this->_rgb_count = this->_rgb_count + 1 ;
	
	cv::Mat depth_image = cv_bridge::toCvCopy(msgD)->image;
	this->_depth_image = depth_image.clone();
	
	this->_depth_count = _depth_count + 1 ;
	
	if((this->_rgb_count==1)&&(this->_depth_count==1))
	{
	   this->_start_time=ros::Time::now();
	   initial_start = true;
	}
	else if((this->_rgb_count==0)||(this->_depth_count==0))
	{
	   initial_start = false;
	}
	else
	{
		initial_start = true;
	}
	
	if(this->_save_rgb_depth_image_file==false)
	{
		initial_start = false;
	}
	
	
	ros::Duration elapse_time = ros::Time::now() - this->_start_time;
	
	if((elapse_time>ros::Duration(this->_collection_time))||(initial_start==false))
	{
		/*intentionally empty*/
	}
	else
	{
		
	    std::cout<<"this->_depth_count : "<<this->_depth_count<<"\n"<<std::endl;
	    std::cout<<"this->_rgb_count : "<<this->_rgb_count<<"\n"<<std::endl;
		
		//int temp_rgb_index = 1000 - this->_rgb_count;
		
		str_rgb_index = std::to_string(this->_rgb_count); //std::to_string(temp_rgb_index);//
		str_depth_index = std::to_string(this->_rgb_count); //std::to_string(temp_rgb_index);//
		
		full_rgb_training_data_dir =  this->_training_set_rgb_dir + "/" +  this->_rgb_image_file_name + "_" + str_rgb_index + this->_rgb_image_extension;
		full_depth_training_data_dir = this->_training_set_depth_dir + "/" +  this->_depth_image_file_name + "_" + str_depth_index + this->_depth_image_extension;
		 
		full_rgb_test_data_dir = this->_test_set_rgb_dir + "/" +  this->_rgb_image_file_name + "_" + str_rgb_index + this->_rgb_image_extension;
		full_depth_test_data_dir = this->_test_set_depth_dir + "/" + this->_depth_image_file_name + "_" + str_depth_index + this->_depth_image_extension;
		
		rgb_data_file_name = this->_rgb_image_file_name + "_" + str_rgb_index + this->_rgb_image_extension;
		depth_data_file_name = this->_depth_image_file_name + "_" + str_depth_index + this->_depth_image_extension;

		
		//std::cout<<"full_rgb_training_data_dir : "<<full_rgb_training_data_dir<<"\n"<<std::endl;
		//std::cout<<"full_depth_training_data_dir : "<<full_depth_training_data_dir<<"\n"<<std::endl;
		//std::cout<<"full_rgb_test_data_dir : "<<full_rgb_test_data_dir<<"\n"<<std::endl;
		//std::cout<<"full_depth_test_data_dir : "<<full_depth_test_data_dir<<"\n"<<std::endl;
		
		if(this->_percent_of_test_set == 0)
		{
			this->_training_file_rgb_list<< rgb_data_file_name <<"\n";
			this->_training_file_depth_list<< depth_data_file_name <<"\n";
			
			cv::imwrite( full_rgb_training_data_dir, this->_rgb_image);
			cv::imwrite( full_depth_training_data_dir, this->_depth_image);
			
			this->_training_set_save_count = this->_training_set_save_count + 1;
			
			return;
		}

		
		if(((this->_depth_count==1)&&(this->_rgb_count==1))||((testing_set_end==true)&&(skip_once==false)))
		{
			
			if((this->_depth_count==1)&&(this->_rgb_count==1))
			{
				testing_set_end=true;
			}
		
			//std::cout<<"_training_set_save_count : "<<this->_training_set_save_count<<"\n"<<std::endl;
			
			this->_training_file_rgb_list<< rgb_data_file_name <<"\n";
			this->_training_file_depth_list<< depth_data_file_name <<"\n";
			
			cv::imwrite( full_rgb_training_data_dir, this->_rgb_image);
			cv::imwrite( full_depth_training_data_dir, this->_depth_image);
			
			this->_training_set_save_count = this->_training_set_save_count + 1;
			
					
		    if(this->_training_set_save_count> this->_num_of_training_set)
		    {
			    training_set_end = true;
			    testing_set_end = false;
				skip_once = true;
			    this->_training_set_save_count = 0;
		    }
			//training_set_end = false;
		}

			
		if((training_set_end==true)&&(skip_once==false))
		{
			//std::cout<<"_test_set_save_count : "<<this->_test_set_save_count<<"\n"<<std::endl;
			
			this->_test_file_rgb_list<< rgb_data_file_name <<"\n";
			this->_test_file_depth_list<< depth_data_file_name <<"\n";
			
			cv::imwrite( full_rgb_test_data_dir, this->_rgb_image);
			cv::imwrite( full_depth_test_data_dir, this->_depth_image);
   
			this->_test_set_save_count = this->_test_set_save_count + 1;
			   
			if(this->_test_set_save_count> this->_num_of_test_set)
		    {
			   training_set_end =false;
			   testing_set_end = true;
			   this->_test_set_save_count = 0;
		    }

		}
	}

	
}


void TRAINING_TEST_SET_ROS::rgb_image_callback(const sensor_msgs::Image::ConstPtr& msg )
{
	//std::lock_guard<std::mutex> lock(_lock);
	
	cv::Mat rgb_image;
	rgb_image = cv_bridge::toCvShare(msg, "bgr8")->image;
	this->_rgb_image = rgb_image.clone();
	
	
}


void TRAINING_TEST_SET_ROS::depth_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
	//std::lock_guard<std::mutex> lock(_lock);
	cv::Mat depth_image = cv_bridge::toCvCopy(msg)->image;
	this->_depth_image = depth_image.clone();
	
}

void TRAINING_TEST_SET_ROS::rgb_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg ,int value)
{
          /*empty*/
		  
    if(value > 0)
	{
		ROS_INFO("rgb_cam_info_callback");
	}
}

void TRAINING_TEST_SET_ROS::depth_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg ,int value)
{        
          this->_fx = msg->K[0];
          this->_fy = msg->K[4];
          this->_px = msg->K[2];
          this->_py = msg->K[5];
	      this->_dscale = msg->D[0];
		  
	if(value > 0)
	{
		ROS_INFO("depth_cam_info_callback");
	}
}


void TRAINING_TEST_SET_ROS::imread_test_func()
{  
     unsigned short depth_tmp2;
	 int starting_point_y;
	 
	 std::string rgb_reading_dir = "/home/kangneoung/stair_detection/src/stair_detection/image_set/training/stair1_rgb_1.png";
	 std::string depth_reading_dir = "/home/kangneoung/stair_detection/src/stair_detection/image_set/training/stair1_depth_1.png";
	 
	 image_read_test_rgb=cv::imread(rgb_reading_dir,cv::IMREAD_COLOR);
	 image_read_test_depth=cv::imread(depth_reading_dir,cv::IMREAD_ANYDEPTH);
	 
	 		
	 cv::imshow("image_read_test_rgb", image_read_test_rgb);
	 cv::waitKey(30);
	 
	 cv::imshow("image_read_test_depth", image_read_test_depth);
	 cv::waitKey(30);
	 
	for (starting_point_y=50; starting_point_y<300;starting_point_y=starting_point_y+5)
	{	 
	    depth_tmp2 =  image_read_test_depth.at<unsigned short>(starting_point_y,424);
		std::cout<<"depth_tmp2 : "<<depth_tmp2<<"\n"<<std::endl;
	}
	 
}


void TRAINING_TEST_SET_ROS::pre_proc_run()
{     
   	   //this->imread_test_func();
}


void TRAINING_TEST_SET_ROS::run()
{  
    while(ros::ok())
	{
	   this->pre_proc_run();
	   
	   //ros::spin();
	   ros::spinOnce();
	   this->_loop_rate->sleep();
	}
	
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "training_test_set_calssifier");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");
  
  TRAINING_TEST_SET_ROS  training_test_set_ros(nh,_nh);
  
  //if(training_test_set_ros._save_rgb_depth_image_file == true)
  //{
     message_filters::Subscriber<sensor_msgs::Image> rgb_msg_filter_sub_test(nh,"front_cam/camera/color/image_raw",5);
     message_filters::Subscriber<sensor_msgs::Image> depth_msg_filter_sub_test(nh,"front_cam/camera/depth/image_rect_raw",5);
     typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  
      // Approximate time policy dosen't work in class but it works in main function, debug is needed, kangneoung lee
      // ApproximateTime takes a queue size as its constructor argument, hence sync_pol(10)
     message_filters::Synchronizer<sync_pol> sync(sync_pol(5), rgb_msg_filter_sub_test,depth_msg_filter_sub_test);
    
     sync.registerCallback(boost::bind(&TRAINING_TEST_SET_ROS::rgb_depth_image_callback,&training_test_set_ros,_1,_2));
 // }
  
  
  
  training_test_set_ros.run();
 
  //image_transport::ImageTransport it(nh);
  //image_transport::Subscriber sub = it.subscribe("front_cam/camera/depth/image_rect_raw", 1, imageCallback);
  //ros::spin();
   return 0;
}
