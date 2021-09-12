#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <sensor_msgs/CameraInfo.h>
#include <boost/bind.hpp>
#include "stair_detection/stair_detec_pre_proc.h"
#include "stair_detection/stair_detec_cost_func.h"
#include "stair_detection/stair_detec_only_rgb.h"
#include <stair_custom_msg/debug_msg_1.h>
#include <stair_custom_msg/stairlocation.h>
#include <stair_custom_msg/pixel_stairlocation.h>
#include <std_msgs/Int8.h>

#include <sys/stat.h> /*directory check*/
//#include <windows.h>  /*directory check*/

std::mutex _lock;
int result_count = 0;
int img_ok_count = 0;
/*offline test variable*/
unsigned int string_read_count = 0;

/*offline test function*/ /*directory check function*/

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

 std:: ofstream _rgb_text_filter_in;
 std:: ofstream _depth_text_filter_in;
 std:: ofstream _not_detected_list;


class STAIR_DETECTION_ROS{
	
	private:
	 
	     int dummy;
		 
		 ros::NodeHandle main_nh;
         ros::NodeHandle param_nh;
	     ros::Rate* _loop_rate;
		 image_transport::ImageTransport  _it;
		 image_transport::Subscriber _rgb_sub;
		image_transport::Subscriber _depth_sub;
		image_transport::Publisher _rgb_roi_pub;
         ros::Subscriber _rgb_cam_info_sub;
		 ros::Subscriber _depth_cam_info_sub;
		 ros::Subscriber _MB_flag_sub;
		 ros::Subscriber  _detec_trigger_sub;
		 
		 int _MB_flag_status;
		 int _detec_trigger_flag;
		 
		 ros::Publisher _stair_pos_pub;
		 ros::Publisher _stair_center_pixel_pub;
		 
		 ros::Publisher _debug_msg_1_pub;
		 
		 cv::Mat _rgb_image;
		 cv::Mat _depth_image;
		 cv::Mat _stair_case_output_image;
		 
		 tf2_ros::Buffer tfBuffer;
		 tf2_ros::TransformListener tfListener;
		 
		 float _dscale;
		 float _fx, _fy, _px, _py;
		 
		 int _update_rate;
		 bool _encoding_rgb_flag;
		 bool _offline_svm_training;
		 bool _use_svm_classifier;
		 bool _offline_performance_test;
		 bool _use_only_rgb_to_detect_staircase;
		 unsigned short _preproc_resize_height;
		 unsigned short _preproc_resize_width;
		 unsigned short _canny_lt;
		 unsigned short _canny_ht;
		 unsigned short _houghp_th;
		 unsigned short _houghp_min_line_len;
		 unsigned short _houghp_max_line_gap;
		 float _maxslope_in_pixel;
		 float _minslope_in_pixel;
		 int _min_numof_lines_4_cluster;
		 int _min_numof_lines_4_cluster_rgb_only;
		 int _predefined_roi_height;
		 int _predefined_roi_width;
		 float _rule_base_grad_max;
         float _rule_base_grad_min;
		 float _rule_base_grad_diff_th_y3;
		 float _rule_base_grad_diff_th_y2;
		 float _rule_base_grad_diff_th_y1;
		 float _rule_base_grad_diff_th_x3;
		 float _rule_base_grad_diff_th_x2;
		 float _rule_base_grad_diff_th_x1;
         float _rule_base_depth_y_error_th_y3;
		 float _rule_base_depth_y_error_th_y2;
		 float _rule_base_depth_y_error_th_y1;
		 float _rule_base_depth_y_error_th_x3;
		 float _rule_base_depth_y_error_th_x2;
		 float _rule_base_depth_y_error_th_x1;
		 float _rule_base_depth_y_error_th_min;
         float _rule_base_x_avg_error_th;		 
		 int _noise_rm_pre_proc_index;
		 bool _implementation_on_jetson;
		
		 std::vector<float> x_coor_cam_frame_vec;
		 std::vector<float> y_coor_cam_frame_vec;
		 std::vector<float> z_coor_cam_frame_vec;
		 std::vector<float> slope_n_pixel;
		 
		 std::vector<int> row_center_vec;
		 std::vector<int> column_center_vec;
		 std::vector<float> depth_info_vec;
		 
		 std::vector<float> debug_gradient;
		 std::vector<float> debug_gradient_diff;
		 std::vector<float> debug_depth_y_error;
		 
		 stair_custom_msg::stairlocation _stair_pose;
		 stair_custom_msg::pixel_stairlocation _stair_center;
		 stair_custom_msg::debug_msg_1 _debug_msg1;
		 
		 std::string  _camera_frame_id;
		 
		 cv::Ptr<cv::ml::SVM> classifier;
		 
		  /*offline test variables*/
		 std::string  _testing_set_dir;
		 std::string  _testing_set_rgb_image_list_file;
		 std::string  _testing_set_depth_image_list_file;
		 
		 std:: ifstream _rgb_in;
		 std:: ifstream _depth_in;
		 
		 std::string _rgb_image_file_full_path;
		 std::string _depth_image_file_full_path;
		 
		 std::string _rgb_image_file;
		 std::string _depth_image_file;
		 
		 bool _image_open_ok = false; 		  
		 unsigned char _reading_status = 0;   /*0 : reading bad, 1 : reading (process), 2 : reading (success)*/
		 
		 std::string _result_save_dir;
		 int image_read_count = 0;
		 
		 bool _stair_case_detc_flag = false;
	
	public:
	
	    void run();
		void pre_proc_run();
		void publish_msg();
		void stair_case_detc_rule_base(const cv::Mat& rgb_input, std::vector<std::vector<float>>* vector_set_for_learning);
		void stair_case_detc_svm(const cv::Mat& rgb_input, std::vector<std::vector<float>>* vector_set_for_learning);
		void stair_case_detc_rgb_only(const cv::Mat& rgb_input, int final_center_point_col,int final_center_point_row);
		void rgb_image_callback(const sensor_msgs::Image::ConstPtr& msg, int value);
		void depth_image_callback(const sensor_msgs::Image::ConstPtr& msg, int value);
		void rgb_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg, int value);
		void depth_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg, int value);
		
		void executeCB_MBflag(const std_msgs::Int8::ConstPtr& msg,int value);
		void executeCB_detect_trigger_flag(const std_msgs::Int8::ConstPtr& msg,int value);
		
		/*offline test function*/
		void read_rgb_depth_file_list();
	 
		/*constructor and destructor*/
	    STAIR_DETECTION_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh);
	    ~STAIR_DETECTION_ROS();
	
	    STAIR_DETEC_PRE_PROC* str_det_prprc;
		 STAIR_DETEC_COST_FUNC* str_det_cost_func;
		 STAIR_DETEC_ONLY_RGB* str_det_only_rgb;
};

STAIR_DETECTION_ROS::STAIR_DETECTION_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh):main_nh(m_nh),param_nh(p_nh),_it(m_nh),tfListener(tfBuffer)
{
	int update_rate = 10;
	
	 std::string  rgb_image_topic = "front_cam/camera/color/image_raw";
	 std::string  depth_image_topic = "front_cam/camera/depth/image_rect_raw";
	 std::string camera_rgb_info_topic = "front_cam/camera/color/camera_info";
	 std::string camera_depth_info_topic = "front_cam/camera/depth/camera_info";
	 std::string camera_stair_pose_topic = "front_cam/camera/stair_pose";
	 std::string camera_stair_center_pixel_topic = "front_cam/camera/center_pixel";
	 std::string MB_flag_sub_topic= "MB_flag";
	 std::string detect_trigger_flag_sub_topic ="detec_logic_trigger";
	 
	
	 param_nh.getParam("depth_image_topic",depth_image_topic);
	 param_nh.getParam("rgb_image_topic",rgb_image_topic);
	 param_nh.getParam("camera_depth_info_topic",camera_depth_info_topic);
	 param_nh.getParam("camera_rgb_info_topic",camera_rgb_info_topic);
	 param_nh.getParam("camera_stair_pose_topic",camera_stair_pose_topic);
	 param_nh.getParam("camera_stair_center_pixel_topic",camera_stair_center_pixel_topic);
	 param_nh.getParam("MB_flag_sub_topic",MB_flag_sub_topic);
	 param_nh.getParam("detect_trigger_flag_sub_topic",detect_trigger_flag_sub_topic);
	 
	 
	 bool encoding_rgb_flag = false;
	 bool offline_svm_training = false;
	 bool use_svm_classifier = false;
	 bool offline_performance_test = false;  /*using image data set to test the logic */
	 bool use_only_rgb_to_detect_staircase = false;  /*detecting staircase only using the pixel coordinate, not using the depth value */
	 bool implementation_on_jetson = false;  /*detecting staircase only using the pixel coordinate, not using the depth value */
	 int preproc_resize_height = 480;
	 int preproc_resize_width = 848;
	 int canny_lt =25;
	 int canny_ht=40;
	 int houghp_th = 75;
	 int houghp_min_line_len = 40;
	 int houghp_max_line_gap = 5;
	 float maxslope_in_pixel = 0.5;
	 float minslope_in_pixel=0.05;
	 int min_numof_lines_4_cluster = 4;
	 int min_numof_lines_4_cluster_rgb_only = 6;
	 int predefined_roi_height=200;
	 int predefined_roi_width=200;
	 float rule_base_grad_max=1.85;
     float rule_base_grad_min=0.3;
	 float rule_base_grad_diff_th_y3 = 0.5;
	 float rule_base_grad_diff_th_y2 = 0.4;
	 float rule_base_grad_diff_th_y1 = 0.15;
	 float rule_base_grad_diff_th_x3 = 1.7;
	 float rule_base_grad_diff_th_x2 = 0.9;
	 float rule_base_grad_diff_th_x1 = 0.3;
     float rule_base_depth_y_error_th_y3=4;
	 float rule_base_depth_y_error_th_y2=3.5;
	 float rule_base_depth_y_error_th_y1=1.5;
	 float rule_base_depth_y_error_th_x3=1.7;
	 float rule_base_depth_y_error_th_x2=0.9;
	 float rule_base_depth_y_error_th_x1=0.3;
     float rule_base_x_avg_error_th=2.7;
	 float rule_base_depth_y_error_th_min=0.2;
	 int noise_rm_pre_proc_index = 1;   /* 0 : no dilate, erode 1: diate 2 : erode*/
	 
	 std::string camera_frame_id = "camera_aligned_depth_to_color_frame";
	 
	 std::string testing_set_dir = "/home/kangneoung/stair_detection/src/stair_detection/image_set/testing/true/long_stair";
	 std::string result_save_dir = "/home/kangneoung/stair_detection/src/stair_detection/image_set/testing/true/long_stair_result";
	 std::string testing_set_rgb_image_list_file = "test_rgb_img_file_list.txt";
	 std::string testing_set_depth_image_list_file = "test_depth_img_file_list.txt";
	  
	 param_nh.getParam("update_rate",update_rate);
	 param_nh.getParam("encoding_rgb_flag",encoding_rgb_flag);
	 param_nh.getParam("offline_svm_training",offline_svm_training); /*generate svm training set using real time image */
	 param_nh.getParam("use_svm_classifier",use_svm_classifier);
	 param_nh.getParam("use_only_rgb_to_detect_staircase",use_only_rgb_to_detect_staircase);
	 param_nh.getParam("image_resize_height",preproc_resize_height);
	 param_nh.getParam("image_resize_width",preproc_resize_width);
	 param_nh.getParam("canny_filter_low_threshold",canny_lt);
	 param_nh.getParam("canny_filter_high_threshold",canny_ht);
	 param_nh.getParam("hough_transform_threshold",houghp_th);
	 param_nh.getParam("hough_transform_minumum_line_length",houghp_min_line_len);
	 param_nh.getParam("hough_transform_maximum_line_point_gap",houghp_max_line_gap);
	 param_nh.getParam("maximum_slope_threshold_for_line",maxslope_in_pixel);
	 param_nh.getParam("minimum_slope_threshold_for_line",minslope_in_pixel);
	 param_nh.getParam("minimum_number_of_lines_for_clustering",min_numof_lines_4_cluster);
	 param_nh.getParam("min_numof_lines_4_cluster_rgb_only",min_numof_lines_4_cluster_rgb_only);
	 param_nh.getParam("predefined_roi_height",predefined_roi_height);
	 param_nh.getParam("predefined_roi_width",predefined_roi_width);
	 param_nh.getParam("stair_gradient_maximum_threshold",rule_base_grad_max);
	 param_nh.getParam("stair_gradient_minimum_threshold",rule_base_grad_min);
	 param_nh.getParam("rule_base_grad_diff_th_y3",rule_base_grad_diff_th_y3);
	 param_nh.getParam("rule_base_grad_diff_th_y2",rule_base_grad_diff_th_y2);
	 param_nh.getParam("rule_base_grad_diff_th_y1",rule_base_grad_diff_th_y1);
	 param_nh.getParam("rule_base_grad_diff_th_x3",rule_base_grad_diff_th_x3);
	 param_nh.getParam("rule_base_grad_diff_th_x2",rule_base_grad_diff_th_x2);
	 param_nh.getParam("rule_base_grad_diff_th_x1",rule_base_grad_diff_th_x1);
	 param_nh.getParam("rule_base_depth_y_error_th_y3",rule_base_depth_y_error_th_y3);
	 param_nh.getParam("rule_base_depth_y_error_th_y2",rule_base_depth_y_error_th_y2);
	 param_nh.getParam("rule_base_depth_y_error_th_y1",rule_base_depth_y_error_th_y1);
	 param_nh.getParam("rule_base_depth_y_error_th_x3",rule_base_depth_y_error_th_x3);
	 param_nh.getParam("rule_base_depth_y_error_th_x2",rule_base_depth_y_error_th_x2);
	 param_nh.getParam("rule_base_depth_y_error_th_x1",rule_base_depth_y_error_th_x1);
	 param_nh.getParam("depth_and_y_leastsquare_error_min_th",rule_base_depth_y_error_th_min);
	 param_nh.getParam("x_average_error",rule_base_x_avg_error_th);
	 param_nh.getParam("camera_frame_id",camera_frame_id);
	 param_nh.getParam("noise_rm_pre_proc_index",noise_rm_pre_proc_index);
	 param_nh.getParam("implementation_on_jetson",implementation_on_jetson);
	 
	 /*offline image set test param*/
	 param_nh.getParam("offline_performance_test",offline_performance_test); /*using image data set to test the logic */
	 param_nh.getParam("testing_set_dir",testing_set_dir);
	 param_nh.getParam("result_save_dir",result_save_dir);
	 param_nh.getParam("testing_set_rgb_image_list_file",testing_set_rgb_image_list_file);
	 param_nh.getParam("testing_set_depth_image_list_file",testing_set_depth_image_list_file);
     
	 this->_update_rate = update_rate;
	 this->_encoding_rgb_flag = encoding_rgb_flag;
	 this->_offline_svm_training = offline_svm_training; /*generate svm training set using real time image */
	 this->_use_svm_classifier =  use_svm_classifier;
	 this->_use_only_rgb_to_detect_staircase =  use_only_rgb_to_detect_staircase;
	 this->_offline_performance_test = offline_performance_test; /*using image data set to test the logic */
	 this->_preproc_resize_height = preproc_resize_height;
	 this->_preproc_resize_width = preproc_resize_width;
	 this->_canny_lt = canny_lt;
	 this->_canny_ht = canny_ht;
	 this->_houghp_th = houghp_th;
	 this->_houghp_min_line_len = houghp_min_line_len;
	 this->_houghp_max_line_gap = houghp_max_line_gap;
	 this->_maxslope_in_pixel = maxslope_in_pixel;
	 this->_minslope_in_pixel = minslope_in_pixel;
	 this->_min_numof_lines_4_cluster = min_numof_lines_4_cluster;
	 this->_min_numof_lines_4_cluster_rgb_only = min_numof_lines_4_cluster_rgb_only;
	 this->_predefined_roi_height = predefined_roi_height;
	 this->_predefined_roi_width = predefined_roi_width;
	 this->_rule_base_grad_max = rule_base_grad_max;
	 this->_rule_base_grad_min = rule_base_grad_min;
	 this->_rule_base_grad_diff_th_y3=rule_base_grad_diff_th_y3;
	 this->_rule_base_grad_diff_th_y2=rule_base_grad_diff_th_y2;
	 this->_rule_base_grad_diff_th_y1=rule_base_grad_diff_th_y1;
	 this->_rule_base_grad_diff_th_x3=rule_base_grad_diff_th_x3;
	 this->_rule_base_grad_diff_th_x2=rule_base_grad_diff_th_x2;
	 this->_rule_base_grad_diff_th_x1=rule_base_grad_diff_th_x1;
	 this->_rule_base_depth_y_error_th_y3 = rule_base_depth_y_error_th_y3;
	 this->_rule_base_depth_y_error_th_y2 = rule_base_depth_y_error_th_y2;
	 this->_rule_base_depth_y_error_th_y1 = rule_base_depth_y_error_th_y1;
	 this->_rule_base_depth_y_error_th_x3 = rule_base_depth_y_error_th_x3;
	 this->_rule_base_depth_y_error_th_x2 = rule_base_depth_y_error_th_x2;
	 this->_rule_base_depth_y_error_th_x1 = rule_base_depth_y_error_th_x1;
	 this->_rule_base_depth_y_error_th_min = rule_base_depth_y_error_th_min;
	 this->_rule_base_x_avg_error_th = rule_base_x_avg_error_th;
	 this->_noise_rm_pre_proc_index = noise_rm_pre_proc_index;
	 this->_implementation_on_jetson = implementation_on_jetson;
	 
	 this->_camera_frame_id = camera_frame_id;
	 
	 std::cout<<"_offline_performance_test : "<<this->_offline_performance_test<<"\n"<<std::endl;
     std::cout<<"_use_only_rgb_to_detect_staircase : "<<this->_use_only_rgb_to_detect_staircase<<"\n"<<std::endl;
	 
	 if(this->_offline_performance_test == true)
	 {	
	     this->_result_save_dir = result_save_dir;
	     this->_testing_set_dir = testing_set_dir;
	     this->_testing_set_rgb_image_list_file = testing_set_rgb_image_list_file;
	     this->_testing_set_depth_image_list_file = testing_set_depth_image_list_file;
	   
	     std::string full_dir_rgb_list = this->_testing_set_dir +"/" +  this->_testing_set_rgb_image_list_file ;
	     std::string full_dir_depth_list = this->_testing_set_dir +"/" +  this->_testing_set_depth_image_list_file ;
         std::string full_dir_rgb_list_filtered = this->_testing_set_dir +"/" + "test_rgb_img_file_list_filtered.txt";
		 std::string full_dir_depth_list_filtered = this->_testing_set_dir +"/" + "test_depth_img_file_list_filtered.txt";
		 std::string full_dir_not_detected_list = this->_result_save_dir + "/" + "not_detected_image_list.txt";
		 
         try
	     {
	          _rgb_in.open(full_dir_rgb_list);
			  _depth_in.open(full_dir_depth_list);
			  //_rgb_text_filter_in.open(full_dir_rgb_list_filtered,std::ofstream::app); /*text filtering*/
			  //_depth_text_filter_in.open(full_dir_depth_list_filtered,std::ofstream::app); /*text filtering*/
			  _not_detected_list.open(full_dir_not_detected_list,std::ofstream::app);
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
		 
	 }
	 else
	 { 
	    this->_rgb_sub =  _it.subscribe(rgb_image_topic, 1,boost::bind(&STAIR_DETECTION_ROS::rgb_image_callback,this,_1,0));
	    this->_depth_sub =  _it.subscribe(depth_image_topic, 1, boost::bind(&STAIR_DETECTION_ROS::depth_image_callback,this,_1,0));
	    this->_rgb_roi_pub =  _it.advertise("camera/color/stair_case_roi", 1);
	 
	    this->_rgb_cam_info_sub = main_nh.subscribe<sensor_msgs::CameraInfo>(camera_rgb_info_topic, 1, boost::bind(&STAIR_DETECTION_ROS::rgb_cam_info_callback, this,_1,0));
	    this->_depth_cam_info_sub = main_nh.subscribe<sensor_msgs::CameraInfo>(camera_depth_info_topic, 1, boost::bind(&STAIR_DETECTION_ROS::depth_cam_info_callback, this,_1,0));
	 }
/** Initialize STAIR_DETEC_PRE_PROC */
       
	 this->_MB_flag_sub = main_nh.subscribe<std_msgs::Int8>(MB_flag_sub_topic, 1, boost::bind(&STAIR_DETECTION_ROS::executeCB_MBflag, this,_1,0));  
	 this->_detec_trigger_sub = main_nh.subscribe<std_msgs::Int8>(detect_trigger_flag_sub_topic, 1, boost::bind(&STAIR_DETECTION_ROS::executeCB_detect_trigger_flag, this,_1,0));  
	   
     this->_stair_pos_pub = main_nh.advertise<stair_custom_msg::stairlocation>(camera_stair_pose_topic, 2);
	 this->_stair_center_pixel_pub = main_nh.advertise<stair_custom_msg::pixel_stairlocation>(camera_stair_center_pixel_topic, 2);

     this->_debug_msg_1_pub = main_nh.advertise<stair_custom_msg::debug_msg_1>("/front_cam/stair_detection/debug_1", 2);

     this->str_det_prprc = new STAIR_DETEC_PRE_PROC();
	 this->str_det_cost_func = new STAIR_DETEC_COST_FUNC( this->_offline_svm_training);
     this->_loop_rate = new ros::Rate(update_rate);	
	 
	 if(this->_use_svm_classifier == true)
	 {
	 	 try
			{
		       classifier = cv::Algorithm::load<cv::ml::SVM>("/home/kangneoung/stair_detection/src/stair_detection/svm_train.xml");
			}
		catch(int e)
			{
				ROS_ERROR("check the directory for svm parameters ");
			}
	 }
}

STAIR_DETECTION_ROS::~STAIR_DETECTION_ROS()
{
	 //_rgb_text_filter_in.close(); /*text filtering*/
	 //_depth_text_filter_in.close(); /*text filtering*/
	 _not_detected_list.close();
	
    delete this->str_det_prprc;
	delete this->str_det_cost_func;
	delete this->_loop_rate;
}

void STAIR_DETECTION_ROS::read_rgb_depth_file_list()
{
    
	std::string rgb_s;
	std::string depth_s;
	
	 std::getline(_rgb_in,rgb_s);
	 std::getline(_depth_in,depth_s);

	 bool rgb_directory_check_flag = false;
	 bool depth_directory_check_flag = false;
	
	if((_rgb_in.bad())||(_depth_in.bad()))
	{
		this->_reading_status = 0;
	}
    else if((_rgb_in.eof())||(_depth_in.eof()))
	{
		this->_reading_status = 2;
	}
	else
	{
		this->_reading_status = 1;
	}
	std::cout<<"reading_status   :"<<_reading_status<<"\n"<<std::endl;

    std::istringstream rgb_ss(rgb_s);
	std::istringstream depth_ss(depth_s);
	
	std::string stringBuffer_rgb;
	std::string stringBuffer_depth;
	
     if(this->_reading_status==1)    /*ignore first line  of text file*/ 
	 {  
	 
	     std::getline(rgb_ss, stringBuffer_rgb);
		 std::getline(depth_ss, stringBuffer_depth);

		this->_rgb_image_file = stringBuffer_rgb;//"image.png"; //
		this->_depth_image_file = stringBuffer_depth;//"image_dummy_depth.png"; //
	    this->_rgb_image_file_full_path = this->_testing_set_dir + "/" + this->_rgb_image_file;
	    this->_depth_image_file_full_path = this->_testing_set_dir + "/" + this->_depth_image_file;
	 
	    std::cout<<" this->_rgb_image_file_full_path :"<<  this->_rgb_image_file_full_path<<"\n"<<std::endl;
	    std::cout<<" this->_depth_image_file_full_path :"<< this->_depth_image_file_full_path<<"\n"<<std::endl;
	 

	     rgb_directory_check_flag = IsPathExist(this->_rgb_image_file_full_path);
	     depth_directory_check_flag = IsPathExist(this->_depth_image_file_full_path);

	    //std::cout<<" rgb_directory_check_flag :"<< rgb_directory_check_flag<<"\n"<<std::endl;
	    //std::cout<<" depth_directory_check_flag :"<< depth_directory_check_flag<<"\n"<<std::endl;
		
	 }
	 

	 
	 this->_image_open_ok = false;
	 
	 if((rgb_directory_check_flag==true)&&(depth_directory_check_flag==true))
	 {
		 this->_image_open_ok = true;
		 
		 //_rgb_text_filter_in << this->_rgb_image_file <<std::endl;   /*text filtering*/
		 //_depth_text_filter_in << this->_depth_image_file <<std::endl; /*text filtering*/
	 }
	 
	  //std::cout<<" this->_image_open_ok :"<< this->_image_open_ok<<"\n"<<std::endl;
	 
	 if(this->_reading_status==1)
	 {
	    string_read_count++;
	 }
	
	   // std::cout<<"string_read_count :"<<string_read_count<<"\n"<<std::endl;
	
	 if(this->_reading_status!=1)
	 {
		 this->_image_open_ok = false;
	 }	
}


void STAIR_DETECTION_ROS::executeCB_detect_trigger_flag(const std_msgs::Int8::ConstPtr& msg,int value)
{
	if(msg->data == 1)
	{
		this->_detec_trigger_flag = 1;
	}
	else
	{
		this->_detec_trigger_flag = 0;
		this->_MB_flag_status = 0;
	}
	
}


void STAIR_DETECTION_ROS::executeCB_MBflag(const std_msgs::Int8::ConstPtr& msg,int value)
{
	if(msg->data == 2)
	{
	   this->_MB_flag_status=msg->data;
	}
}



void STAIR_DETECTION_ROS::rgb_image_callback(const sensor_msgs::Image::ConstPtr& msg , int value)
{
	std::lock_guard<std::mutex> lock(_lock);
	
	cv::Mat rgb_image;
	rgb_image = cv_bridge::toCvShare(msg, "bgr8")->image;
	this->_rgb_image = rgb_image.clone();
	
	if(value > 0)
	{
		ROS_INFO("rgb_image_callback");
	}
	
}



void STAIR_DETECTION_ROS::depth_image_callback(const sensor_msgs::Image::ConstPtr& msg,int value)
{
	std::lock_guard<std::mutex> lock(_lock);
	cv::Mat depth_image = cv_bridge::toCvCopy(msg)->image;
	this->_depth_image = depth_image.clone();
	
	if(value > 0)
	{
		ROS_INFO("depth_image_callback");
	}
}

void STAIR_DETECTION_ROS::rgb_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg ,int value)
{
          /*empty*/
		  
    if(value > 0)
	{
		ROS_INFO("rgb_cam_info_callback");
	}
}

void STAIR_DETECTION_ROS::depth_cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg ,int value)
{        
          this->_fx = msg->K[0];
          this->_fy = msg->K[4];
          this->_px = msg->K[2];
          this->_py = msg->K[5];
	      this->_dscale = 0.001;//this->_dscale = msg->D[0];
		  
	if(value > 0)
	{
		ROS_INFO("depth_cam_info_callback");
	}
}

void STAIR_DETECTION_ROS::publish_msg()
{
		 
	 this->_stair_pose.header.stamp = ros::Time::now();
	 this->_stair_pose.header.frame_id = this->_camera_frame_id;
	 
	 this->_stair_center.header.stamp = ros::Time::now();
	 this->_stair_center.header.frame_id = this->_camera_frame_id;
	 
	 this->_debug_msg1.header.stamp = ros::Time::now();
	 this->_debug_msg1.header.frame_id = this->_camera_frame_id;
	 
	 if(this->_stair_case_detc_flag == true)
	 {  
		 //this->x_coor_cam_frame_vec.push_back(1);
		 //this->y_coor_cam_frame_vec.push_back(-0.5);
		 //this->z_coor_cam_frame_vec.push_back(4);
		 //this->slope_n_pixel.push_back(0);
		 
		 //this->x_coor_cam_frame_vec.push_back(-1);
		 //this->y_coor_cam_frame_vec.push_back(0.5);
		 //this->z_coor_cam_frame_vec.push_back(6);
		 //this->slope_n_pixel.push_back(0);
		 
         this->_stair_pose.x = this->x_coor_cam_frame_vec;
	     this->_stair_pose.y = this->y_coor_cam_frame_vec;
	     this->_stair_pose.z = this->z_coor_cam_frame_vec;
		 this->_stair_pose.slope_n = this->slope_n_pixel;
		 
		 this->_stair_pos_pub.publish(this->_stair_pose);
		 
		 this->_stair_center.row_center = this->row_center_vec;
		 this->_stair_center.column_center = this->column_center_vec;
		 this->_stair_center.depth_vec = this->depth_info_vec;
		 
		 this->_stair_center_pixel_pub.publish(this->_stair_center);
		  

		 this->_debug_msg1.gradient_vec = this->debug_gradient;
		 this->_debug_msg1.gradient_diff_vec = this->debug_gradient_diff;
		 this->_debug_msg1.depth_y_error_vec = this->debug_depth_y_error;
		 
		 this->_debug_msg_1_pub.publish(this->_debug_msg1);
		 
	 }	 
	 
	 this->x_coor_cam_frame_vec.clear();
	 this->y_coor_cam_frame_vec.clear();
	 this->z_coor_cam_frame_vec.clear();
	 this->slope_n_pixel.clear();
	 
	 this->row_center_vec.clear();
	 this->column_center_vec.clear();
	 this->depth_info_vec.clear();
	 
	  this->debug_gradient.clear();
	  this->debug_gradient_diff.clear();
	  this->debug_depth_y_error.clear();
	
}

void STAIR_DETECTION_ROS::pre_proc_run()
{     
       std::vector<cv::Vec4i>* edge_lines_main = new std::vector<cv::Vec4i>();
	   std::vector<std::vector<float>>*  vector_set_for_learning_main = new  std::vector<std::vector<float>>();
	   std::vector<cv::Vec4i>* final_edge_lines_main = new  std::vector<cv::Vec4i>;
	   std::map<float, std::vector<cv::Vec4i>>*  lines_hist_main = new std::map<float, std::vector<cv::Vec4i>>;
	   std::map<int, std::vector<cv::Vec4i>>* lines_hist_top_three_main = new std::map<int, std::vector<cv::Vec4i>>;
	   cv::Mat resized_gray_image_main; 
       cv::Mat resized_rgb_image_main; 	   
	   
	   int while_count = 0;
	   int while_count_th =1;
	 
     //std::cout<<"_offline_performance_test : "<<this->_offline_performance_test<<"\n"<<std::endl;	 
	 //std::cout<<"_use_only_rgb_to_detect_staircase : "<<this->_use_only_rgb_to_detect_staircase<<"\n"<<std::endl;
	   
	 if(this->_offline_performance_test == true)
	 { 
		if(this->_reading_status<2)
	    {
            this->read_rgb_depth_file_list();
	    }
		   
		if(this->_image_open_ok == true)
		{
			img_ok_count = img_ok_count +1;
			this->_rgb_image = cv::imread(this->_rgb_image_file_full_path,cv::IMREAD_COLOR);
			this->_depth_image = cv::imread(this->_depth_image_file_full_path,cv::IMREAD_ANYDEPTH);
		}	
	 }
	 
	//std::cout<<"this->_image_open_ok  : "<<this->_image_open_ok <<"\n"<<std::endl;	 
	
	 if(this->_image_open_ok == false)
	 {
		 while_count_th =0; /*no while loop*/
	 }
	 
	 if(this->_offline_performance_test == false)
	 {
		 while_count_th =1;   /*one loop*/
	 }
	 
	 this->_stair_case_detc_flag = false;
	
	 while((while_count<while_count_th)&&(this->_stair_case_detc_flag==false))
	 {   
         if(this->_stair_case_detc_flag)
		 {
			 break;
		 }
	     if(!this->_rgb_image.empty())
	     {
	        this->str_det_prprc->stair_line_extraction(this->_rgb_image, edge_lines_main,&resized_rgb_image_main,&resized_gray_image_main,this->_encoding_rgb_flag,this->_preproc_resize_height,this->_preproc_resize_width,
	        this->_canny_lt,this->_canny_ht,this->_houghp_th,this->_houghp_min_line_len,this->_houghp_max_line_gap,this->_noise_rm_pre_proc_index);
	     }
	   
	     if(!resized_gray_image_main.empty())
         {
	        this->str_det_prprc->select_lines_from_hough(resized_rgb_image_main,resized_gray_image_main,edge_lines_main,final_edge_lines_main,lines_hist_main,lines_hist_top_three_main,this->_maxslope_in_pixel,
		    this->_minslope_in_pixel);	
	      }
	
	     cv::Mat roi_center_point_out_main; 
	     cv::Mat midp_of_all_lines_main; 
	
	     if(!resized_gray_image_main.empty())
         {
	         this->str_det_prprc->grouping_lines_and_define_centers(resized_rgb_image_main, lines_hist_top_three_main, &roi_center_point_out_main, &midp_of_all_lines_main, this->_min_numof_lines_4_cluster,   this->_predefined_roi_height,   this->_predefined_roi_width);
	     }
		 
		 
	     if(this->_use_only_rgb_to_detect_staircase == true)
		 {
			  if(!resized_gray_image_main.empty())
			  {
				  int final_center_point_col_temp;
				  int final_center_point_row_temp;
				  float depth_center;
				  
				  this->_stair_case_detc_flag= this->str_det_only_rgb->stair_case_detec_only_rgb(resized_rgb_image_main,roi_center_point_out_main,midp_of_all_lines_main,&final_center_point_col_temp,&final_center_point_row_temp,this->_min_numof_lines_4_cluster_rgb_only, this->_predefined_roi_height,  this->_predefined_roi_width, this->_preproc_resize_height,this->_preproc_resize_width);
				  this->stair_case_detc_rgb_only(resized_rgb_image_main,final_center_point_col_temp,final_center_point_row_temp);
				  
				  if(!_depth_image.empty())
				  {
					  cv::Mat resized_depth_img_tmp, resized_depth_img_tmp_32f; 
	                  cv::resize(_depth_image, resized_depth_img_tmp, cv::Size(this->_preproc_resize_width,this->_preproc_resize_height), cv::INTER_AREA);  /*resize image to reduce computation cost */
					  if(resized_depth_img_tmp.type() == CV_16UC1) resized_depth_img_tmp.convertTo(resized_depth_img_tmp_32f, CV_32F, this->_dscale);
					  
					  depth_center =  resized_depth_img_tmp_32f.at<float>(final_center_point_row_temp,final_center_point_col_temp);
					  
					  if(depth_center<0.1)
					  {
						  depth_center = 5;
					  }
				  } 
				  else
				  {
					  depth_center = 5;
				  }
				  
				  if(this->_stair_case_detc_flag == true)
				  {
					  this->row_center_vec.push_back(final_center_point_row_temp);
			          this->column_center_vec.push_back(final_center_point_col_temp);
					  this->depth_info_vec.push_back(depth_center);
				  }
			  }
		 }
		 else
		 {
	        if(!_depth_image.empty())
	        {
		           this->str_det_cost_func->cal_cost_wrapper(resized_rgb_image_main, this->_depth_image,roi_center_point_out_main,midp_of_all_lines_main,vector_set_for_learning_main,this->_fx,this->_fy,this->_px,this->_py,this->_dscale,this->_min_numof_lines_4_cluster, this->_predefined_roi_height,  this->_predefined_roi_width, this->_preproc_resize_height,this->_preproc_resize_width);
	        }
	
	        if(!resized_gray_image_main.empty())
            {
	           if(this->_use_svm_classifier == true)
	           {
		           this->stair_case_detc_svm(resized_rgb_image_main, vector_set_for_learning_main);	
	            }
	           else
	           {
                  this->stair_case_detc_rule_base(resized_rgb_image_main, vector_set_for_learning_main);		 
	           }  
	        }
		 }
		 
		 while_count = while_count +1;
		 
		 //std::cout<<"while_count : "<<while_count<<"\n"<<std::endl;
	 }
	 
	 if((this->_image_open_ok == true)&&(this->_stair_case_detc_flag == false))
	 {
		 _not_detected_list<<this->_rgb_image_file<<std::endl;
	 }
	 
	//std::cout<<"img_ok_count : "<<img_ok_count<<"\n"<<std::endl;
	 //std::cout<<"result_count : "<<result_count<<"\n"<<std::endl;
	 
     this->publish_msg();
	 
	 delete edge_lines_main;
	 delete final_edge_lines_main;
	 delete lines_hist_main;
	 delete lines_hist_top_three_main;
	 delete vector_set_for_learning_main;
}

void STAIR_DETECTION_ROS::stair_case_detc_rule_base(const cv::Mat& rgb_input, std::vector<std::vector<float>>* vector_set_for_learning)
{
	std::vector<std::vector<float>>::iterator it;
	
	float gradient;
	float gradient_diff;
	float depth_y_error;
	float x_avg_error;
	float x_coor_cam_frame_tmp;
	float y_coor_cam_frame_tmp;
	float z_coor_cam_frame_tmp;
	float x_center_pixel;
	float y_center_pixel;
	
	float gradient_diff_th_interp;
	float rule_base_depth_y_error_th_interp;
	
	bool gradient_condition_ok = false;
	bool depth_y_error_condition_ok = false;
	bool avg_x_error_condition_ok = false;
	bool stair_case_detc_flag = false;
	
    /*vector_for_learning include 8 elements*/
	/* stair gradient , least square error(x coordinate of map frame (depth) and z coordinate of map frame(rows)) per line,  average error (y coordinate of map frame (cols)), center point x (cam frame), center point y(cam frame), depth, center point x pixel, center point y pixel*/
	
	int i=0;
	
	std::cout<<"vector_set_for_learning_size : "<<vector_set_for_learning->size()<<"\n"<<std::endl;
	
	for(it=vector_set_for_learning->begin(); it!=vector_set_for_learning->end(); it++)
	{  
	   
		gradient = it->at(0);
		
		std::cout<<"gradient : "<<gradient<<"\n"<<std::endl;
		
		gradient_diff = it->at(1);
		
		std::cout<<"gradient_diff : "<<gradient_diff<<"\n"<<std::endl;
		
		depth_y_error = (it->at(2))*10;
		
		std::cout<<"depth_y_error : "<<depth_y_error<<"\n"<<std::endl;
		
		x_avg_error = it ->at(3);
		
		//std::cout<<"x_avg_error : "<<x_avg_error<<"\n"<<std::endl;
		if((gradient>3)||(std::abs(gradient_diff)>2)||(depth_y_error>10))
		{
			continue;
		}
		
		gradient_diff_th_interp=interpol3(gradient,this->_rule_base_grad_diff_th_x1,this->_rule_base_grad_diff_th_x2,this->_rule_base_grad_diff_th_x3,
		                                                                       this->_rule_base_grad_diff_th_y1,this->_rule_base_grad_diff_th_y2,this->_rule_base_grad_diff_th_y3);
		rule_base_depth_y_error_th_interp=interpol3(gradient, this->_rule_base_depth_y_error_th_x1, this->_rule_base_depth_y_error_th_x2, this->_rule_base_depth_y_error_th_x3, 
		                                                                                               this->_rule_base_depth_y_error_th_y1, this->_rule_base_depth_y_error_th_y2, this->_rule_base_depth_y_error_th_y3);
		
		std::cout<<"gradient : "<<gradient<<"\n"<<std::endl;
		std::cout<<"gradient_diff_th_interp : "<<gradient_diff_th_interp<<"\n"<<std::endl;
		std::cout<<"rule_base_depth_y_error_th_interp : "<<rule_base_depth_y_error_th_interp<<"\n"<<std::endl;
		
		
		if((gradient<this->_rule_base_grad_max)&&(gradient>this->_rule_base_grad_min)&&(gradient_diff<gradient_diff_th_interp))
		{
			gradient_condition_ok = true;
		}
		else
		{
			gradient_condition_ok = false;
		}
        
         if((depth_y_error<rule_base_depth_y_error_th_interp)&&(depth_y_error>_rule_base_depth_y_error_th_min))
		{
			depth_y_error_condition_ok = true;
		}
        else
		{
            depth_y_error_condition_ok = false;
		}			
		
		if(x_avg_error < this->_rule_base_x_avg_error_th)
		{
			avg_x_error_condition_ok = true;
		}
		else
		{
			avg_x_error_condition_ok = false;
		}
		
		if((gradient_condition_ok==true)&&(depth_y_error_condition_ok==true)&&(avg_x_error_condition_ok==true))
		{
			stair_case_detc_flag = true;
			this->_stair_case_detc_flag = stair_case_detc_flag;
			x_coor_cam_frame_tmp = it->at(4);
			y_coor_cam_frame_tmp = it->at(5);
			z_coor_cam_frame_tmp = it->at(6);
			x_center_pixel = it->at(7);
			y_center_pixel = it->at(8);
			
			this->x_coor_cam_frame_vec.push_back(x_coor_cam_frame_tmp);
			this->y_coor_cam_frame_vec.push_back(y_coor_cam_frame_tmp);
			this->z_coor_cam_frame_vec.push_back(z_coor_cam_frame_tmp);
			this->slope_n_pixel.push_back(0);
			
			this->row_center_vec.push_back(x_center_pixel);
			this->column_center_vec.push_back(y_center_pixel);
			
			 if(this->_use_only_rgb_to_detect_staircase == true)
			 {
				this->depth_info_vec.push_back(0);
			 }
			 else
             {
				this->depth_info_vec.push_back(z_coor_cam_frame_tmp);
			 }		 
			
			 this->debug_gradient.push_back(gradient);
			 this->debug_gradient_diff.push_back(gradient_diff);
			 this->debug_depth_y_error.push_back(depth_y_error);
			
			//break;
		}
		else
		{
			stair_case_detc_flag = false;
		}
		
	   if(stair_case_detc_flag==true)
	   {
			cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
			
			rectangle_tmp1_pt1.x =x_center_pixel-30;
		    rectangle_tmp1_pt1.y =y_center_pixel+30;
		
		    rectangle_tmp1_pt2.x =x_center_pixel+30;
		    rectangle_tmp1_pt2.y =y_center_pixel-30;
			
			 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);
			 
			 std::cout<<"y_coor_cam_frame_tmp : "<<y_coor_cam_frame_tmp<<"\n"<<std::endl;
			 std::cout<<"z_coor_cam_frame_tmp : "<<z_coor_cam_frame_tmp<<"\n"<<std::endl;
	   }

		i++;
	}
	
	//std::cout<<"gradient_condition_ok : "<<gradient_condition_ok<<"\n"<<std::endl;
	//std::cout<<"depth_y_error_condition_ok : "<<depth_y_error_condition_ok<<"\n"<<std::endl;
	//std::cout<<"avg_x_error_condition_ok : "<<avg_x_error_condition_ok<<"\n"<<std::endl;
	//std::cout<<"stair_case_detc_flag : "<<stair_case_detc_flag<<"\n"<<std::endl;
	
	//if(stair_case_detc_flag==true)
	//{
	//		cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
			
	//		rectangle_tmp1_pt1.x =x_center_pixel-30;
	//	    rectangle_tmp1_pt1.y =y_center_pixel+30;
		
	//	    rectangle_tmp1_pt2.x =x_center_pixel+30;
	//	    rectangle_tmp1_pt2.y =y_center_pixel-30;
			
	//		 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);
			 
	//		 std::cout<<"y_coor_cam_frame_tmp : "<<y_coor_cam_frame_tmp<<"\n"<<std::endl;
	//		 std::cout<<"z_coor_cam_frame_tmp : "<<z_coor_cam_frame_tmp<<"\n"<<std::endl;
	//}

	
	this->_stair_case_output_image=rgb_input.clone();
	
	 if(this->_implementation_on_jetson == false)
	 {
		  cv::imshow("stair case ROI", rgb_input);
	      cv::waitKey(5);
	 }
    
	std::string output_rgb_save_dir;
	std::string index = std::to_string(string_read_count);
	
	if(this->_offline_performance_test == true)
	{
		if(1/*this->_stair_case_detc_flag==true*/)
		{
			output_rgb_save_dir = this->_result_save_dir + "/" +  this->_rgb_image_file + "result"+".png";
			cv::imwrite( output_rgb_save_dir, this->_stair_case_output_image);
			
			if(this->_stair_case_detc_flag==true)
			{
			    result_count=result_count+1;
			}
		}
	}
}


void STAIR_DETECTION_ROS::stair_case_detc_svm(const cv::Mat& rgb_input, std::vector<std::vector<float>>* vector_set_for_learning)
{
	std::vector<std::vector<float>>::iterator it;
	
	float gradient;
	float gradient_diff;
	float depth_y_error;
	float x_avg_error;
	float x_coor_cam_frame_tmp;
	float y_coor_cam_frame_tmp;
	float z_coor_cam_frame_tmp;
	float x_center_pixel;
	float y_center_pixel;
	
	bool stair_case_detc_flag = false;
    /*vector_for_learning include 8 elements*/
	/* stair gradient , least square error(x coordinate of map frame (depth) and z coordinate of map frame(rows)) per line,  average error (y coordinate of map frame (cols)), center point x (cam frame), center point y(cam frame), depth, center point x pixel, center point y pixel*/
	
	int i=0;
	
     double t;
     if(time_debug_flag == true) t = (double)cv::getTickCount();
	
	for(it=vector_set_for_learning->begin(); it!=vector_set_for_learning->end(); it++)
	{  
	   
		gradient = it->at(0);
		
		//std::cout<<"gradient : "<<gradient<<"\n"<<std::endl;
		
		gradient_diff = it->at(1);
		
		//std::cout<<"gradient_diff : "<<gradient_diff<<"\n"<<std::endl;
		
		depth_y_error = (it->at(2))*10;
		
		//std::cout<<"depth_y_error : "<<depth_y_error<<"\n"<<std::endl;
		
		x_avg_error = it ->at(3);
		
		
		
		if((gradient>3)||(std::abs(gradient_diff)>2)||(depth_y_error>10))
		{
			continue;
		}

		//std::cout<<"x_avg_error : "<<x_avg_error<<"\n"<<std::endl;
	    
		float training_array_temp[1][3] = {gradient, gradient_diff, depth_y_error};  /* svm was trained with 10 times depth_y_error*/
		
		cv::Mat test_mat(1, 3, CV_32F,training_array_temp);
		
		float response = classifier->predict(test_mat);
		
		if(response==1)
		{
			stair_case_detc_flag = true;
			this->_stair_case_detc_flag = stair_case_detc_flag;
			x_coor_cam_frame_tmp = it->at(4);
			y_coor_cam_frame_tmp = it->at(5);
			z_coor_cam_frame_tmp = it->at(6);
			x_center_pixel = it->at(7);
			y_center_pixel = it->at(8);
			
			 std::cout<<"gradient : "<<gradient<<" gradient_diff : "<<gradient_diff<<" depth_y_error :"<<depth_y_error<<"\n"<<std::endl;
			 
			this->x_coor_cam_frame_vec.push_back(x_coor_cam_frame_tmp);
			this->y_coor_cam_frame_vec.push_back(y_coor_cam_frame_tmp);
			this->z_coor_cam_frame_vec.push_back(z_coor_cam_frame_tmp);
			this->slope_n_pixel.push_back(0);
			 
			this->row_center_vec.push_back(x_center_pixel);
			this->column_center_vec.push_back(y_center_pixel);
			
			 if(this->_use_only_rgb_to_detect_staircase == true)
			 {
				this->depth_info_vec.push_back(0);
			 }
			 else
             {
				 this->depth_info_vec.push_back(z_coor_cam_frame_tmp);
			 }		 
			 
			 
			 this->debug_gradient.push_back(gradient);
			 this->debug_gradient_diff.push_back(gradient_diff);
			 this->debug_depth_y_error.push_back(depth_y_error);
			 
			 
			//break;
		}
		
		
	   if(stair_case_detc_flag==true)
	   {
			cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
			
			rectangle_tmp1_pt1.x =x_center_pixel-30;
		    rectangle_tmp1_pt1.y =y_center_pixel+30;
		
		    rectangle_tmp1_pt2.x =x_center_pixel+30;
		    rectangle_tmp1_pt2.y =y_center_pixel-30;
			
			 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);
			 
			 std::cout<<"y_coor_cam_frame_tmp : "<<y_coor_cam_frame_tmp<<"\n"<<std::endl;
			 std::cout<<"z_coor_cam_frame_tmp : "<<z_coor_cam_frame_tmp<<"\n"<<std::endl;
	    }

		i++;
	}
	
	
	//if(stair_case_detc_flag==true)
	//{
	//		cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
			
	//		rectangle_tmp1_pt1.x =x_center_pixel-30;
	//	    rectangle_tmp1_pt1.y =y_center_pixel+30;
		
	//	    rectangle_tmp1_pt2.x =x_center_pixel+30;
	//	    rectangle_tmp1_pt2.y =y_center_pixel-30;
			
	//		 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);
			 
	//		 std::cout<<"y_coor_cam_frame_tmp : "<<y_coor_cam_frame_tmp<<"\n"<<std::endl;
	//		 std::cout<<"z_coor_cam_frame_tmp : "<<z_coor_cam_frame_tmp<<"\n"<<std::endl;
	//}

	
	this->_stair_case_output_image=rgb_input.clone();
	
	if(this->_implementation_on_jetson == false)
	{
         cv::imshow("stair case ROI", rgb_input);
	     cv::waitKey(5);		
	}

	
    std::string output_rgb_save_dir;
	std::string index = std::to_string(string_read_count);
	
	if(this->_offline_performance_test == true)
	{
		if(1/*this->_stair_case_detc_flag==true*/)
		{
			output_rgb_save_dir = this->_result_save_dir + "/" + this->_rgb_image_file + "result"+".png";
			cv::imwrite( output_rgb_save_dir, this->_stair_case_output_image);
			
			if(this->_stair_case_detc_flag==true)
			{
			    result_count=result_count+1;
			}
		}
	}
	
	
	
     
	if(time_debug_flag == true){
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        printf("[INFO] svm_classifier() ---- took %.4lf ms (%.2lf Hz)\r\n", t*1000.0, (1.0/t));
    }
	
}




void STAIR_DETECTION_ROS::stair_case_detc_rgb_only(const cv::Mat& rgb_input, int final_center_point_col,int final_center_point_row)
{

	float x_center_pixel;
	float y_center_pixel;
	
    /*vector_for_learning include 8 elements*/
	/* stair gradient , least square error(x coordinate of map frame (depth) and z coordinate of map frame(rows)) per line,  average error (y coordinate of map frame (cols)), center point x (cam frame), center point y(cam frame), depth, center point x pixel, center point y pixel*/
	
	int i=0;
	
	   if(this->_stair_case_detc_flag==true)
	   {
			cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
			
			rectangle_tmp1_pt1.x =final_center_point_col-30;
		    rectangle_tmp1_pt1.y =final_center_point_row+30;
		
		    rectangle_tmp1_pt2.x =final_center_point_col+30;
		    rectangle_tmp1_pt2.y =final_center_point_row-30;
			
			 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);
			
	    }
	//if(stair_case_detc_flag==true)
	//{
	//		cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
			
	//		rectangle_tmp1_pt1.x =x_center_pixel-30;
	//	    rectangle_tmp1_pt1.y =y_center_pixel+30;
		
	//	    rectangle_tmp1_pt2.x =x_center_pixel+30;
	//	    rectangle_tmp1_pt2.y =y_center_pixel-30;
			
	//		 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  150,   150,   30),2/*thickness*/);
			 
	//		 std::cout<<"y_coor_cam_frame_tmp : "<<y_coor_cam_frame_tmp<<"\n"<<std::endl;
	//		 std::cout<<"z_coor_cam_frame_tmp : "<<z_coor_cam_frame_tmp<<"\n"<<std::endl;
	//}

	
	this->_stair_case_output_image=rgb_input.clone();
	
	if(this->_implementation_on_jetson == false)
	{
	   cv::imshow("stair case ROI", rgb_input);
	   cv::waitKey(5);
	}
	
    std::string output_rgb_save_dir;
	std::string index = std::to_string(string_read_count);
	
	if(this->_offline_performance_test == true)
	{
		if(1/*this->_stair_case_detc_flag==true*/)
		{
			output_rgb_save_dir = this->_result_save_dir + "/" + this->_rgb_image_file + "result"+".png";
			cv::imwrite( output_rgb_save_dir, this->_stair_case_output_image);
			
			if(this->_stair_case_detc_flag==true)
			{
			    result_count=result_count+1;
			}
		}
	}

	
}




void STAIR_DETECTION_ROS::run()
{  
    while(ros::ok())
	{
		if((this->_MB_flag_status !=2)&&(this->_detec_trigger_flag == 1))
		{  
		   this->pre_proc_run();
	   
	       sensor_msgs::ImagePtr rgb_roi_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->_stair_case_output_image).toImageMsg();
	   
	       this->_rgb_roi_pub.publish(rgb_roi_msg);
		}
	   //ros::spin();
	   ros::spinOnce();
	   this->_loop_rate->sleep();
	}
	
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "stair_detection");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");
  
  STAIR_DETECTION_ROS  stair_detect_ros(nh,_nh);
  
  stair_detect_ros.run();
 
  //image_transport::ImageTransport it(nh);
  //image_transport::Subscriber sub = it.subscribe("front_cam/camera/depth/image_rect_raw", 1, imageCallback);
  //ros::spin();
   return 0;
}
