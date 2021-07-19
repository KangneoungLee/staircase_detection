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

unsigned int string_read_count = 0;
bool svm_predict_test = true;

class SVM_TRAINING_ROS{

	private:
	 	 
		 ros::NodeHandle main_nh;
         ros::NodeHandle param_nh;
	     ros::Rate* _loop_rate;
		 
		 std:: ifstream in;
		 
		 std::vector<float> gradient_vector;
		 std::vector<float> gradient_diff_vector;
		 std::vector<float> avg_depth_y_error_vector;
		 std::vector<float> avg_x_error_vector;
		 std::vector<float> label_vector;
		 
		 cv::Ptr<cv::ml::SVM> svm;
		 cv::Ptr<cv::ml::SVM> classifier;
		 
		 float training_array[NUMBER_OF_ELEMENT];
		 
		 int label;
		 
		 bool train_ok =false;
		 unsigned char reading_status;   /*0 : reading bad, 1 : reading (process), 2 : reading (success)*/
		 
	public:
	    
		 void run();
		 void read_file_and_assign_array();
		 void svm_test_predit();
		 void svm_training();
		 void svm_save();
		 
		 /*constructor and destructor*/
	    SVM_TRAINING_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh);
	    ~SVM_TRAINING_ROS();

};

SVM_TRAINING_ROS::SVM_TRAINING_ROS(ros::NodeHandle m_nh, ros::NodeHandle p_nh):main_nh(m_nh),param_nh(p_nh)
{     
       try
	   {
	        in.open("/home/kangneoung/stair_detection/src/stair_detection/training_set_label.txt");
	    }
		catch(int e)
		{
			ROS_ERROR("check the directory for training data ");
		}
	   int update_rate = 10;
	   
	   this->_loop_rate = new ros::Rate(update_rate);
       
	   if(svm_predict_test==false)
	   {
          svm = cv::ml::SVM::create();	   
	      svm->setType(cv::ml::SVM::C_SVC);
	      //svm->setKernel(cv::ml::SVM::LINEAR);
		 // svm->setC(1);
		  svm->setKernel(cv::ml::SVM::RBF);
		  svm->setGamma(0.22192);//svm->setGamma(0.12192);//svm->setGamma(0.18192);  /*high gamma value means close to training data set*/
	      svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	   }
	   else
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

SVM_TRAINING_ROS::~SVM_TRAINING_ROS()
{   
    in.close();
	
	this->svm_save();
	
	delete this->_loop_rate;

}

void SVM_TRAINING_ROS::read_file_and_assign_array()
{
    
	std::string s;
	
	std::getline(in,s);
	
	if(in.bad())
	{
		reading_status = 0;
	}
    else if(in.eof())
	{
		reading_status = 2;
	}
	else
	{
		reading_status = 1;
	}
	std::cout<<"reading_status   :"<<reading_status<<"\n"<<std::endl;

    std::istringstream ss(s);
	
	std::string stringBuffer;
	
	std::vector<std::string> s_vector;

     if((string_read_count>0)&&(reading_status==1))    /*ignore first line  of text file*/ 
	 {  
	   while(std::getline(ss, stringBuffer, '/'))  /*split the string by '/' */
	   {
		   s_vector.push_back(stringBuffer);	
		   
		   std::cout<<stringBuffer<<std::endl;
	   }
	 }
	 
	 if(reading_status==1)
	 {
	    string_read_count++;
	 }
	 
	 std::cout<<"string_read_count :"<<string_read_count<<"\n"<<std::endl;
	 
	std::vector<std::string>::iterator it;
   
    int i = 0;
   
    for(it = s_vector.begin(); it!=s_vector.end(); ++it)
	{
		    float temp;
		   std::stringstream ssfloat(*it);   /*string to float*/
		   
		   ssfloat >> temp;
		   
		   if(i==1)
		   {
			   training_array[GRADIENT] = temp;
			   
			   std::cout<<"GRADIENT : "<< training_array[GRADIENT]<<std::endl;
			   
			   gradient_vector.push_back(temp);
		   }
		   
		   if(i==2)
		   {
			   training_array[GRADIENT_DIFF] = temp;
			   
			   std::cout<<"GRADIENT_DIFF : "<<training_array[GRADIENT_DIFF] <<std::endl;
			   
			   gradient_diff_vector.push_back(temp);
		   }
		   
		   if(i==3)
		   {
			   training_array[AVG_DEPTH_Y_ERROR] = temp;
			   
			   std::cout<<"AVG_DEPTH_Y_ERROR : "<<training_array[AVG_DEPTH_Y_ERROR]<<std::endl;
			   
			   avg_depth_y_error_vector.push_back(temp);
		   }
		   
		   if(i==7)
		   {
			   label = (int)temp;
			   
			   std::cout<<"label : "<<label<<std::endl;
			   
			   label_vector.push_back(temp);
		   }
		   
		   
		   //std::cout<<"iterator : "<<i<<"  "<<ssfloat.str()<<std::endl;
		   
		   i++;
	}
	
	
}

void SVM_TRAINING_ROS::svm_training()
{
	    //float training_array_temp[2][3] = {{1, 2, 3},{2,4, 9}};
		//float label_temp[2] ={1, -1};
		
		std::vector<float>::iterator it;
		
		
	    cv::Mat trainingDataMat(string_read_count-1, NUMBER_OF_ELEMENT, CV_32F);
		cv::Mat labelsMat(string_read_count-1, 1, CV_32SC1);
		
		int i = 0;
		
		for(it = gradient_vector.begin(); it != gradient_vector.end(); it++)
        {
			trainingDataMat.at<float>(i,0) = gradient_vector[i];
			trainingDataMat.at<float>(i,1) = gradient_diff_vector[i];
			trainingDataMat.at<float>(i,2) = avg_depth_y_error_vector[i];
			
			labelsMat.at<int>(i) = label_vector[i];
			
			if(label_vector[i]==0)
			{
				std::cout<<"invalid line \n"<<std::endl;
				break;
			}
			
			i++;
			
			
		}
          std::cout<<"debug line \n"<<std::endl;
	    train_ok = svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
		
        std::cout<<"string_read_count :"<<string_read_count<<"\n"<<std::endl;
		
		 std::cout<<"gradient_vector.size :"<<gradient_vector.size()<<"\n"<<std::endl;
		
}


void SVM_TRAINING_ROS::svm_test_predit()
{	   
        //float training_array_temp[1][3] = {2.040191,0.9935576, 0.27187};
		float training_array_temp[1][3] = {num1, num2, num3};
		
		printf("%f, %f, %f\n", num1, num2, num3);
		
		cv::Mat test_mat(1, 3, CV_32F,training_array_temp);
		
        float response = classifier->predict(test_mat);
		
		std::cout<<"response :"<<response<<"\n"<<std::endl;
}


void SVM_TRAINING_ROS::svm_save()
{	   
         if(svm_predict_test==false)
		 {
     	     svm->save("/home/kangneoung/stair_detection/src/stair_detection/svm_train.xml");
		 }
}




void SVM_TRAINING_ROS::run()
{
	while(ros::ok())
	{ 
	    if(svm_predict_test==false)  
		{  
		   if(train_ok==false)
	       {
               this->read_file_and_assign_array();
	       }
	   
	       if((reading_status==2)&&(string_read_count>0)&&(train_ok==false))   /*ignore first line and  stop the svm_training function if the line meets the end of txt*/
	       {  
	            std::cout<<"enter the training module \n"<<std::endl;
		        this->svm_training();
	       }
		}
		else
		{
			  this->svm_test_predit();
		}
		
	   ros::spinOnce();
	   this->_loop_rate->sleep();
	}
	
	
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "svm_training");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");
  
  if(svm_predict_test==true)
  {	  
     num1 = atof(argv[1]);
     num2 = atof(argv[2]);
     num3 = atof(argv[3]);
  }
  
  SVM_TRAINING_ROS  svm_training_ros(nh,_nh);
  
  svm_training_ros.run();
 
  //image_transport::ImageTransport it(nh);
  //image_transport::Subscriber sub = it.subscribe("front_cam/camera/depth/image_rect_raw", 1, imageCallback);
  //ros::spin();
   return 0;
}