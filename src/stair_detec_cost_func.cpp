#include "stair_detection/stair_detec_cost_func.h"

int function_call_count = 0;

/*constructor and destructor*/
STAIR_DETEC_COST_FUNC::STAIR_DETEC_COST_FUNC(bool svm_oft_flag, bool coor_trans_flag):_svm_offline_training_flag(svm_oft_flag), _coor_trans_flag(coor_trans_flag)
{
	if(this->_svm_offline_training_flag == true)
	{
		in.open("training_set.txt");
		in<<"function_call_count /"<<"gradient, unit:m/m  /"<<"continuity_factor_out, unit:m /"<<"deviation_cost, unit:m /"<<"x_avg_error, unit:m/"<<"center_point_x_pixel, unit: pixel/"<<"center_point_y_pixel, unit :pixel"<<std::endl ;
	}
	
	this->_small_roi_depth_th1 = 6;
	this->_small_roi_depth_th2 = 3;
	this->_small_roi_y_th_low = 1.5;
	this->_small_roi_y_th_high = 3;
	this->_invaild_depth_th = 9;
	this->_invail_depth_count_th =1000;
	
	
}

STAIR_DETEC_COST_FUNC::~STAIR_DETEC_COST_FUNC()
{
	if(this->_svm_offline_training_flag == true)
	{
		in.close();
	}
	
}

void STAIR_DETEC_COST_FUNC::cal_cost_wrapper_offline_image_true(cv::Mat& depth_img, int x_pixel, int y_start_pixel, int y_end_pixel, float fx, float fy, float px, float py, float dscale, unsigned short preproc_resize_height, unsigned short preproc_resize_width)
{
     int label = 1;
	 float invaild_depth_th = 9;
	 int invail_depth_count_th =1000;
	 
     cv::Mat resized_depth_img_tmp, resized_depth_img_tmp_32f; 
	 std::cout<<"cal_cost_offline_image_true 1 "<<" \n"<<std::endl;
	 cv::resize(depth_img, resized_depth_img_tmp, cv::Size(preproc_resize_width,preproc_resize_height), cv::INTER_AREA);  /*resize image to reduce computation cost */
	 
	 if(resized_depth_img_tmp.type() == CV_16UC1)
	 {		 
         resized_depth_img_tmp.convertTo(resized_depth_img_tmp_32f, CV_32F, dscale);
	 }
	 else
	 {
	    resized_depth_img_tmp_32f = resized_depth_img_tmp.clone();
	 }
	 
	 std::cout<<"cal_cost_offline_image_true 2 "<<" \n"<<std::endl;
	 
	 int center_point_col = std::round(x_pixel);
	 int center_point_row = std::round((y_start_pixel+y_end_pixel)/2);
	 
	 float depth_center =  resized_depth_img_tmp_32f.at<float>(center_point_row,center_point_col);
	 
	 float center_point_x_coor;     /*xyz coordinate*/
     float center_point_y_coor;    /*xyz coordinate*/
	 
	 if(depth_center>0.1)
	   {
		   center_point_x_coor = ((float)center_point_col - px) * depth_center / fx;
		   center_point_y_coor = ((float)center_point_row - py) * depth_center / fy;
	    }
	 else
	    {
		   center_point_x_coor = 0;
		   center_point_y_coor = 0;
	    }
	
	 int interval = 2;
	 
	 cv::Mat array_x_coor = cv::Mat::zeros(100, 1, CV_32F);    /* this array is for x (camera frame) */
	 cv::Mat array_depth_and_y_coor = cv::Mat::zeros(100, 2, CV_32F);        /* this array is for depth (z of camera frame) and y (camera frame)*/
		
	 int n = 0;
	 
	 bool  invalid_depth_flag = false;
	 int invalid_depth_count = 0;
	 int starting_point_y;
	  
	 for (starting_point_y=y_start_pixel; starting_point_y<y_end_pixel;starting_point_y=starting_point_y+interval)
	 {
		 //std::cout<<"img width : "<<resized_depth_img_tmp_32f.cols<<"img height : "<<resized_depth_img_tmp_32f.rows<<" \n"<<std::endl;		
		// std::cout<<"cal_cost_offline_image_true : starting_point_y : "<<starting_point_y<<"center_point_col : "<<center_point_col<<" \n"<<std::endl;				    
		  float depth_tmp2 =  resized_depth_img_tmp_32f.at<float>(starting_point_y,center_point_col);
		  
		  if(depth_tmp2>invaild_depth_th)
		 {
			 invalid_depth_count = invalid_depth_count +1;
		 }
					   
		 if(invalid_depth_count>=invail_depth_count_th)
		 {
			 invalid_depth_flag= true;
			 continue;
		  }
					   
		 if(depth_tmp2>0.1)
		 {
			 float y_coor_tmp = (py-(float)starting_point_y) * depth_tmp2 / fy;   /*there is a reason why py - starting_point_y  (not starting_point_y -py) */
					 
			 //std::cout<<"depth_tmp2 : "<< depth_tmp2<<" \n"<<std::endl;
			 //std::cout<<"y_coor_tmp : "<< y_coor_tmp<<" \n"<<std::endl;
			 //std::cout<<"n : "<< n<<" \n"<<std::endl;
					 
			 array_depth_and_y_coor.at<float>(n,0) =  depth_tmp2+30;   /* the reason why adding 30 is to avoid singular problem of opencv solve algorithm*/
			 array_depth_and_y_coor.at<float>(n,1) =  y_coor_tmp;   
					 
			 n++;
						   
		 }
		 
		 if (n>=99) break;
	 }
	 
	 cv::Mat  array_x_coor_final = array_x_coor.clone();
	 cv::Mat  array_depth_and_y_coor_final = array_depth_and_y_coor(cv::Range(0, n), cv::Range::all());
	  
	 float gradient_out;
	 float deviation_cost_out;
	 float x_avg_error_out;
	 float continuity_factor_out;
	 std::cout<<"debugn line1 "<<" \n"<<std::endl;
	 this->least_square_fit(array_x_coor_final, array_depth_and_y_coor_final, &gradient_out, &continuity_factor_out, &deviation_cost_out, &x_avg_error_out);
	 
	if(gradient_out>5)
	{
	    gradient_out =5;
	}
	else if(gradient_out < -5)
	{
		gradient_out =-5;
	}
						
	if(continuity_factor_out>5)
	{
		continuity_factor_out =5;
	}
	else if(continuity_factor_out < -5)
	{
		continuity_factor_out =-5;
	}
						
						
	if(deviation_cost_out>10)
	{
		deviation_cost_out =10;
	}
	 
	 std::cout<<"debugn line2 "<<" \n"<<std::endl;
	 in<<0<<"/"<<gradient_out<<"/"<<continuity_factor_out<<"/"<<deviation_cost_out<<"/"<<x_avg_error_out*5<<"/"<<center_point_col<<"/"<<center_point_row<<"/"<<label<<"/"<<depth_center<<std::endl;


}	


void STAIR_DETEC_COST_FUNC::cal_cost_wrapper(const cv::Mat& rgb_input, cv::Mat& depth_img, cv::Mat& roi_center_point_out, cv::Mat& midp_of_all_lines,std::vector <std::vector<float>>* vector_set_for_learning,float fx, float fy, float px, float py, float dscale,
                                                                                                int min_numof_lines_4_cluster,  int predefined_roi_height, int predefined_roi_width, unsigned short preproc_resize_height, unsigned short preproc_resize_width)
																								
{
   cv::Mat resized_depth_img_tmp, resized_depth_img_tmp_32f; 
	

   cv::resize(depth_img, resized_depth_img_tmp, cv::Size(preproc_resize_width,preproc_resize_height), cv::INTER_AREA);  /*resize image to reduce computation cost */
	
   if(resized_depth_img_tmp.type() == CV_16UC1)   resized_depth_img_tmp.convertTo(resized_depth_img_tmp_32f, CV_32F, dscale);
   else if(resized_depth_img_tmp.type() == CV_32F)   resized_depth_img_tmp_32f = resized_depth_img_tmp.clone();

   double t;
   if(time_debug_flag == true) t = (double)cv::getTickCount();
	
   int num_roi_size = 2;
   
   int height_denominator[2] ={2,4};
   int height_upper_offset[2]={0,0};
   
   int num_center_width_offset = 3;
   int num_center_height_offset = 3;
   
   int width_offset_ary[3]={0,25,-25};
   int height_offset_ary[3] = {0, -15, 15};

   
   bool y_coor_and_depth_fail_for_small_roi = false;
   
   int i;
   for( i = 0; i <  roi_center_point_out.rows; i++)
   { 
	  
	  cv::Mat array_x_coor = cv::Mat::zeros(100, 1, CV_32F);    /* this array is for x (camera frame) */
	 
	  /*if K_mean_cluster algortihm doesn't find suitable center position, it returns meaningless vlaue. Therefore, the invaild center position should be filtered out*/
	  if((roi_center_point_out.at<cv::Vec2f>(i)[0] <= 0)||(roi_center_point_out.at<cv::Vec2f>(i)[0] >= preproc_resize_width))   continue;
      else if(( roi_center_point_out.at<cv::Vec2f>(i)[1]<=0)||(roi_center_point_out.at<cv::Vec2f>(i)[1]>=preproc_resize_height))   continue;


	   float center_point_x  = roi_center_point_out.at<cv::Vec2f>(i)[0];   /*pixel coordinate*/
	   float center_point_y = roi_center_point_out.at<cv::Vec2f>(i)[1];   /*pixel coordinate*/
       //float center_point_x  = 168.375;
	   //float center_point_y  = 197;
		  
	   int center_point_col = std::round(center_point_x);
	   int center_point_row = std::round(center_point_y);
	   
	   float center_point_x_coor;     /*xyz coordinate*/
       float center_point_y_coor;    /*xyz coordinate*/
	   

	   float depth_center =  resized_depth_img_tmp_32f.at<float>(center_point_row,center_point_col);
	 //  float y_world_coor_center = (py - center_point_y)/fy*depth_center;  

	
	  
	  if(depth_center>0.1)
	  {
		   center_point_x_coor = ((float)center_point_col - px) * depth_center / fx;
		   center_point_y_coor = ((float)center_point_row - py) * depth_center / fy;
	   }
	   else
	   {
		   center_point_x_coor = 0;
		   center_point_y_coor = 0;
	   }
	  
	   if(0/*visualize_flag==2*/)  /*this code is for visualization */
	   {
		   cv::Point2i temp_point;
		   
		   temp_point.x = center_point_x;
		   temp_point.y = center_point_y;
		   
		   if((center_point_x>20)&&(center_point_x<(preproc_resize_width-20))&&(center_point_y>20)&&(center_point_y<(preproc_resize_height-20)))
		   {
			   cv::circle( rgb_input, temp_point, 10, cv::Scalar(  255,   0,   255), 2/*thickness*/ );   
		   }
	   }
	  
	  int midpoint_count = 0;
	  int roi_size_index; 
      for(roi_size_index=0; roi_size_index< num_roi_size;roi_size_index++)
      {
          /*exclude extreme geometry */
		  if((depth_center<this->_small_roi_depth_th2))
		  {
			  y_coor_and_depth_fail_for_small_roi = true;
		  }
		  else if((center_point_y_coor<-this->_small_roi_y_th_low)&&(depth_center<this->_small_roi_depth_th1))
		  {
			  y_coor_and_depth_fail_for_small_roi = true;
		  }
		  else if(center_point_y_coor<-this->_small_roi_y_th_high)
		  {
			 y_coor_and_depth_fail_for_small_roi = true;  
		  }
		  else
		  {
			  y_coor_and_depth_fail_for_small_roi = false;
		  }
		  
		  if((roi_size_index==1)&&(y_coor_and_depth_fail_for_small_roi==true))
		  {
			  continue;     /*roi_size_index == 1 means that the roi is small and if the object is too close, then skip the small roi*/
		  }
	   
	      float upper_limit = center_point_y + predefined_roi_height/height_denominator[roi_size_index] + height_upper_offset[roi_size_index];     /*define the rectangle roi boundary*/
	      float lower_limit = center_point_y  - predefined_roi_height/height_denominator[roi_size_index] + height_upper_offset[roi_size_index];     /*define the rectangle roi boundary*/
	      float left_limit = center_point_x - predefined_roi_width/2;            /*define the rectangle roi boundary*/
	      float right_limit = center_point_x + predefined_roi_width/2;        /*define the rectangle roi boundary*/		  
		  
		  /*upper_limit, lower_limit min_max range check to prevent cv error*/
		  if(upper_limit<15)   upper_limit = 15;
		  else if(upper_limit>(preproc_resize_height-15))   upper_limit =  preproc_resize_height-15;	
          
          if(lower_limit<15)   lower_limit = 15;
		  else if(lower_limit>(preproc_resize_height-15))   lower_limit =  preproc_resize_height-15;  
		  

	    
		  cv::Mat  array_x_of_mid_point_pixel = cv::Mat::zeros(100, 1, CV_32F);
		  cv::Mat array_y_of_mid_point_pixel = cv::Mat::zeros(100, 1, CV_32F);		
		  cv::Mat array_depth_and_y_coor = cv::Mat::zeros(100, 2, CV_32F);        /* this array is for depth (z of camera frame) and y (camera frame)*/
		  

		  if(roi_size_index==0)
		  {
			 int j;
		     for(j=0; j < midp_of_all_lines.rows; j++)
		     {
			
			     if((midp_of_all_lines.at<cv::Vec2f>(j)[0] > left_limit)&&(midp_of_all_lines.at<cv::Vec2f>(j)[0] < right_limit))
			     {
			        if((midp_of_all_lines.at<cv::Vec2f>(j)[1] > lower_limit)&&(midp_of_all_lines.at<cv::Vec2f>(j)[1] < upper_limit))
				     {
					     int column_p = std::round(midp_of_all_lines.at<cv::Vec2f>(j)[0]);
					     int row_p = std::round(midp_of_all_lines.at<cv::Vec2f>(j)[1]);
					 
					     //std::cout<<"column_p : "<< column_p<<" \n"<<std::endl;
					     //std::cout<<"row_p : "<< row_p<<" \n"<<std::endl;
					  
					     float depth_tmp =  resized_depth_img_tmp_32f.at<float>(row_p,column_p);
					 
					     if(depth_tmp>0.1)
					     {
					         float x_coor_tmp = ((float)column_p - px) * depth_tmp / fx;						
						   //std::cout<<"x_coor_tmp : "<< x_coor_tmp<<" \n"<<std::endl;
		
						     array_x_coor.at<float>(midpoint_count)= x_coor_tmp;
						
						     array_x_of_mid_point_pixel.at<float>(midpoint_count) =  midp_of_all_lines.at<cv::Vec2f>(j)[0];
					         array_y_of_mid_point_pixel.at<float>(midpoint_count) =  midp_of_all_lines.at<cv::Vec2f>(j)[1];
						  
						     midpoint_count++;
						   
					    }
				     } 
			      }
		       }
	        }

 
		   int roi_width_offset_index;
		   float center_point_x_w_offset;
           
		   int  num_center_width_offset_new;
		   
		   /* if roi size is large one*/
		   if(roi_size_index == 0)   num_center_width_offset_new =  1;
		   else   num_center_width_offset_new  = num_center_width_offset;
		   
		    for(roi_width_offset_index=0;roi_width_offset_index<num_center_width_offset_new;roi_width_offset_index++)
			{ 
				 center_point_x_w_offset =  center_point_x + width_offset_ary[roi_width_offset_index];
					
				 /*min max range check to prevent cv error*/
				if(center_point_x_w_offset<=1)   center_point_x_w_offset = 1;
				else if(center_point_x_w_offset>=preproc_resize_width)   center_point_x_w_offset = preproc_resize_width-1;

			   int num_center_height_offset_new;
			   int roi_height_offset_index;
			   
     		   if(midpoint_count>min_numof_lines_4_cluster)
		       {   
		            /* if roi width offset is zero*/
		            if(roi_width_offset_index == 0)    num_center_height_offset_new =  num_center_height_offset;
		            else   num_center_height_offset_new  = 1;
		            
		  
		           for(roi_height_offset_index = 0;roi_height_offset_index<num_center_height_offset_new;roi_height_offset_index++)
				   {
					    int height_offset =  height_offset_ary[roi_height_offset_index];
						
						int data_count = 0;
			            bool  invalid_depth_flag = false;
			             
						/* collect depth and y point along a line within the ROI */

						this->depth_data_collect(resized_depth_img_tmp_32f, array_depth_and_y_coor, fy, py, lower_limit+10+height_offset, upper_limit-10+height_offset, center_point_x_w_offset, &data_count, &invalid_depth_flag);

				        if(invalid_depth_flag==true)   continue;

		                if(data_count>min_numof_lines_4_cluster)
		                {  
		                    /*vector_for_learning include 8 elements*/
			                /* stair gradient , least square error(x coordinate of map frame (depth) and z coordinate of map frame(rows)) per line,  average error (y coordinate of map frame (cols)), center point x (cam frame), center point y(cam frame), depth, center point x pixel, center point y pixel*/
		                    std::vector<float>  vector_for_learning ;  
		   	                //std::cout<<"debug_line 7 \n"<<std::endl;
		                    cv::Mat  array_x_coor_final = array_x_coor(cv::Range(0, data_count), cv::Range::all());
		                    cv::Mat  array_depth_and_y_coor_final = array_depth_and_y_coor(cv::Range(0, data_count), cv::Range::all());
		                    cv::Mat  array_x_of_mid_point_final = array_x_of_mid_point_pixel(cv::Range(0, midpoint_count), cv::Range::all());
		                    cv::Mat  array_y_of_mid_point_final = array_y_of_mid_point_pixel(cv::Range(0, midpoint_count), cv::Range::all());
            
			                float gradient_out;
			                float deviation_cost_out;
			                float x_avg_error_out;
				            float continuity_factor_out;
		   
			                this->least_square_fit(array_x_coor_final, array_depth_and_y_coor_final, &gradient_out, &continuity_factor_out, &deviation_cost_out, &x_avg_error_out);
							     
						    float roi_h;
					        if(roi_size_index ==0)   roi_h = 200;
					        else   roi_h = 100;
						 
			                if(this->_svm_offline_training_flag == true)
	                        {  
	                           in<<function_call_count<<"/"<<gradient_out<<"/"<<continuity_factor_out<<"/"<<deviation_cost_out<<"/"<<x_avg_error_out*5<<"/"<<center_point_x_w_offset<<"/"<<center_point_y<<"/"<<roi_h<<"/"<<depth_center<<std::endl;
	                        } 
						
						    /* limit the value of outlier (gradient, gradient diff depth y error */
						    if(gradient_out>5)   gradient_out =5;
						    else if(gradient_out < -5)   gradient_out =-5;
						
						    if(continuity_factor_out>5)   continuity_factor_out =5;
				            else if(continuity_factor_out < -5)   continuity_factor_out =-5;
								
						    if(deviation_cost_out>10)   deviation_cost_out =10;

			                vector_for_learning.push_back(gradient_out);
				            vector_for_learning.push_back(continuity_factor_out);
			                vector_for_learning.push_back(deviation_cost_out);
			                vector_for_learning.push_back(x_avg_error_out);
			                vector_for_learning.push_back(center_point_x_coor);
			                vector_for_learning.push_back(center_point_y_coor);
		                    vector_for_learning.push_back(depth_center);
			                vector_for_learning.push_back(center_point_x);
			                vector_for_learning.push_back(center_point_y);
						    vector_for_learning.push_back(roi_h);
			 
			                vector_set_for_learning->push_back(vector_for_learning);
			
			                //std::cout<<"vector_set_for_learning : \n"<<std::endl;
			                //std::cout<<"gradient_out : \n"<<gradient_out<<std::endl;
			                //std::cout<<"deviation_cost_out : \n"<<deviation_cost_out<<std::endl;
			                //std::cout<<"x_avg_error_out : \n"<<x_avg_error_out<<std::endl;
			                //std::cout<<"center_point_x : \n"<<center_point_x<<std::endl;
			                //std::cout<<"center_point_y : \n"<<center_point_y<<std::endl;

		                }
			        }
			    }
            }
        }
    }		
    if(time_debug_flag == true){
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        printf("[INFO] cal_cost_wrapper() ---- took %.4lf ms (%.2lf Hz)\r\n", t*1000.0, (1.0/t));
     }
}


void STAIR_DETEC_COST_FUNC::least_square_fit(cv::Mat& array_x_coor_final_ls, cv::Mat& array_depth_and_y_coor_final_ls, float* gradient_out, float* gradient_sub_diff , float* avg_depth_y_error_out, float* x_avg_error_out)
{
	
	int least_square_index =  std::round(array_depth_and_y_coor_final_ls.rows/2);
	cv::Mat  array_depth_and_y_coor_final_sub1 = array_depth_and_y_coor_final_ls(cv::Range(0, least_square_index), cv::Range::all());
	cv::Mat  array_depth_and_y_coor_final_sub2 = array_depth_and_y_coor_final_ls(cv::Range(least_square_index, array_depth_and_y_coor_final_ls.rows), cv::Range::all());
	cv::Scalar avg_x=cv::mean(array_x_coor_final_ls);

	float error_abs_sum = 0;
	float x_avg_error=0;
	  
	int i;
	  
	for(i=0;i<array_x_coor_final_ls.rows;i++)
	{
		float error = array_x_coor_final_ls.at<float>(i)-avg_x.val[0];
		  
		error_abs_sum = error_abs_sum + std::abs(error);
		  
	}
	x_avg_error = error_abs_sum/array_x_coor_final_ls.rows;

	cv::Mat least_square_out(2,1,CV_32F);
	cv::Mat least_square_out_sub1(2,1,CV_32F);
	cv::Mat least_square_out_sub2(2,1,CV_32F);
	
	cv::Mat array_for_constant = cv::Mat::ones(array_depth_and_y_coor_final_ls.rows, 1, CV_32F);
	cv::Mat array_for_constant_sub1 = cv::Mat::ones(array_depth_and_y_coor_final_sub1.rows, 1, CV_32F);
	cv::Mat array_for_constant_sub2 = cv::Mat::ones(array_depth_and_y_coor_final_sub2.rows, 1, CV_32F);
	
	/*ax+by=c  --> (a/c)x + (b/c)y = 1 find (a/c) and (b/c)  gradient m is  a/b */
	cv::solve(array_depth_and_y_coor_final_ls, array_for_constant, least_square_out, cv::DECOMP_SVD);
	cv::solve(array_depth_and_y_coor_final_sub1, array_for_constant_sub1, least_square_out_sub1, cv::DECOMP_SVD);
	cv::solve(array_depth_and_y_coor_final_sub2, array_for_constant_sub2, least_square_out_sub2, cv::DECOMP_SVD);
	
	
	float gradient=0;
	float gradient_sub1=0;
	float gradient_sub2=0;
		
	if(std::abs(least_square_out.at<float>(1)) < 0.001)
	{
		 gradient = 1000;
	}
	else
	{
		gradient = -least_square_out.at<float>(0)/least_square_out.at<float>(1);
	}
	
	if(std::abs(least_square_out_sub1.at<float>(1)) < 0.001)
	{
		 gradient_sub1 = 1000;
	}
	else
	{
		gradient_sub1 = -least_square_out_sub1.at<float>(0)/least_square_out_sub1.at<float>(1);
	}
	
     if(std::abs(least_square_out_sub2.at<float>(1)) < 0.001)
	{
		 gradient_sub2 = 1000;
	}
	else
	{
		gradient_sub2 = -least_square_out_sub2.at<float>(0)/least_square_out_sub2.at<float>(1);
	}
	
	  
	 float depth_y_error=0;
	 float  depth_y_error_abs_sum = 0;
	 float  avg_depth_y_error=0;
	  
	 int n;
	  
	  bool non_stair_suspect_flag = false;
	  float non_stair_suspect_pivot_point;
	  float error_weight_factor = 1;
	  
	 for(n=0;n<array_depth_and_y_coor_final_ls.rows;n++)
	 {
		 /* error = (a/b)*x + 1/(b/c) - y */	  
	     depth_y_error= gradient*array_depth_and_y_coor_final_ls.at<float>(n,0)+1/least_square_out.at<float>(1) - array_depth_and_y_coor_final_ls.at<float>(n,1);
		 
		 if(((array_depth_and_y_coor_final_ls.at<float>(n-1,0)+0.3)<array_depth_and_y_coor_final_ls.at<float>(n,0))&&((n-1)>0)&&(non_stair_suspect_flag==false))
		 {
			 non_stair_suspect_flag = true;
			 non_stair_suspect_pivot_point = array_depth_and_y_coor_final_ls.at<float>(n+1,0);
			 error_weight_factor = 50;
		 }
		 else if((non_stair_suspect_flag == true)&&((non_stair_suspect_pivot_point+0.3)<array_depth_and_y_coor_final_ls.at<float>(n,0)))
		 {
			 non_stair_suspect_flag = true;
			 error_weight_factor = 50;
		 }
		 else if((non_stair_suspect_pivot_point>2)&&(non_stair_suspect_pivot_point>array_depth_and_y_coor_final_ls.at<float>(n,0)))
		 {
			 non_stair_suspect_flag = false;
			 error_weight_factor = 1;
		 }
		 	 
		 //std::cout<<"depth_y_error : "<< depth_y_error<<" \n"<<std::endl;
		 
		 depth_y_error_abs_sum = depth_y_error_abs_sum + std::abs(depth_y_error)*error_weight_factor;
	 }
	 
	 avg_depth_y_error = depth_y_error_abs_sum/array_depth_and_y_coor_final_ls.rows;
		 
     //std::cout<<"depth_y_error_abs_sum : "<< depth_y_error_abs_sum<<" \n"<<std::endl;
	 //std::cout<<"avg_depth_y_error : "<< avg_depth_y_error<<" \n"<<std::endl;
	 //std::cout<<"least_square_out.at<float>_1 : "<< least_square_out.at<float>(1)<<" \n"<<std::endl;
	 //std::cout<<"gradient : "<< gradient<<" \n"<<std::endl;
	 
	 *gradient_out=gradient;
	 *avg_depth_y_error_out=avg_depth_y_error;
	 *avg_depth_y_error_out=*avg_depth_y_error_out*10; /*scale up */
	 *x_avg_error_out=x_avg_error;
	 *gradient_sub_diff = std::abs(std::abs(gradient_sub1) - std::abs(gradient_sub2));
	 
	 function_call_count++;
	 
	
}

void STAIR_DETEC_COST_FUNC::depth_data_collect(cv::Mat& depth_img, cv::Mat& array_depth_y, float fy, float py, float lower_lim, float upper_lim, float width_point, int* data_cnt, bool* invalid_depth_on)
{
	int invalid_depth_count = 0;
	int interval = 2;	   
	int starting_point_y;
	
	for (starting_point_y=lower_lim; starting_point_y<upper_lim;starting_point_y=starting_point_y+interval)
	{  		
			int column_p2= std::round(width_point);
			float depth_tmp2 =  depth_img.at<float>(starting_point_y,column_p2);   /*depth image was  scaled using depth scale*/

			if(depth_tmp2>this->_invaild_depth_th)
			{
				invalid_depth_count = invalid_depth_count +1;
				continue;
			}
					   
			if(invalid_depth_count>=this->_invail_depth_count_th)
			{
			    *invalid_depth_on= true;
			    continue;
			}
					   
			if(depth_tmp2>0.1)
			{
				float y_coor_tmp = (py-(float)starting_point_y) * depth_tmp2 / fy;   /*there is a reason why py - starting_point_y  (not starting_point_y -py) */

				if(this->_coor_trans_flag == true)
				{
					geometry_msgs::PointStamped  point_in, point_out;
								  
					point_in.header.stamp = ros::Time::now();
                    point_in.header.frame_id = "human_view"; 
					point_in.point.x = 0;
					point_in.point.y = -y_coor_tmp;  /*since the sign of y_coor_tmp is reversed, restoring original sign*/ 
					point_in.point.z = depth_tmp2;
								  
					this->coordinate_transform(point_in,point_out,15,0.8);

					y_coor_tmp = -point_out.point.y; /*reversing the sign from the original sign convention*/
					depth_tmp2 = point_out.point.z;

				}
				else{}								 
	             
				array_depth_y.at<float>(*data_cnt,0) =  depth_tmp2+30;   /* the reason why adding 30 is to avoid singular problem of opencv solve algorithm*/
				array_depth_y.at<float>(*data_cnt,1) =  y_coor_tmp;
	 
				*data_cnt = *data_cnt + 1;   /* do not use *data_cnt++;  --> trap occurs */
			}
	} 	
}


void STAIR_DETEC_COST_FUNC::coordinate_transform(geometry_msgs::PointStamped& point_in, geometry_msgs::PointStamped& point_out, float head_down_angle, float height)
{
	static tf2_ros::Buffer _tfBuffer;
	static tf2_ros::TransformListener _tfListener(_tfBuffer);
	
	static tf2_ros::StaticTransformBroadcaster br;
	
	geometry_msgs::TransformStamped transformStamped;
	
	transformStamped.header.stamp = ros::Time::now();
	transformStamped.header.frame_id = "ground_cam";
	transformStamped.child_frame_id = "human_view";
	transformStamped.transform.translation.x = 0;
	transformStamped.transform.translation.y = -height; /*-(robot height(0.8m) - a_waltr height(0.2m)*/
	transformStamped.transform.translation.z = 0.0;
	
	 tf2::Quaternion t;
	 t.setRPY(-head_down_angle/57.3/*-20degree*/, 0, 0);
	 
	 transformStamped.transform.rotation.x = t.x();
	 transformStamped.transform.rotation.y = t.y();
	 transformStamped.transform.rotation.z = t.z();
	 transformStamped.transform.rotation.w = t.w();
	 
	 br.sendTransform(transformStamped);
	
	
	_tfBuffer.transform(point_in,point_out,"ground_cam");
}