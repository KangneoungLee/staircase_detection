#include "stair_detection/stair_detec_only_rgb.h"

int function_call_count = 0;

/*constructor and destructor*/
STAIR_DETEC_ONLY_RGB::STAIR_DETEC_ONLY_RGB()
{
}

STAIR_DETEC_ONLY_RGB::~STAIR_DETEC_ONLY_RGB()
{
}

bool STAIR_DETEC_ONLY_RGB::stair_case_detec_only_rgb(const cv::Mat& rgb_input, cv::Mat& roi_center_point_out, cv::Mat& midp_of_all_lines,int* final_center_point_col, int* final_center_point_row,
                                                                                                int min_numof_lines_4_cluster_rgb_only,  int predefined_roi_height, int predefined_roi_width, unsigned short preproc_resize_height, unsigned short preproc_resize_width)
{

	double t;
    if(time_debug_flag == true) t = (double)cv::getTickCount();
	
   int i;
   int max_count=0;
   int final_center_point_col_temp;
   int final_center_point_row_temp;
   bool stair_case_detec_flag = false;

   for( i = 0; i <  roi_center_point_out.rows; i++)
   { 
      
	  int count = 0;
	
	  /*if K_mean_cluster algortihm doesn't find suitable center position, it returns meaningless vlaue. Therefore, the invaild center position should be filtered out*/
	  if((roi_center_point_out.at<cv::Vec2f>(i)[0] <= 0)||(roi_center_point_out.at<cv::Vec2f>(i)[0] >= preproc_resize_width))
	  {
		    continue;
	   }
	  else if(( roi_center_point_out.at<cv::Vec2f>(i)[1]<=0)||(roi_center_point_out.at<cv::Vec2f>(i)[1]>=preproc_resize_height))
	   {
		    continue;
	   }

	   float center_point_x  = roi_center_point_out.at<cv::Vec2f>(i)[0];   /*pixel coordinate*/
	   float center_point_y = roi_center_point_out.at<cv::Vec2f>(i)[1];   /*pixel coordinate*/
       //float center_point_x  = 168.375;
	   //float center_point_y  = 197;
		  
	   int center_point_col = std::round(center_point_x);
	   int center_point_row = std::round(center_point_y);

	   float upper_limit = center_point_y + predefined_roi_height/2;     /*define the rectangle roi boundary*/
	   float lower_limit = center_point_y  - predefined_roi_height/2;     /*define the rectangle roi boundary*/
		  
	   /*upper_limit, lower_limit min_max range check to prevent cv error*/
	   if(upper_limit<15)
	   {
		   upper_limit = 15;
	   }
	   else if(upper_limit>(preproc_resize_height-15))
	   {
		   upper_limit =  preproc_resize_height-15;
	   }
          
       if(lower_limit<15)
	   {
		   lower_limit = 15;
	   }
	   else if(lower_limit>(preproc_resize_height-15))
	   {
		   lower_limit =  preproc_resize_height-15;
	   }		  
		  
	    float left_limit = center_point_x - predefined_roi_width/2;            /*define the rectangle roi boundary*/
	    float right_limit = center_point_x + predefined_roi_width/2;        /*define the rectangle roi boundary*/
	    
		  
		int max_y_pixel = 0;

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
					 	
					 if(row_p>max_y_pixel)
					 {
							  max_y_pixel = row_p;
					 }
					 count++;
				  } 
			} 
		}
        
		if(count>min_numof_lines_4_cluster_rgb_only)
		{
			stair_case_detec_flag = true;
			
			if(count>max_count)
			{
				max_count = count;
				final_center_point_col_temp = center_point_col;
				final_center_point_row_temp = center_point_row;
			}
			
		}
		
     }
	 
	 *final_center_point_col = final_center_point_col_temp;
	 *final_center_point_row = final_center_point_row_temp;
	 
	 
	 if(time_debug_flag == true){
     t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
     printf("[INFO] cal_cost_wrapper() ---- took %.4lf ms (%.2lf Hz)\r\n", t*1000.0, (1.0/t));
     }
	 
	 return stair_case_detec_flag;
}		
