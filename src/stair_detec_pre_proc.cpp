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

#include "stair_detection/stair_detec_pre_proc.h"
#include <iostream>
#include <fstream>
#include<sstream>

/*constructor and destructor*/
STAIR_DETEC_PRE_PROC::STAIR_DETEC_PRE_PROC(){}
STAIR_DETEC_PRE_PROC::~STAIR_DETEC_PRE_PROC(){}

void STAIR_DETEC_PRE_PROC::stair_line_extraction(cv::Mat rgb_image,   std::vector<cv::Vec4i> *edge_lines, cv::Mat* resized_rgb_image, cv::Mat* resized_gray_image, bool encoding_rgb_flag,
unsigned short preproc_resize_height, unsigned short preproc_resize_width,  unsigned short canny_lt, unsigned short canny_ht,  unsigned short houghp_th,  
unsigned short houghp_min_line_len, unsigned short houghp_max_line_gap, int noise_rm_pre_proc_index)
{
	cv::Mat resized_rgb_image_tmp;
	cv::Mat resized_gray_image_tmp;
	cv::Mat resized_gray_image_dilate_out;
	cv::Mat resized_gray_image_tmp_out_bilat, resized_gray_image_out;
	std::vector<cv::Vec4i>  edge_lines_tmp;
	
	cv::resize(rgb_image, resized_rgb_image_tmp, cv::Size(preproc_resize_width,preproc_resize_height), cv::INTER_AREA);  /*resize image to reduce computation cost */
	
	*resized_rgb_image=resized_rgb_image_tmp.clone();  /*allocate resize rgb image to output */
	
	if(encoding_rgb_flag == true)   /*gray conversion depends on rgb encoding method */
	{
	    cv::cvtColor(resized_rgb_image_tmp, resized_gray_image_tmp, cv::COLOR_RGB2GRAY);
		//printf("[INFO] RGB to GRAY \n");
	}
	else
	{
	    cv::cvtColor(resized_rgb_image_tmp, resized_gray_image_tmp, cv::COLOR_BGR2GRAY);
		//printf("[INFO] BGR to GRAY \n");
	}
	
	 double t;
     if(time_debug_flag == true) t = (double)cv::getTickCount();
	 
	cv::Mat element = getStructuringElement( cv::MORPH_ELLIPSE , cv::Size( 20,10),cv::Point( -1, -1) );  /* Size (width, height),  cv::Point( -1, -1) means center point of kernel*/
	
	if(noise_rm_pre_proc_index == 1)   /*dilate*/
	{
	    cv::dilate(resized_gray_image_tmp , resized_gray_image_dilate_out, element );	
	}
	else if(noise_rm_pre_proc_index == 2)  /*erode*/
	{ 
      cv::erode(resized_gray_image_tmp , resized_gray_image_dilate_out, element );
	}
	else  /*no noise remove*/
	{
		resized_gray_image_dilate_out = resized_gray_image_tmp.clone();
	}
	 //cv::bilateralFilter(resized_gray_image_dilate_out, resized_gray_image_tmp_out_bilat, 5, 15, 20);
	 
	cv::Canny(resized_gray_image_dilate_out, resized_gray_image_out, canny_lt, canny_ht, 3);
	
	/*HoughLinesP output lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines*/
	cv::HoughLinesP(resized_gray_image_out, edge_lines_tmp, 1, CV_PI / 60, houghp_th, houghp_min_line_len, houghp_max_line_gap);
	

	 
	
	if(!edge_lines_tmp.empty())
	{   
 
		//std::cout << "edge_lines_tmp not empty: " "\n"<< std::endl;

        int i;
		for(i=0; i<edge_lines_tmp.size(); i++)
		{
			edge_lines->push_back(edge_lines_tmp[i]);
		}

	}
	 


 /***************************** visualization for debugging *************************************************************/
	if(visualize_flag>=1)
	{  
        cv::Mat  resized_gray_image_with_line;
		
		resized_gray_image_with_line = resized_gray_image_out.clone();
		
		if(visualize_flag>=2)
		{
           //cv::namedWindow( "edge_line", cv::WINDOW_AUTOSIZE );
	       cv::imshow("edge_line", resized_gray_image_out);
		   cv::waitKey(30);
		
	        // cv::imshow("edge_line_before_df", resized_canny_image_out);
		   //cv::waitKey(30);
		
	       cv::imshow("image_with_dilate", resized_gray_image_dilate_out);
		   cv::waitKey(30);
		}

		
		int i;
		
		if(!edge_lines->empty())
		{
		   for(i = 0 ; i< edge_lines->size() ; i ++)
		   {  
	           cv::Vec4i single_line = (*edge_lines)[i];
			   cv::line(resized_rgb_image_tmp, cv::Point(single_line[0], single_line[1]), cv::Point(single_line[2], single_line[3]), cv::Scalar(255, 0, 0));
		   }
		}
		
		if(visualize_flag>=2)
		{
		   //cv::namedWindow( "edge_line of hough P", cv::WINDOW_AUTOSIZE );
	       cv::imshow("edge_line of hough P", resized_rgb_image_tmp);
		   cv::waitKey(30);
		}
		
	}

	if(time_debug_flag == true){
          t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
          printf("[INFO] stair_line_extraction() ---- took %.4lf ms (%.2lf Hz)\r\n", t*1000.0, (1.0/t));
     }
	
	*resized_gray_image = resized_gray_image_out.clone();
	
}

void STAIR_DETEC_PRE_PROC::select_lines_from_hough(const cv::Mat& rgb_input, const cv::Mat& gray_input, std::vector<cv::Vec4i> *org_edge_lines,  std::vector<cv::Vec4i> *final_edge_lines, 
                                                                                                                      std::map<float, std::vector<cv::Vec4i>> *lines_hist, std::map<int, std::vector<cv::Vec4i>> *lines_hist_top_three, float maxslope_in_pixel , float minslope_in_pixel)
{
	float slope_tmp;
	float valid_slope_tmp_sum = 0;
	float slope_max=0;
	float slope_min=0;
	float abs_slope_tmp;
	std::vector<float> slope_list;  /* m list is the graident vector for selected lines */ 

	float slope_range_len = maxslope_in_pixel*2;
	float slope_interval_num = 2;
	float slope_interval[4];
	float avg_vaild_slope = 0;

	double t;
     if(time_debug_flag == true) t = (double)cv::getTickCount();
	
	if(!org_edge_lines->empty())
	{
		
	//	std::cout << "org_edge_lines not empty: \n"<< std::endl;
	//	std::cout << "org_edge_lines size: "<< org_edge_lines->size()<<" \n"<< std::endl;
		
	 int i;
	/*collect the graidents of lines which meet a defined criteria*/
	 int valid_slope_count =0; 
	 
       for(i = 0 ; i< org_edge_lines->size() ; i ++)
	   {
		   cv::Vec4f single_line = (*org_edge_lines)[i];   /*the type of point should be float. (should be cv::Vec4f  if cv::Vec4i --> truncation error occurs when calculating the slop*/
		
		   if(abs(single_line[0] - single_line[2] ) > 3)
		   {
			    slope_tmp = (single_line[3] - single_line[1])/(single_line[2] - single_line[0]);
			    abs_slope_tmp = abs(slope_tmp);
			 
			    if((abs_slope_tmp < maxslope_in_pixel))
			    {     
			           if(slope_tmp>slope_max)
					   {
						   slope_max = slope_tmp;
					   }
					   else if(slope_tmp<slope_min)
					   {
						   slope_min = slope_tmp;
					   }
					   
					   valid_slope_tmp_sum = valid_slope_tmp_sum + slope_tmp;
					   
					 //std::cout << "original slope_max  : \n"<< slope_max<<std::endl;
		             //std::cout << "original slope_min  : \n"<< slope_min<<std::endl;
					   
				 	   slope_list.push_back(slope_tmp);
					   final_edge_lines->push_back(single_line);
					   
					   valid_slope_count++;
					   
					   avg_vaild_slope =  valid_slope_tmp_sum/valid_slope_count;
			    }
		   }
	   }		
    }
	

	float slope_min_max_diff = slope_max-slope_min;  
	
	/* if slope min value and slope max value is too close, the logic can be sensitvie so forced to enlarge the slope range*/
	if((slope_min_max_diff) < 0.1*slope_interval_num)
	{
		std::cout << "slope diff is too small : \n"<< std::endl;

		slope_max = slope_max +(0.1*slope_interval_num - slope_min_max_diff)/2;
		slope_min = slope_min -(0.1*slope_interval_num - slope_min_max_diff)/2;
		
		slope_interval_num = 1;
		
		slope_interval[0] = slope_min;
        slope_interval[1] = slope_max;		
		slope_interval[2] = 0;   /*dummy array element*/
        slope_interval[3] = 0;	/*dummy array element*/	
		/*if slope difference is too small, then there is no seperation of slope */
		
	}
	else
	{
		slope_interval_num =3;
		
		slope_interval[0] = slope_min; 
		slope_interval[1] = slope_min + 0.5*(avg_vaild_slope - slope_min);
		slope_interval[2] = slope_max + 0.5*(slope_max - avg_vaild_slope);
		slope_interval[3] = slope_max;
		
	}
	
	//std::cout << "slope_incremental[0]  :"<< slope_interval[0]<< "\n"<<std::endl;
	//std::cout << "slope_incremental[1]  :"<< slope_interval[1]<< "\n"<<std::endl;
	//std::cout << "slope_incremental[2]  :"<< slope_interval[2]<< "\n"<<std::endl;
	//std::cout << "slope_incremental[3]  :"<< slope_interval[3]<< "\n"<<std::endl;
	
	int k;
	
	for(k=0; k<slope_interval_num; k++)
	{
		float map_index =  slope_interval[k];
		float lower_slope_lim = slope_interval[k];
		float upper_slope_lim = slope_interval[k+1];
		std::vector<cv::Vec4i> map_lines_w_index;
	    
		int i;
		
		if(!slope_list.empty())
		{
			
		 // std::cout << "final_edge_lines not empty: \n"<< std::endl;
		//  std::cout << "final_edge_lines size: "<< final_edge_lines->size()<<" \n"<< std::endl;
			
	       for(i=0; i< slope_list.size(); i++)
     	   {
	           if((slope_list[i]>=lower_slope_lim)&&(slope_list[i]<upper_slope_lim))
			   {
				   map_lines_w_index.push_back((*final_edge_lines)[i]);	
			   }
		   }
		}
		
		if(!map_lines_w_index.empty())
		{
		    lines_hist->insert(std::pair<float, std::vector<cv::Vec4i>>(map_index,map_lines_w_index));
		}
	}
	
     float slope_tmp_for_debug_first= maxslope_in_pixel*(-1);
	 float slope_tmp_for_debug_second=maxslope_in_pixel*(-1);
	 float slope_tmp_for_debug_third=maxslope_in_pixel*(-1);
	 float slope_tmp_for_debug_cnt_first=0;
	 float slope_tmp_for_debug_cnt_second=0;
	 float slope_tmp_for_debug_cnt_third=0;
		 
	 std::vector<cv::Vec4i> map_lines_for_debug_index;
    
	 /* count the number of lines  and sort the map based on the counting*/
	if(!lines_hist->empty())
	{ 
	     std::map<float, std::vector<cv::Vec4i>>::iterator it;
			
		for(it = lines_hist->begin(); it != lines_hist->end(); it++)
		{
             map_lines_for_debug_index =it->second;
               
			 if(map_lines_for_debug_index.size() >slope_tmp_for_debug_cnt_first)
			 {
				 slope_tmp_for_debug_third=slope_tmp_for_debug_second;
				 slope_tmp_for_debug_second = slope_tmp_for_debug_first;
				 slope_tmp_for_debug_first = it->first;
				   
				 slope_tmp_for_debug_cnt_third = slope_tmp_for_debug_cnt_second;
				 slope_tmp_for_debug_cnt_second = slope_tmp_for_debug_cnt_first;
				 slope_tmp_for_debug_cnt_first = map_lines_for_debug_index.size();
				  
			 } 
             else if(map_lines_for_debug_index.size() >  slope_tmp_for_debug_cnt_second)
			 {
				 slope_tmp_for_debug_third=slope_tmp_for_debug_second;
			     slope_tmp_for_debug_second = it->first;
					
				 slope_tmp_for_debug_cnt_third = slope_tmp_for_debug_cnt_second;
			     slope_tmp_for_debug_cnt_second = map_lines_for_debug_index.size();

			 }
			else if(map_lines_for_debug_index.size()>slope_tmp_for_debug_cnt_third)
			{
				      slope_tmp_for_debug_third = it->first;
				   
				      slope_tmp_for_debug_cnt_third = map_lines_for_debug_index.size();
			 }
	    } 
			
	 }

     std::vector<cv::Vec4i> map_lines_for_debug_index_first = (*lines_hist)[slope_tmp_for_debug_first];
     std::vector<cv::Vec4i> map_lines_for_debug_index_second = (*lines_hist)[slope_tmp_for_debug_second];
     std::vector<cv::Vec4i> map_lines_for_debug_index_third= (*lines_hist)[slope_tmp_for_debug_third];
	 
	// std::cout << "map_lines_for_debug_index_first size  :"<< map_lines_for_debug_index_first.size()<< "\n"<<std::endl;
	// std::cout << "map_lines_for_debug_index_second size  :"<< map_lines_for_debug_index_second.size()<< "\n"<<std::endl;
	// std::cout << "map_lines_for_debug_index_third size  :"<< map_lines_for_debug_index_third.size()<< "\n"<<std::endl;
		
	 /* lines_hist_top_three : final output of map (top 3 vectors (based on the number of lines) */
	 lines_hist_top_three->insert(std::pair<int, std::vector<cv::Vec4i>>(1,map_lines_for_debug_index_first));
	 lines_hist_top_three->insert(std::pair<int, std::vector<cv::Vec4i>>(2,map_lines_for_debug_index_second));
	 lines_hist_top_three->insert(std::pair<int, std::vector<cv::Vec4i>>(3,map_lines_for_debug_index_third));
		

	 if(time_debug_flag == true){
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        printf("[INFO] select_lines_from_hough() ---- took %.4lf ms (%.2lf Hz)\r\n", t*1000.0, (1.0/t));
     }
 
 
 /***************************** visualization for debugging *************************************************************/
	 if(visualize_flag>=1)
	 {
		 int i;
		 
         cv::Mat input2 =rgb_input.clone();
		 
		 for(i = 0 ; i< final_edge_lines->size() ; i ++)
		{  
	        cv::Vec4i single_line = (*final_edge_lines)[i];
			cv::line(rgb_input, cv::Point(single_line[0], single_line[1]), cv::Point(single_line[2], single_line[3]), cv::Scalar(255, 0, 0));
		}
		 
		 if(visualize_flag>=2)
		 {
		   cv::namedWindow( "edge_line_w_maximum_gradient_reject", cv::WINDOW_AUTOSIZE );
	       cv::imshow("edge_line_w_maximum_gradient_reject", rgb_input);
		 }
		 
		 if(!map_lines_for_debug_index_first.empty())
		 {
			 
			//std::cout << "map_lines_for_debug_index_first not empty: \n"<< std::endl;
		    //std::cout << "map_lines_for_debug_index_first size: "<< map_lines_for_debug_index_first.size()<<" \n"<< std::endl;
			 
		    for(i = 0 ; i< map_lines_for_debug_index_first.size() ; i ++)
		    {  
	           cv::Vec4i single_line = map_lines_for_debug_index_first[i];
			   cv::line(input2, cv::Point(single_line[0], single_line[1]), cv::Point(single_line[2], single_line[3]), cv::Scalar(255, 0, 0));
		    }
		 }
		
		 
		 if(!map_lines_for_debug_index_second.empty())
		 {
			 //std::cout << "map_lines_for_debug_index_second not empty: \n"<< std::endl;
		    //std::cout << "map_lines_for_debug_index_second size: "<< map_lines_for_debug_index_second.size()<<" \n"<< std::endl;
			 
    		 for(i = 0 ; i< map_lines_for_debug_index_second.size() ; i ++)
		    {  
	           cv::Vec4i single_line = map_lines_for_debug_index_second[i];
			   cv::line(input2, cv::Point(single_line[0], single_line[1]), cv::Point(single_line[2], single_line[3]), cv::Scalar(0, 255, 0));
		    }
		 }
		 
		 
		 
		 if(!map_lines_for_debug_index_third.empty())
		 {
			 
			 //std::cout << "map_lines_for_debug_index_third not empty: \n"<< std::endl;
		    //std::cout << "map_lines_for_debug_index_third size: "<< map_lines_for_debug_index_third.size()<<" \n"<< std::endl;
			 
		   for(i = 0 ; i< map_lines_for_debug_index_third.size() ; i ++)
		   {   
	           cv::Vec4i single_line = map_lines_for_debug_index_third[i];
			   cv::line(input2, cv::Point(single_line[0], single_line[1]), cv::Point(single_line[2], single_line[3]), cv::Scalar(0, 0, 255));
		   }
		 }
		 
		 if(visualize_flag>=2)
		 {
		    cv::namedWindow( "edge_line_grouping 1st 2nd 3th", cv::WINDOW_AUTOSIZE );
	        cv::imshow("edge_line_grouping 1st 2nd 3th", input2);
		 }
	 }
	  
}

void STAIR_DETEC_PRE_PROC::grouping_lines_and_define_centers(const cv::Mat& rgb_input, std::map<int, std::vector<cv::Vec4i>> *lines_hist_top_three,  cv::Mat* roi_center_point_out, cv::Mat* midp_of_all_lines,
int min_numof_lines_4_cluster, int predefined_roi_height, int predefined_roi_width)
{   
   /*Mat class to store mid point of lines*/
  
	cv::Mat  center_p_of_1stlines((*lines_hist_top_three)[1].size(), 1 , CV_32FC2);   
	cv::Mat  center_p_of_2ndlines((*lines_hist_top_three)[2].size(), 1 ,CV_32FC2);
	
	std::vector<cv::Vec4i>::iterator it; 
	float center_p_x;
	float center_p_y;
	
	int clusterCount =3;
	int total_n_of_midpoints;
	
	double t;
   if(time_debug_flag == true) t = (double)cv::getTickCount();
	
	
   if(!(*lines_hist_top_three)[1].empty())
    {	
        int i = 0;
   	
	   for(it = (*lines_hist_top_three)[1].begin(); it!=(*lines_hist_top_three)[1].end(); ++it)
	   {
		   center_p_x = ((*it)[0] + (*it)[2])/2;
		   center_p_y= ((*it)[1] + (*it)[3])/2;
		   
		   center_p_of_1stlines.at<cv::Vec2f>(i)[0] = center_p_x;
		  center_p_of_1stlines.at<cv::Vec2f>(i)[1] = center_p_y;
		  
		    i ++;
	   }
    }
   
    if(!(*lines_hist_top_three)[2].empty())
    {	
        int i = 0;   
	   for(it = (*lines_hist_top_three)[2].begin(); it!=(*lines_hist_top_three)[2].end(); ++it)
	   {
		   center_p_x = ((*it)[0]+ (*it)[2])/2;
		   center_p_y= ((*it)[1] + (*it)[3])/2;
		   
		   center_p_of_2ndlines.at<cv::Vec2f>(i)[0]  = center_p_x;
		   center_p_of_2ndlines.at<cv::Vec2f>(i)[1]  = center_p_y;
		  
		    i ++;
	   }
    }
   
	
   cv::Mat center_p_of_1stlines_centers = cv::Mat::zeros(clusterCount, 2, CV_32F); 
   cv::Mat center_p_of_1stlines_labels =cv::Mat::zeros(clusterCount, 1, CV_32S); 
   
   if(center_p_of_1stlines.rows>min_numof_lines_4_cluster)
    {

	     float errorsquaresum_1st = cv::kmeans(center_p_of_1stlines, clusterCount, center_p_of_1stlines_labels,
            cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 5, 10.0),
               4, cv::KMEANS_PP_CENTERS, center_p_of_1stlines_centers);
			   
			   /* type of  center_p_of_1stlines_centers is Mat class with number of label  by  number of dimensions */
			   /* if we feed (x,y) coordinate and set 3 labels for classification, the output of center is  3 by 2 mat class , first column means x coordinate and second column means y cooridnate*/ 
			   
			 //std::cout<<"errorsquaresum_1st : "<<errorsquaresum_1st <<"\n"<<std::endl;
    }
	
   cv::Mat center_p_of_2ndlines_centers = cv::Mat::zeros(clusterCount, 2, CV_32F);
   cv::Mat center_p_of_2ndlines_labels;
   
    if(center_p_of_2ndlines.rows>min_numof_lines_4_cluster)
    {
	     float errorsquaresum_2nd = cv::kmeans(center_p_of_2ndlines, clusterCount, center_p_of_2ndlines_labels,
            cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 5, 10.0),
               4, cv::KMEANS_PP_CENTERS, center_p_of_2ndlines_centers);
			   
			   
			 //std::cout<<"errorsquaresum_2nd : "<<errorsquaresum_2nd <<"\n"<<std::endl;
	
    }
	int number_of_centers = 0;
	
	if(!center_p_of_1stlines_centers.empty())
	{
		number_of_centers =number_of_centers +clusterCount;
	}
	
	if(!center_p_of_2ndlines_centers.empty())
	{
		number_of_centers =number_of_centers +clusterCount;
	}
	
	cv::Mat  roi_center_point_out_tmp(number_of_centers, 1 ,CV_32FC2);     /*6*1 matrix and each cell include  (center x, center y)*/
	
	if(!center_p_of_1stlines_centers.empty())
	{
	     roi_center_point_out_tmp.at<cv::Vec2f>(0)[0] =  center_p_of_1stlines_centers.at<float>(0,0);   /* first group  first center point x */
	     roi_center_point_out_tmp.at<cv::Vec2f>(0)[1] =  center_p_of_1stlines_centers.at<float>(0,1);   /* first group  first center point y */
	
         roi_center_point_out_tmp.at<cv::Vec2f>(1)[0] =  center_p_of_1stlines_centers.at<float>(1,0);   /* first group  second center point x */
	     roi_center_point_out_tmp.at<cv::Vec2f>(1)[1] =  center_p_of_1stlines_centers.at<float>(1,1);   /* first group  second center point y */
		 
		 roi_center_point_out_tmp.at<cv::Vec2f>(2)[0] =  center_p_of_1stlines_centers.at<float>(2,0);   /* first group  second center point x */
	     roi_center_point_out_tmp.at<cv::Vec2f>(2)[1] =  center_p_of_1stlines_centers.at<float>(2,1);   /* first group  second center point y */
		 

	}
	
	if(!center_p_of_2ndlines_centers.empty())
	{
	
          roi_center_point_out_tmp.at<cv::Vec2f>(3)[0] =  center_p_of_2ndlines_centers.at<float>(0,0);   /* second group  first center point x */
	      roi_center_point_out_tmp.at<cv::Vec2f>(3)[1] =  center_p_of_2ndlines_centers.at<float>(0,1);   /* second group  first center point y */
	
          roi_center_point_out_tmp.at<cv::Vec2f>(4)[0] =  center_p_of_2ndlines_centers.at<float>(1,0);   /* second group  second center point x */
	      roi_center_point_out_tmp.at<cv::Vec2f>(4)[1] =  center_p_of_2ndlines_centers.at<float>(1,1);   /* second group  second center point y */

          roi_center_point_out_tmp.at<cv::Vec2f>(5)[0] = center_p_of_2ndlines_centers.at<float>(0,0);   /* second group  first center point x */ 
	      roi_center_point_out_tmp.at<cv::Vec2f>(5)[1] = center_p_of_2ndlines_centers.at<float>(0,1);   /* second group  first center point y */ 
		  	  
	} 
	
	*roi_center_point_out = roi_center_point_out_tmp.clone();  /*function output */
	
	

	int midpoint_n_first, midpoint_n_second;
	
	if(!(*lines_hist_top_three)[1].empty())
	{
		midpoint_n_first =  (*lines_hist_top_three)[1].size();
	}
	else
	{
		midpoint_n_first = 0;
	}
	
	
	if(!(*lines_hist_top_three)[2].empty())
	{
		midpoint_n_second =  (*lines_hist_top_three)[2].size();
	}
	else
	{
		midpoint_n_second = 0;
	}
	
	
	total_n_of_midpoints=midpoint_n_first + midpoint_n_second; 
	
	cv::Mat  midp_of_all_lines_tmp(total_n_of_midpoints, 1 , CV_32FC2);   
	
	int m, k ;
	
	for(m=0; m<midpoint_n_first; m++)
	{
		
		midp_of_all_lines_tmp.at<cv::Vec2f>(m)[0] =  center_p_of_1stlines.at<cv::Vec2f>(m)[0];
		midp_of_all_lines_tmp.at<cv::Vec2f>(m)[1] =  center_p_of_1stlines.at<cv::Vec2f>(m)[1];
	}
	
	//for(k=0;k<midpoint_n_second; k++)
	//{
	//	midp_of_all_lines_tmp.at<cv::Vec2f>(m)[0] =  center_p_of_2ndlines.at<cv::Vec2f>(k)[0];
	//	midp_of_all_lines_tmp.at<cv::Vec2f>(m)[1] =  center_p_of_2ndlines.at<cv::Vec2f>(k)[1];
		
	//	m++;
	//}
	
	//std::cout<<" m value  :"<< m<<"\n"<<std::endl;
	//std::cout<<" k value  :"<< k<<"\n"<<std::endl;
	//std::cout<<" total_n_of_midpoints  :"<< total_n_of_midpoints<<"\n"<<std::endl;
	//std::cout<<" midpoint_n_first :"<< midpoint_n_first<<"\n"<<std::endl;
	//std::cout<<" midpoint_n_second  :"<<midpoint_n_second<<"\n"<<std::endl;
	
	*midp_of_all_lines = midp_of_all_lines_tmp.clone(); /*function  output */
	
	
   	 if(time_debug_flag == true){
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        printf("[INFO] grouping_lines_and_define_centers() ---- took %.4lf ms (%.2lf Hz)\r\n", t*1000.0, (1.0/t));
     }	
	 cv::Point2i temp_point;
	 
	if(visualize_flag>=1)
	{
		if(!center_p_of_1stlines_centers.empty())
		{ 
//		   if(center_p_of_1stlines_centers.at<float>(0,0)<center_p_of_1stlines_centers.at<float>(1,0))
//		   {
//			   temp_point.x = center_p_of_1stlines_centers.at<float>(0,0);
//	           temp_point.y = center_p_of_1stlines_centers.at<float>(0,1);
		
//	           cv::circle( rgb_input, temp_point, 10/*radius*/, cv::Scalar(  255,   255,   255), 4/*thickness*/ );
	   
//	           temp_point.x = center_p_of_1stlines_centers.at<float>(1,0);
//	           temp_point.y = center_p_of_1stlines_centers.at<float>(1,1);
	   
//	           cv::circle( rgb_input,temp_point, 10/*radius*/, cv::Scalar(  0,   0,   0), 4/*thickness*/);
			   
//		   }
//		   else
//		   {
//			   temp_point.x = center_p_of_1stlines_centers.at<float>(0,0);
//	           temp_point.y = center_p_of_1stlines_centers.at<float>(0,1);
		
//	           cv::circle( rgb_input,temp_point, 10/*radius*/, cv::Scalar(  0,   0,   0), 4/*thickness*/);
	   
//	           temp_point.x = center_p_of_1stlines_centers.at<float>(1,0);
//	           temp_point.y = center_p_of_1stlines_centers.at<float>(1,1);
	   
//			   cv::circle( rgb_input, temp_point, 10/*radius*/, cv::Scalar(  255,   255,   255), 4/*thickness*/ );
//		   }

		
	      if(!center_p_of_1stlines.empty())
	      {
	         int i;

	         for (i = 0; i < center_p_of_1stlines.rows; i++)
	          {  
                  temp_point.x = std::abs(center_p_of_1stlines.at<cv::Vec2f>(i)[0]);
			      temp_point.y = std::abs(center_p_of_1stlines.at<cv::Vec2f>(i)[1]);
				  
				 // std::cout<<"temp_point x \n"<<temp_point.x<<std::endl;
				  //std::cout<<" temp_point y \n"<<temp_point.y<<std::endl;
				  //std::cout<<" center_p_of_1stlines_labels  \n"<<  center_p_of_1stlines_labels.at<int>(i)<<std::endl;
				
			
                 // std::cout<<" integer i :"<< i << "    label at <int> i : " <<     center_p_of_1stlines_labels.at<int>(i)  << " center_p_of_1stlines.at<cv::Vec2f>(i)  : " <<  center_p_of_1stlines.at<cv::Vec2f>(i) << "\n" << std::endl;
			 
		         if(center_p_of_1stlines_labels.at<int>(i) ==0)
		         {  
					 if(center_p_of_1stlines_centers.at<float>(0,0)<center_p_of_1stlines_centers.at<float>(1,0))
					 {
				 
					        if(center_p_of_1stlines_centers.at<float>(0,0)<center_p_of_1stlines_centers.at<float>(2,0))
							{
					
							       cv::circle( rgb_input, temp_point, 5, cv::Scalar(  255,   255,   255), 2/*thickness*/ );	
							}
			                else
							{
								   cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   0,   0), 2/*thickness*/ );
							}
					  }
					 else
					 {  
					        if(center_p_of_1stlines_centers.at<float>(0,0)<center_p_of_1stlines_centers.at<float>(2,0))
							{
						        cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   0,   0), 2/*thickness*/ );		
							}
						    else
							{
								cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   255,   0), 2/*thickness*/ );		
							}
					  }
		         }
		         else if(center_p_of_1stlines_labels.at<int>(i) == 1)
		         {  
				     if(center_p_of_1stlines_centers.at<float>(0,0)<center_p_of_1stlines_centers.at<float>(1,0))
					 {   
					     if(center_p_of_1stlines_centers.at<float>(2,0)<center_p_of_1stlines_centers.at<float>(1,0))
						 {
							 cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   255,   0), 2/*thickness*/ );
						 }
						 else
						 {
						     cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   0,   0), 2/*thickness*/ );	 
						 }
					  }
					  else
					  { 
					     
						 if(center_p_of_1stlines_centers.at<float>(2,0)<center_p_of_1stlines_centers.at<float>(1,0))
						 {
							  cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   0,   0), 2/*thickness*/ );	 
						 }
					     else
						 {
                             cv::circle( rgb_input, temp_point, 5, cv::Scalar(  255,   255,   255), 2/*thickness*/ );
						 }							  
					  }
		         }
				 else if(center_p_of_1stlines_labels.at<int>(i) == 2)
		         {	 
				     if(center_p_of_1stlines_centers.at<float>(0,0)<center_p_of_1stlines_centers.at<float>(2,0))
					 {   
					     if(center_p_of_1stlines_centers.at<float>(1,0)<center_p_of_1stlines_centers.at<float>(2,0))
						 {
							 cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   255,   0), 2/*thickness*/ );
						 }
						 else
						 {
						     cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   0,   0), 2/*thickness*/ );	 
						 }
					  }
					  else
					  { 
						 if(center_p_of_1stlines_centers.at<float>(1,0)<center_p_of_1stlines_centers.at<float>(2,0))
						 {
							  cv::circle( rgb_input, temp_point, 5, cv::Scalar(  0,   0,   0), 2/*thickness*/ );	 
						 }
					     else
						 {
                             cv::circle( rgb_input, temp_point, 5, cv::Scalar(  255,   255,   255), 2/*thickness*/ );
						 }							  
					  }
		         }
	          }
	        }
		   
		   	cv::Point2f  rectangle_tmp1_pt1, rectangle_tmp1_pt2;
		    cv::Point2f  rectangle_tmp2_pt1, rectangle_tmp2_pt2;
			cv::Point2f  rectangle_tmp3_pt1, rectangle_tmp3_pt2;
		
		    rectangle_tmp1_pt1.x =center_p_of_1stlines_centers.at<float>(0,0)-100 ;
		    rectangle_tmp1_pt1.y =center_p_of_1stlines_centers.at<float>(0,1)+100/1 ;
		
		    rectangle_tmp1_pt2.x =center_p_of_1stlines_centers.at<float>(0,0)+100 ;
		    rectangle_tmp1_pt2.y =center_p_of_1stlines_centers.at<float>(0,1)-100/1 ;
		
			rectangle_tmp2_pt1.x =center_p_of_1stlines_centers.at<float>(1,0)-100 ;
		    rectangle_tmp2_pt1.y =center_p_of_1stlines_centers.at<float>(1,1)+100/1 ;
		
		    rectangle_tmp2_pt2.x =center_p_of_1stlines_centers.at<float>(1,0)+100 ;
		    rectangle_tmp2_pt2.y =center_p_of_1stlines_centers.at<float>(1,1)-100/1;
			
			rectangle_tmp3_pt1.x =center_p_of_1stlines_centers.at<float>(2,0)-100 ;
		    rectangle_tmp3_pt1.y =center_p_of_1stlines_centers.at<float>(2,1)+100/1 ;
		
		    rectangle_tmp3_pt2.x =center_p_of_1stlines_centers.at<float>(2,0)+100 ;
		    rectangle_tmp3_pt2.y =center_p_of_1stlines_centers.at<float>(2,1)-100/1;

	        if(center_p_of_1stlines_centers.at<float>(0,0) < center_p_of_1stlines_centers.at<float>(1,0))
	        {
				 if(center_p_of_1stlines_centers.at<float>(0,0) < center_p_of_1stlines_centers.at<float>(2,0))
				 {
				     cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  255,   255,   255),2/*thickness*/);
 					 
					 if(center_p_of_1stlines_centers.at<float>(1,0) < center_p_of_1stlines_centers.at<float>(2,0))
					 {
						 cv::rectangle(rgb_input, rectangle_tmp2_pt1, rectangle_tmp2_pt2, cv::Scalar(  0,   0,   0),2/*thickness*/);
						 cv::rectangle(rgb_input, rectangle_tmp3_pt1, rectangle_tmp3_pt2, cv::Scalar(  0,   255,   0),2/*thickness*/);
					 }
					 else
					 {
						 cv::rectangle(rgb_input, rectangle_tmp2_pt1, rectangle_tmp2_pt2, cv::Scalar(  0,  255,   0),2/*thickness*/);
						 cv::rectangle(rgb_input, rectangle_tmp3_pt1, rectangle_tmp3_pt2, cv::Scalar(  0,   0,   0),2/*thickness*/); 
					 }
				 }
	             else
				 {
					 cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  0,   0,   0),2/*thickness*/);
					 cv::rectangle(rgb_input, rectangle_tmp2_pt1, rectangle_tmp2_pt2, cv::Scalar(  0,  255,   0),2/*thickness*/);
					 cv::rectangle(rgb_input, rectangle_tmp3_pt1, rectangle_tmp3_pt2, cv::Scalar(  255,   255,   255),2/*thickness*/); 
					 
				 }
	           
	        }
           else
	       {
			   if(center_p_of_1stlines_centers.at<float>(0,0) < center_p_of_1stlines_centers.at<float>(2,0))
			   {
				  cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  0,   0,   0),2/*thickness*/);
	              cv::rectangle(rgb_input, rectangle_tmp2_pt1, rectangle_tmp2_pt2, cv::Scalar(  255,   255,   255),2/*thickness*/);
				  cv::rectangle(rgb_input, rectangle_tmp3_pt1, rectangle_tmp3_pt2, cv::Scalar(  0,  255,   0),2/*thickness*/);
			   }
			   else
			   {   
			        cv::rectangle(rgb_input, rectangle_tmp1_pt1, rectangle_tmp1_pt2, cv::Scalar(  0,   255,   0),2/*thickness*/);
					
				    if(center_p_of_1stlines_centers.at<float>(1,0) < center_p_of_1stlines_centers.at<float>(2,0))
					{
						 cv::rectangle(rgb_input, rectangle_tmp2_pt1, rectangle_tmp2_pt2, cv::Scalar(  255,   255,   255),2/*thickness*/);
						 cv::rectangle(rgb_input, rectangle_tmp3_pt1, rectangle_tmp3_pt2, cv::Scalar(  0,  0,   0),2/*thickness*/);
					}
					else
					{
						 cv::rectangle(rgb_input, rectangle_tmp2_pt1, rectangle_tmp2_pt2, cv::Scalar(  0,   0,   0),2/*thickness*/);
						 cv::rectangle(rgb_input, rectangle_tmp3_pt1, rectangle_tmp3_pt2, cv::Scalar(  255,  255,   255),2/*thickness*/);
						
					}
				   
			   }

	       }

		   
		   if(visualize_flag>=2)
		   {
	          cv::namedWindow( "center_p_of_1stlines_centers", cv::WINDOW_AUTOSIZE );
	          cv::imshow("center_p_of_1stlines_centers 1st 2nd 3th", rgb_input);
		   }
	    }

	   
	}
}
	
