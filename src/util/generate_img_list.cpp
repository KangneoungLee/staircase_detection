#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <string>
#include <io.h> /*this io header is for Windows*/

#include <sys/types.h>/*this io header is for Linux*/
#include <dirent.h>/*this io header is for Linux*/

#include    <sys/stat.h>

 std:: ofstream text_filter_in;
 std::vector<std::string> rugdset_folder;
 
  std::vector<std::string> return_;
 
 bool hh_isdir( const std::string& _path )
 {
     int         i;
    struct stat buf;
    char        *ptr;

    printf("%s: ",_path.c_str());
    if (lstat(_path.c_str(), &buf) < 0) {
            //err_ret("lstat error");
            return false;
        }
    
	if (S_ISDIR(buf.st_mode))  return true;
	
	return false; 

}
 
 
 bool get_files_inDirectory_Linux(const std::string& _path)
{
 	
	DIR *dir;
	
	struct dirent *ent;
   
    dir = opendir(_path.c_str());
   
   
	
	if(dir != NULL)
	{
		while ((ent = readdir(dir))!=NULL)
		{	
			return_.push_back(ent->d_name);	
		}
		closedir(dir);
		return true;
	}
	else
	{
		/*could not open directory*/
		std::cout<<"could not open directory"<<std::endl;
		return false;
	    //     perror("");	
	}

 
    return false;
}
 
 
 


/*std::vector<std::string> get_files_inDirectory_Windows(const std::string& _path)
{
    std::string searching = _path;
 
    std::vector<std::string> return_;
 
    _finddata_t fd;
    long handle = _findfirst(searching.c_str(), &fd);  
 
    if (handle == -1)    return return_;
    
    int result = 0;
    do 
    {
        return_.push_back(fd.name);
        result = _findnext(handle, &fd);
    } while (result != -1);
 
    _findclose(handle);
 
    return return_;
} */


class GEN_IMG_LIST{
	
	private:
	     
		 ros::NodeHandle main_nh;
		 ros::NodeHandle param_nh;
		 
		 ros::Rate* _loop_rate;

		 std::string  _img_folder_dir;
		 //std::string  _annotation_dir;
		 std::string  _txt_save_folder_dir;
		 
		 std::string  _image_dir_prefix;
		 std::string  _annotation_dir_prefix;
		 std::string  _dataset_prefix;
		 
         bool _segmentation_list_extract = false;
			 
		 ros::Time _start_time;
		 
		 
	
	
	public:
		
	    void run();
		void pre_proc_run();
	 
		/*constructor and destructor*/
	    GEN_IMG_LIST(ros::NodeHandle m_nh, ros::NodeHandle p_nh);
	    ~GEN_IMG_LIST();

};

GEN_IMG_LIST::GEN_IMG_LIST(ros::NodeHandle m_nh, ros::NodeHandle p_nh):main_nh(m_nh),param_nh(p_nh)
{
	 std::string  img_folder_dir ="/home/kangneoung/stair_detection/src/stair_detection/image_set/tamu_cs/testing"; 
	 param_nh.getParam("img_folder_dir",img_folder_dir);
	 this->_img_folder_dir = img_folder_dir; 
	 
	 std::string txt_save_folder_dir="/home/kangneoung/stair_detection/src/stair_detection/image_set/tamu_cs/testing"; 
	  param_nh.getParam("txt_save_folder_dir",txt_save_folder_dir);
	  this->_txt_save_folder_dir = txt_save_folder_dir;
	 
	 bool segmentation_list_extract =false;
	 param_nh.getParam("segmentation_list_extract",segmentation_list_extract);
	 this->_segmentation_list_extract = segmentation_list_extract;
	  
	 if( this->_segmentation_list_extract == true)
	 {
	     //std::string  annotation_dir ="/home/kangneoung/RUGD_dataset/RUGD_annotations_gray/training";	 
		 //param_nh.getParam("annotation_dir",annotation_dir);
		 
		 //this->_annotation_dir = annotation_dir; 
		 std::string  image_dir_prefix ="images";
		 param_nh.getParam("image_dir_prefix",image_dir_prefix);
		 this->_image_dir_prefix = image_dir_prefix;
		 
		 std::string  annotation_dir_prefix ="annotations";
		 param_nh.getParam("annotation_dir_prefix",annotation_dir_prefix);
		 this->_annotation_dir_prefix = annotation_dir_prefix;
		 
		 std::string  dataset_prefix ="training";
		 param_nh.getParam("dataset_prefix",dataset_prefix);
		 this->_dataset_prefix = dataset_prefix;
		 
	 }
	 
	 std::string  text_file_name ="testing_rgb_img_file_list.txt";
	 param_nh.getParam("text_file_name",text_file_name);
	  
     int update_rate = 10;

     this->_loop_rate = new ros::Rate(update_rate);	
	
	
	 std::string  text_save_full_dir = this->_txt_save_folder_dir + "/" + text_file_name;

	 try
	 {
		text_filter_in.open(text_save_full_dir,std::ofstream::app); /*text filtering*/

	 }
     catch(int e)
	{
	     ROS_ERROR("check the directory for training data ");
	}
	 
	 
	
	 
}

GEN_IMG_LIST::~GEN_IMG_LIST()
{
    text_filter_in.close();

	delete this->_loop_rate;
}



void GEN_IMG_LIST::pre_proc_run()
{     
     if( this->_segmentation_list_extract == true)
	 {
		 rugdset_folder.push_back("creek");
		 rugdset_folder.push_back("park-1");
		 rugdset_folder.push_back("park-2");
		 rugdset_folder.push_back("park-8");
		 rugdset_folder.push_back("trail");
		 rugdset_folder.push_back("trail-3");
		 rugdset_folder.push_back("trail-4");
		 rugdset_folder.push_back("trail-5");
		 rugdset_folder.push_back("trail-6");
		 rugdset_folder.push_back("trail-7");
		 rugdset_folder.push_back("trail-9");
		 rugdset_folder.push_back("trail-10");
		 rugdset_folder.push_back("trail-11");
		 rugdset_folder.push_back("trail-12");
		 rugdset_folder.push_back("trail-13");
		 rugdset_folder.push_back("trail-14");
		 rugdset_folder.push_back("trail-15");
		 rugdset_folder.push_back("village");
		 
		 std::vector<std::string> files_in_dir;
		 
		 std::vector<std::string>::iterator tr;
		 for (tr = rugdset_folder.begin(); tr != rugdset_folder.end(); tr++)
		 {
			     std::string dir_temp = this->_img_folder_dir +"/" + *tr ;
		         bool flag = get_files_inDirectory_Linux(dir_temp);
		 }
		 
		 files_in_dir=return_;
		 
		 std::vector<std::string>::iterator it;
		 
		 
		 for (it = files_in_dir.begin(); it != files_in_dir.end(); it++)
         {
		    std::istringstream ss(*it);
		    std::string stringBuffer;
		
		    bool img_file_name_pass_flag = false;
		    
			std::cout<<*it<<std::endl;
			
		    while (std::getline(ss, stringBuffer, '.'))
		    {
                if((stringBuffer == "png")||(stringBuffer == "jpg"))
			    {
                   img_file_name_pass_flag = true;
				   break;
			    } 				 

            }
		 
		   if(img_file_name_pass_flag == true)
		   {
			   std::string images_dir =  this->_image_dir_prefix+ "/"+ this->_dataset_prefix+"/"+ *it ;
			   std::string annotations_dir =  this->_annotation_dir_prefix+ "/"+ this->_dataset_prefix+"/"+ *it ;
		       text_filter_in << images_dir <<" "<<annotations_dir<<std::endl;
		   }
	     }
		 
	 }
		 
	 
	 else
	 {
        std::vector<std::string> files_in_dir;
     
		bool flag = get_files_inDirectory_Linux(this->_img_folder_dir);
		 
		files_in_dir=return_;
	 
	    std::vector<std::string>::iterator it;
	 
	    for (it = files_in_dir.begin(); it != files_in_dir.end(); it++)
        {
		    std::istringstream ss(*it);
		    std::string stringBuffer;
		
		    bool img_file_name_pass_flag = false;
		 
		    while (std::getline(ss, stringBuffer, '.'))
		    {
                if((stringBuffer == "png")||(stringBuffer == "jpg"))
			    {
                   img_file_name_pass_flag = true;
				   break;
			    } 				 

            }
		 
		   if(img_file_name_pass_flag == true)
		   {
		       text_filter_in << *it <<std::endl;
		   }
	    }
	 }
}


void GEN_IMG_LIST::run()
{  
    //while(ros::ok())
	//{
	   this->pre_proc_run();
	   
	   //ros::spin();
	   ros::spinOnce();
	   this->_loop_rate->sleep();
	//}
	
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "gen_img_list_util");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");
  
  GEN_IMG_LIST  gen_img_list(nh,_nh);
   
  gen_img_list.run();
 
   return 0;
}
