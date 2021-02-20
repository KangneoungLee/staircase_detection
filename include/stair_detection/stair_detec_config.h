#ifndef STAIR_DETEC_CONFIG_H_
#define STAIR_DETEC_CONFIG_H_


unsigned char visualize_flag = 2;   /*0 : no visualization  1: final output visualization  2: each step visualization */
unsigned char time_debug_flag = 1; 

float interpol3(float input, float x1, float x2, float x3, float y1, float y2, float y3)
{
	float temp_out;
	
	if(input < x1)
	{
		temp_out = y1;
	}
	else if(input <x2)	
	{
		temp_out = (y2-y1)/(x2-x1)*(input - x1)+y1;	
	}
	else if(input < x3)
	{
		temp_out = (y3-y2)/(x3-x2)*(input - x2)+y2;
	}
	else
	{
		temp_out = y3;
	}
	
	return temp_out;
	
}


#endif
