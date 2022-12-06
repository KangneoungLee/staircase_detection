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
