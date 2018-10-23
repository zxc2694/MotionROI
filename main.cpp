#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

bool motionDetect(Mat& frame, Mat& motionImg);
void AvoidOutOfRange(const cv::Mat& img, cv::Rect& rect);
vector<Rect> GetComponents(const Mat &motionImg);

int MOTION_FRAME_HALF_INTERVAL = 2;
int IMG_DIFF_THRESH = 10;
int NOISE_MOTION_SIZE = 100;

int main(void)
{
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) {
		return -1;
	}
	Mat source, grayFrame,tmpFrame, motionImg;
	int nframes = 0;
	cap >> source;
	cvtColor(source, tmpFrame, COLOR_BGR2GRAY);

	while (1)
	{
		cap >> source;
		if (source.empty())
			return -1;
		
		cvtColor(source, grayFrame, COLOR_BGR2GRAY);
		motionDetect(grayFrame, motionImg);
		vector<Rect> imgComponVect = GetComponents(motionImg);
		
		// Plot motion ROI
		if (imgComponVect.size() > 0)
		{
			for (int i = 0; i < imgComponVect.size(); i++)
			{
				Rect Rec(imgComponVect[i].x, imgComponVect[i].y, imgComponVect[i].width, imgComponVect[i].height);
				rectangle(grayFrame, Rec, Scalar(255), 1, 8, 0);
			}
		}
		imshow("grayFrame", grayFrame);
		if (!motionImg.empty())
			imshow("motionImg", motionImg);
      
		// Save image output
		char outputPath[100];
		char outputPath2[100];
		sprintf(outputPath, "img//%05d.png", nframes + 1);
		sprintf(outputPath2, "imgM//%05d.png", nframes + 1);
		imwrite(outputPath, grayFrame);
		imwrite(outputPath2, motionImg);
		nframes++;
    
		if (waitKey(30) >= 0) 
			break;
	}
	return 0;
}

bool motionDetect(Mat& frame, Mat& motionImg)
{
	static int motionFrame = 0;
	static Mat startFrame, middleFrame;
	if (startFrame.empty())
		frame.copyTo(startFrame);
	else if (++motionFrame == MOTION_FRAME_HALF_INTERVAL)
	{
		motionFrame = 0;
		if (middleFrame.empty())
			frame.copyTo(middleFrame);
		else
		{
			Mat img1 = frame, img2 = startFrame;
			absdiff(img1, img2, motionImg);
			threshold(motionImg, motionImg, IMG_DIFF_THRESH, 255, THRESH_BINARY);
			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
			morphologyEx(motionImg, motionImg, MORPH_OPEN, element);
			startFrame = middleFrame;
			middleFrame = frame;
			return true;
		}
	}
	return false;
}

vector<Rect> GetComponents(const Mat &motionImg)
{
	cout << "getComponents" << endl;
	vector<Rect> components;
	Mat contoursImg = motionImg.clone();
	vector<vector<cv::Point>> contours;
	findContours(contoursImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	contoursImg.setTo(0);
	copyMakeBorder(contoursImg, contoursImg, 11, 11, 11, 11, BORDER_CONSTANT, Scalar(0));
	for (int i = 0; i < contours.size(); ++i)
	{
		Rect rect = boundingRect(contours[i]);
		if (countNonZero(motionImg(rect)) > NOISE_MOTION_SIZE)
		{
			for (int j = 0; j < contours[i].size(); ++j)
				contours[i][j] += Point(11, 11);
			drawContours(contoursImg, contours, i, Scalar(255), CV_FILLED);
		}
	}
	dilate(contoursImg, contoursImg, getStructuringElement(MORPH_RECT, Size(23, 23)));
	findContours(contoursImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (const auto &contour : contours)
	{
		Rect rect = boundingRect(contour);
		rect -= Size(22, 22);
		AvoidOutOfRange(motionImg, rect);
		components.push_back(rect);
	}
	return components;
}

void AvoidOutOfRange(const cv::Mat& img, cv::Rect& rect)
{
	if (rect.x < 0)
	{
		rect.width += rect.x;
		rect.x = 0;
	}
		if (rect.y < 0)
	{
		rect.height += rect.y;
		rect.y = 0;
	}
	if (rect.x + rect.width > img.cols)
		rect.width = img.cols - rect.x;
	if (rect.y + rect.height > img.rows)
		rect.height = img.rows - rect.y;
}
