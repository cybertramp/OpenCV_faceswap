#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;
using namespace cv::face;

void drawPolyline(
	Mat& img,
	const vector<Point2f>& landmarks,
	const int start,
	const int end,
	bool isClosed
);

// main.cpp
void drawLandmarksPoints(Mat& im, vector<Point2f>& landmarks);
void drawLandmarks(Mat& im, vector<Point2f>& landmarks);
vector<Point2f> imgFaceDetection(Mat img, String title, CascadeClassifier faceCascade, Ptr<Facemark> faceMark);
static void calcDelaunayTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& delaunayTri, Mat img);
void warpTriangle(Mat& img1, Mat& img2, vector<Point2f>& t1, vector<Point2f>& t2);
void processFaceSwap(Mat& img1, Mat& img2, Mat& res);

// cameraDetect.cpp
int processCamFaceEyesDetect();
int processCamFacemarkDetect();
