//pers.cpp grad-2018
#include "procimagem.h"
int main() {
	Mat_<float> src = (Mat_<float>(4,2) <<
	139,45,
	325,34,
	107,296,
	353,295);
	
	Mat_<float> dst = (Mat_<float>(4,2) <<
	107,34,
	355,34,
	107,302,
	355,302);
	
	Mat_<double> m=getPerspectiveTransform(src,dst);
	
	//Corrige a perspectiva
	Mat_<Vec3b> a=imread("calib_result.jpg",1);
	Mat_<Vec3b> b;
	warpPerspective(a,b,m,a.size());
	imwrite("calib1.jpg",b);
	
	
	imshow("Original", a);
	imshow("Corrijida",b);
	waitKey(0);
}