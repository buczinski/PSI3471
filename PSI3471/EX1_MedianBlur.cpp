#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){

	Mat a = imread("fever-1.pgm", 0);
	Mat b(a.rows, a.cols,0);

	medianBlur(a,b,5); 
	medianBlur(b,b,5);
	medianBlur(b,b,5);
	

	imshow("Original", a);
	imshow("Borrada", b);
	imwrite("fever-1-filtrada.pgm", b);
    waitKey(0);
    return 0;
	
}