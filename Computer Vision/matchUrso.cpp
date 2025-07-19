//Pedro Buczisnki Lages de Carvalho - 12555266
//matchUrso.cpp

#include "procimagem.h"

Mat_<Vec3f> marca(Mat_<float> a, Mat_<float> p, float limiar) {
	Mat_<Vec3f> d;
	cvtColor(a,d,COLOR_GRAY2BGR);
	for (int l=0; l<a.rows; l++)
		for (int c=0; c<a.cols; c++)
			if (p(l,c)>=limiar)
				circle(d, Point(c, l), 3, Scalar(0.0, 0.0, 1.0), -1);
	return d;
	
}

int main() {
	
	Mat_<float> a=imread("a.png",0); a = a/255;
	Mat_<float> q=imread("q.png",0); q = q/255;
	
	Mat_<float> p = matchTemplateSame(a, q, TM_CCOEFF_NORMED);
	p = abs(p);
	imwrite("q_ncc.pgm",255*p);
	
	Mat_<Vec3f> m = marca(a,p,0.8);
	imwrite("qr-m.png",255.0*m);
	
	imshow("Ursos", m);
	waitKey(0);
}