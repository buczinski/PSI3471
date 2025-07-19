// eliminabrpr.cpp
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
    Mat_<uchar> a = imread("mickeyr.bmp", 0);
    int rows = a.rows;
    int cols = a.cols;

    // Cria imagem com padding de 1 pixel
    Mat_<uchar> a_pad(rows + 2, cols + 2);

    // Copia imagem original para o centro da com padding
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            a_pad(i + 1, j + 1) = a(i, j);

    // Replica bordas
    for (int j = 0; j < cols; j++) {
        a_pad(0, j + 1) = a(0, j);
        a_pad(rows + 1, j + 1) = a(rows - 1, j);
    }
    for (int i = 0; i < rows; i++) {
        a_pad(i + 1, 0) = a(i, 0);
        a_pad(i + 1, cols + 1) = a(i, cols - 1);
    }
    a_pad(0, 0) = a(0, 0);
    a_pad(0, cols + 1) = a(0, cols - 1);
    a_pad(rows + 1, 0) = a(rows - 1, 0);
    a_pad(rows + 1, cols + 1) = a(rows - 1, cols - 1);

    // Cria imagem de saída
    Mat_<uchar> b(rows, cols);
	int branco = 0, preto = 0;
    // Aplica filtro com base na imagem com borda
    for (int l = 1; l <= rows; l++) {
        for (int c = 1; c <= cols; c++) {
            for (int dl = -1; dl <= 1; dl++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (a_pad(l + dl, c + dc) == 255)
                        branco++;
                    else
                        preto++;
                }
            }

	if (branco > preto)
    		b(l - 1, c - 1) = 255;
	else
    		b(l - 1, c - 1) = 0;

        }
    }

    imshow("Original", a);
    imshow("Filtrada", b);
	imwrite("mickeyr_filtrada.bmp", b);
    waitKey(0);
    return 0;
}
