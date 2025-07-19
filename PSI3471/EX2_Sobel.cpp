//exercicio sobel.cpp - 2024
#include "procimagem.h"

int main() {
	Mat_<float> a=imread("circulo.png",0);
	if (a.total()==0) erro("Erro leitura");
	Mat_<float> sx,sy,ox,oy;
	Sobel(a,sx,-1,1,0,3); ox=sx/4.0+128; imwrite("ox.png",ox);
	Sobel(a,sy,-1,0,1,3); oy=sy/4.0+128; imwrite("oy.png",oy);
	
	//modulo
	Mat_<float> tx; pow(sx,2,tx);
	Mat_<float> ty; pow(sy,2,ty);
	Mat_<float> mod; pow(tx+ty,0.5,mod);
	imwrite("modulo.png",mod);
	
	
	Mat_<float> ver_esq, ver_dir, ver_tudo, hor_cima, hor_baixo, hor_tudo;
	
	//ver bordas horizontais
	hor_cima = sy; //gradiente positivo -> claro pra escuro
	imwrite("hor_cima.png",hor_cima);
	hor_baixo = -sy; //gradiente negativo -> escuro para claro 
	imwrite("hor_baixo.png",hor_baixo);
	hor_tudo = abs(sy);
	imwrite("hor_tudo.png",hor_tudo);
	
	//ver bordas verticais 
	ver_esq = sx;
	imwrite("ver_esq.png",ver_esq);
	ver_dir = -sx;
	imwrite("ver_dir.png",ver_dir);
	ver_tudo = abs(sx);
	imwrite("ver_tudo.png",ver_tudo);
	
	return 0;
}