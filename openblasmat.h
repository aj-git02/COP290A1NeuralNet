#include <iostream>
#include <cblas.h>



using namespace std;
//-lopenblas
void matopenblas(float *a, float *b, float *c,int x,int y,int z){

cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,x,z,y,1,a,y,b,z,1,c,z);

}