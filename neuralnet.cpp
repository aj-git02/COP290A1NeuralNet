#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <cassert>
#include<mkl.h>
#include <pthread.h>
#include <chrono>
#include "openblasmat.h"

using namespace std;

float input_mat[250][1];
float layer1[144][1];
float layer2[144][1];
float layer3[144][1];
float layer4[144][1];
float layer5[144][1];
float layer6[144][1];
float layer7[12][1];
float output_mat[12][1];
float weight1[250][144];
float weight2[144][144];
float weight3[144][144];
float weight4[144][12];
float bias1[144][1];
float bias2[144][1];
float bias3[144][1];
float bias4[12][1];


void DNN(){
    matMulopenblas(input_mat,weight1,bias1,layer1);
    reLU(layer1,layer2);
    matMulopenblas(layer2,weight2,bias2,layer3);
    reLU(layer3,layer4);
    matMulopenblas(layer4,weight3,bias3,layer5);
    reLU(layer5,layer6);
    matMulopenblas(layer6,weight4,bias4,layer7);
    sigmoid(layer7,output_mat);
}