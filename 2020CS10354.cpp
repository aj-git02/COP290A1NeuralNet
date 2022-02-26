#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <cassert>
#include<mkl.h>
#include "dnn_weights.h"
#include<algorithm>
using namespace std;

// a is input
// b is weight
// c is bias
// d is output
float a[1000*1000];
// float b[1000*1000];
// float c[1000*1000];
// float d[1000*1000];
int x,y,z;

void matMULmkl(float WT[],float BI[]){
    
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,x,z,y,1,a,y,WT,z,1,BI,z);

    for (int i=0;i<z;i++){
        for (int j=0;j<x;j++){
            a[i+j*z]=BI[i+j*z];
        }
    }
    y=z;
    return;
}

void maxPool(vector<vector<float>> &M,vector<vector<float>> &N,int stride){
    int a=N.size();
    int b=N[0].size();
    for (int i=0;i<a;i++){
        for ( int j=0;j<b;j++){
            float x=0;
            for (int k=0;k<stride;k++){
                for (int w=0;w<stride;w++){
                    x=max(x,M[i*stride+k][j*stride+w]);
                }
            }
            N[i][j]=x;
        }
    }
    return;
}

void avgPool(vector<vector<float>> &M,vector<vector<float>> &N,int stride){
    int a=N.size();
    int b=N[0].size();
    for (int i=0;i<a;i++){
        for ( int j=0;j<b;j++){
            float x=0;
            int number=0;
            for (int k=0;k<stride;k++){
                for (int w=0;w<stride;w++){
                    x+=M[i*stride+k][j*stride+w];
                    number+=1;
                }
            }
            N[i][j]=x/number;
        }
    }
    return;
}

void softmax(float a[]){
    float s=0;
    int n = x*z;
    for (int i=0;i<n;i++){
        s=s+exp(a[i]);
    }
    for (int i=0;i<n;i++){
        a[i]=exp(a[i])/s;
    }
    return;
}

void sigmoid(vector<float> &a,vector<float> &b){
    int n=a.size();
    for (int i=0;i<n;i++){
        b[i]=1/(1+exp(-1*a[i]));
    }
    return;
}

float reLU_single(float x){
    if (x>0) return x;
    else return 0;
}

// returns N=relu(M) (relu applied to each element of M)
void reLU(float M[]){
    int n= x*z;
    for (int i=0;i<n;i++){
        M[i] = reLU_single(M[i]);
        
    }
    return;
}

float tanh_single(float x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

// returns N=tanh(M) (tanh applied to each element of M)
void tanh(vector<vector<float>> &M,vector<vector<float>> &N){
    int n=M.size();
    int m=M[0].size();
    for (int i=0;i<n;i++){
        for ( int j=0;j<m;j++){
            N[i][j]=tanh_single(M[i][j]);
        }
    }
    return;
}
    float BI1[1000*1000]=IP1_BIAS;
    float WT1[1000*1000]=IP1_WT;
    float BI2[1000*1000]=IP2_BIAS;
    float WT2[1000*1000]=IP2_WT;
    float BI3[1000*1000]=IP3_BIAS;
    float WT3[1000*1000]=IP3_WT;
    float BI4[1000*1000]=IP4_BIAS;
    float WT4[1000*1000]=IP4_WT;
 int DNN(){
    x=1;y=250;z=144;
    matMULmkl(WT1,BI1);
    reLU(a);

    x=1;y=144;z=144;
    matMULmkl(WT2,BI2);
    reLU(a);

    x=1;y=144;z=144;
    matMULmkl(WT3,BI3);
    reLU(a);

    x=1;y=144;z=12;
    matMULmkl(WT4,BI4);
    softmax(a);

    return 0;
 }
bool cmp(pair<float,int> a, pair<float,int> b){
    return a > b;
}
int main(int argc,char *argv[]){
    
    if(argc != 3){ // when this command is to be called we require 6 components including the ./yourcode.out
        cerr<<"Error : Not enough inputs declared in the command\n";
        cout<<"To know about the valid command type 'help' or type 'exit' to exit this\n";
     	string a;
     	cin >> a;
     	if(a == "help"){
     		cout<<"Following is a recongnisable command\n";
            cout<<"./yourcode.out audiosamplefile outputfile \n";
        }
        else{
     		cout<<"Exiting.....\n";
     	}
        return 0;
    }
    // passing matrices through fully connected layer


    
    //reading input matrix
    string file=string(argv[1]);
    ifstream source;
    source.open(file);
    if (!source.is_open()) { 
        cerr<<"File could not be opened"<<"\n";
        return 0;
    }
    x=1;
    y=250;
    for (int i=0;i<y;i++){
        //cout<<"hello";
        source>>a[i];
    }


    source.close();

    int zoo = DNN();

    string features[12] = {"silence","unknown","yes","no","up","down","left","right","on","off","stop","go"};
    int first = 0,second = 0,third = 0;

    //asserting that mat multiplication and addition will be valid
    // assert(input_matrix[0].size()==weight_matrix.size());
    // assert(output_matrix.size()==bias_matrix.size());
    // assert(output_matrix[0].size()==bias_matrix[0].size());

    pair<float,int> prob[11];
    for (int i = 0; i < 12; i++)
    {
        pair<float,int> p;
        p.first = a[i];
        p.second = i;
        prob[i] =  p;
    }

    sort(prob,prob + 12,cmp);
    first = prob[0].second;
    second = prob[1].second;
    third = prob[2].second;
    
    string file1=string(argv[2]);
    ofstream destination;
    destination.open(file1,std::ios_base::app);
    if (!destination.is_open()) { 
        cerr<<"File could not be opened"<<"\n";
        return 0;
    }

    destination << file << " " << features[first] << " " << features[second] << " " << features[third] << " ";
    destination << a[first] << " " << a[second] << " " << a[third] << "\n"; 

    destination.close();
    
    // if (false){ // if any other argument is given, a help block is displayed to the user
    // 	cerr<<"Error : The given command is not recongnisable by the program\n";
   
    // }
    return 0;
}