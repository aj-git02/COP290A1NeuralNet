#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <cassert>
#include<mkl.h>
#include "dnn_weights.h"

using namespace std;

// a is input
// b is weight
// c is bias
// d is output
float a[1000*1000];
float b[1000*1000];
float c[1000*1000];
float d[1000*1000];
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

void softmax(vector<float> &a,vector<float> &b){
    float x=0;
    int n=a.size();
    for (int i=0;i<n;i++){
        x=x+exp(a[i]);
    }
    for (int i=0;i<n;i++){
        b[i]=exp(a[i])/x;
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
void reLU(vector<vector<float>> &M,vector<vector<float>> &N){
    int n=M.size();
    int m=M[0].size();
    for (int i=0;i<n;i++){
        for ( int j=0;j<m;j++){
            N[i][j]=reLU_single(M[i][j]);
        }
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

void neuralnet(){
    x=1;y=250;z=144;
    float BI1[1000*1000]=IP1_BIAS;
    float WT1[1000*1000]=IP1_WT;
    cout<<"hello";
    matMULmkl(WT1,BI1);
    cout<<"hello";
    x=1;y=144;z=144;
    float BI2[1000*1000]=IP2_BIAS;
    float WT2[1000*1000]=IP2_WT;
    matMULmkl(WT2,BI2);
    return;
}

int main(int argc,char *argv[]){
    //cout<<"hello";
    if(argc != 3){ // when this command is to be called we require 6 components including the ./yourcode.out
        cerr<<"Error : Not enough inputs declared in the command\n";
        return 0;
    }
    // passing matrices through fully connected layer

    //assuming for now that arg[6] contains the type of mul to do
    
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
    //cout<<"hello";
    // source>>y>>x;
    for (int i=0;i<y;i++){
        //cout<<"hello";
        source>>a[i];
        cout<<i<<" ";
    }
    source.close();
    // float arr[24]=IP4_BIAS;
    // for (int g=0;g<12;g++){
    //     cout<<arr[g]<<"\n";
    // }
cout<<"hello";
    neuralnet();

    //reading weight matrix
    

    //reading bias matrix
    

    //asserting that mat multiplication and addition will be valid
    // assert(input_matrix[0].size()==weight_matrix.size());
    // assert(output_matrix.size()==bias_matrix.size());
    // assert(output_matrix[0].size()==bias_matrix[0].size());

    cout<<"hello";
    //matMULmkl();
    
    file=string(argv[2]);
    ofstream destination;
    destination.open(file);
    if (!destination.is_open()) { 
        cerr<<"File could not be opened"<<"\n";
        return 0;
    }
    // destination<<z<<"\n";
    // destination<<x<<"\n";
    for (int i=0;i<y;i++){
        for (int j=0;j<x;j++){
            destination<<a[j*z+i];
            destination<<"\n";
        }
    }
    destination.close();
    
    if (false){ // if any other argument is given, a help block is displayed to the user
    	cerr<<"Error : The given command is not recongnisable by the program\n";
    	cout<<"To know about the valid commands type 'help' or type 'exit' to exit this\n";
    	string a;
    	cin >> a;
    	if(a == "help"){
    		cout<<"Following are the recongnisable commands\n";
            cout<<"./yourcode.out fullyconnected inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt mkl \n";
    		cout<<"./yourcode.out fullyconnected inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt openblas \n";
            cout<<"./yourcode.out fullyconnected inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt pthread \n";
            cout<<"./yourcode.out fullyconnected inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt \n";
    		cout<<"./yourcode.out activation relu inputmatrix.txt outputmatrix.txt\n";
    		cout<<"./yourcode.out activation tanh inputmatrix.txt outputmatrix.txt\n";
    		cout<<"./yourcode.out pooling max inputmatrix.txt stride outputmatrix.txt\n";
    		cout<<"./yourcode.out pooling average inputmatrix.txt stride outputmatrix.txt\n";
    		cout<<"./yourcode.out probability softmax inputvector.txt outputvector.txt\n";
    		cout<<"./yourcode.out probability sigmoid inputvector.txt outputvector.txt\n";
    	}
    	else{
    		cout<<"Exiting.....\n";
    	}
    }
    return 0;
}