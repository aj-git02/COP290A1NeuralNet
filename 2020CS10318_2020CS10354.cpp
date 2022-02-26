#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <cassert>
#include<mkl.h>

using namespace std;

// a is input
// b is weight
// c is bias
// d is output
float a[1000][1000];
float b[1000][1000];
float c[1000][1000];
float d[1000][1000];
int x,y,z;

// void matMULmkl(){
//     float *A,*B,*C;
//     //using mkl_malloc to allocate space for pointers
//     A = (float *)mkl_malloc( x*y*sizeof( float ), 64 );
//     B = (float *)mkl_malloc( y*z*sizeof( float ), 64 );
//     C = (float *)mkl_malloc( x*z*sizeof( float ), 64 );
//     if (A == NULL || B == NULL || C == NULL) {
//         cout << "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n";
//         mkl_free(A);
//         mkl_free(B);
//         mkl_free(C);
//         return;
//     }
//     // creating matrices
//     int f = 0;
//     for (int i=0;i<x;i++){
//         for (int j=0;j<y;j++,f++){
//             A[f] = a[i][j];
//         }
//     }
//     f= 0;
//     for (int i=0;i<y;i++){
//         for (int j=0;j<z;j++,f++){
//             B[f] = b[i][j];
//         }
//     }
//     f = 0;
//     for (int i=0;i<x;i++){
//         for (int j=0;j<z;j++,f++){
//             C[f] = c[i][j];
//         }
//     }
// 	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,x,z,y,1,A,y,B,z,1,C,z);

//     for (int i=0;i<z;i++){
//         for (int j=0;j<x;j++){
//             d[j][i]=C[i+j*z];
//         }
//     }
//     //deallocating the space taken by the pointers
//     mkl_free(A);
//     mkl_free(B);
//     mkl_free(C);
//     return;
// }

void matMULmkl(){
    // float *A,*B,*C;
    // //using mkl_malloc to allocate space for pointers
    // A = (float *)mkl_malloc( x*y*sizeof( float ), 64 );
    // B = (float *)mkl_malloc( y*z*sizeof( float ), 64 );
    // C = (float *)mkl_malloc( x*z*sizeof( float ), 64 );
    // if (A == NULL || B == NULL || C == NULL) {
    //     cout << "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n";
    //     mkl_free(A);
    //     mkl_free(B);
    //     mkl_free(C);
    //     return;
    // }
    // creating matrices
    // int f = 0;
    // for (int i=0;i<x;i++){
    //     for (int j=0;j<y;j++,f++){
    //         A[f] = a[i][j];
    //     }
    // }
    // f= 0;
    // for (int i=0;i<y;i++){
    //     for (int j=0;j<z;j++,f++){
    //         B[f] = b[i][j];
    //     }
    // }
    // f = 0;
    // for (int i=0;i<x;i++){
    //     for (int j=0;j<z;j++,f++){
    //         C[f] = c[i][j];
    //     }
    // }
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,x,z,y,1,a,y,b,z,1,c,z);

    for (int i=0;i<z;i++){
        for (int j=0;j<x;j++){
            d[j][i]=c[i+j*z];
        }
    }
    //deallocating the space taken by the pointers
    // mkl_free(A);
    // mkl_free(B);
    // mkl_free(C);
    return;
}

bool check_if_int(string str) {
   for (int i = 0; i < str.length(); i++){
        if (isdigit(str[i]) == false) return false;
   }
   return true;
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

int main(int argc,char *argv[]){

    //assert(argc>=2);
    if ( string(argv[1])=="fullyconnected"){
        if(argc != 6 && argc != 7){ // when this command is to be called we require 6 components including the ./yourcode.out
    		cerr<<"Error : Not enough inputs declared in the command\n";
    		return 0;
    	}
        // passing matrices through fully connected layer

        //assuming for now that arg[6] contains the type of mul to do
        

        //reading input matrix
        string file=string(argv[2]);
        ifstream source;
        source.open(file);
        if (!source.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        source>>y>>x;
        for (int i=0;i<y;i++){
            for (int j=0;j<x;j++){
                source>>a[j][i];
            }
        }
        source.close();

        //reading weight matrix
        source.open(string(argv[3]));
        if (!source.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        int first_dim;
        source>>z>>first_dim;
        assert (first_dim==y);
        for (int i=0;i<z;i++){
            for (int j=0;j<y;j++){
                source>>b[j][i];
            }
        }
        source.close();

        //reading bias matrix
        source.open(string(argv[4]));
        if (!source.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        int second_dim;
        source>>second_dim>>first_dim;
        assert(first_dim==x);
        assert(second_dim==z);
        for (int i=0;i<z;i++){
            for (int j=0;j<x;j++){
                source>>c[j][i];
            }
        }
        source.close();

        //asserting that mat multiplication and addition will be valid
        // assert(input_matrix[0].size()==weight_matrix.size());
        // assert(output_matrix.size()==bias_matrix.size());
        // assert(output_matrix[0].size()==bias_matrix[0].size());

        if (argc == 7 && (string) argv[6]=="mkl"){
            matMULmkl();
        }
        else if (argc == 7 && (string) argv[6]=="openblas"){
            matMULopenblas();
        }
        else if (argc == 7 && (string) argv[6]=="pthread"){
            matMULpthread();
        }
        else if (argc == 6){
            matMUL();
        }
        else {
            cerr<<"Invalid command line argument for mode of implementation"<<endl;
            return 0;
        }
        file=string(argv[5]);
        ofstream destination;
        destination.open(file);
        if (!destination.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        destination<<z<<"\n";
        destination<<x<<"\n";
        for (int i=0;i<z;i++){
            for (int j=0;j<x;j++){
                destination<<d[j][i];
                destination<<"\n";
            }
        }
        destination.close();
    }
    else{ // if any other argument is given, a help block is displayed to the user
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
