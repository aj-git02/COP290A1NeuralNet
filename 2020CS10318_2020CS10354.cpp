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
using namespace std::chrono;

// a is input
// b is weight
// c is bias
// d is output
float a[1000][1000];
float b[1000][1000];
float c[1000][1000];
float d[1000][1000];
int x,y,z;

auto get_time(){
    return std::chrono::high_resolution_clock::now();
}

void* routine(void* args){
    //pthread_detach(pthread_self());
    
    int index=*(int*)args;
    int end_index= index+x/4;
    if (3*(x/4)==index || x < 10) end_index=x;
    for (int k = index; k < end_index;k++)
    {
        for (int i=0;i<z;i++){
            d[k][i]=c[k][i];
            for (int j=0;j<y;j++){
                d[k][i]+=a[k][j]*b[j][i];
            }
        }
    }
    
    
    free(args);

    pthread_exit(NULL);
}
void matMULmkl(){
    float *A,*B,*C;
    //using mkl_malloc to allocate space for pointers
    A = (float *)mkl_malloc( x*y*sizeof( float ), 64 );
    B = (float *)mkl_malloc( y*z*sizeof( float ), 64 );
    C = (float *)mkl_malloc( x*z*sizeof( float ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
        cout << "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n";
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return;
    }
    // creating matrices
    int f = 0;
    for (int i=0;i<x;i++){
        for (int j=0;j<y;j++,f++){
            A[f] = a[i][j];
        }
    }
    f= 0;
    for (int i=0;i<y;i++){
        for (int j=0;j<z;j++,f++){
            B[f] = b[i][j];
        }
    }
    f = 0;
    for (int i=0;i<x;i++){
        for (int j=0;j<z;j++,f++){
            C[f] = c[i][j];
        }
    }
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,x,z,y,1,A,y,B,z,1,C,z);

    for (int i=0;i<z;i++){
        for (int j=0;j<x;j++){
            d[j][i]=C[i+j*z];
        }
    }
    //deallocating the space taken by the pointers
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return;
}

void matMULopenblas(){//function for openblas lib
    float *A,*B,*C;
    A = (float *)malloc( x*y*sizeof( float ));
    B = (float *)malloc( y*z*sizeof( float ));
    C = (float *)malloc( x*z*sizeof( float ));
    if (A == NULL || B == NULL || C == NULL) {
        cout << "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n";
        return;
    }

    int f = 0;
    for (int i=0;i<x;i++){
        for (int j=0;j<y;j++,f++){
            A[f] = a[i][j];
        }
    }
    f= 0;
    for (int i=0;i<y;i++){
        for (int j=0;j<z;j++,f++){
            B[f] = b[i][j];
        }
    }
    f = 0;
    for (int i=0;i<x;i++){
        for (int j=0;j<z;j++,f++){
            C[f] = c[i][j];
        }
    }

    matopenblas(A,B,C,x,y,z);// this function is present in the openblasmat.h file
    for (int i=0;i<z;i++){
        for (int j=0;j<x;j++){
            d[j][i]=C[i+j*z];
        }
    }
    free(A);
    free(B);
    free(C);
    return;
}

void matMULpthread(){// pthread implementation
    pthread_t t[4];

    int *h=(int *)malloc(sizeof(int));
    *h=0;
    pthread_create(t,NULL,&routine,h);
    if(x >= 10){
        int *g=(int *)malloc(sizeof(int));
        *g=x/4;
        pthread_create(t+1,NULL,&routine,g);
        int *r=(int *)malloc(sizeof(int));
        *r=2*(x/4);
        pthread_create(t+2,NULL,&routine,r);
        int *s=(int *)malloc(sizeof(int));
        *s=3*(x/4);
        pthread_create(t+3,NULL,&routine,s);
    }
    int nm = 1;
    if(x >= 10){
        nm = 4;
    }
    for (int i=0;i<nm;i++){
        pthread_join(t[i],NULL);
    }
    return;
}

void matMUL(){ // naive implementation
    for (int f=0;f<x;f++){
        for (int i=0;i<z;i++){
            d[f][i]=c[f][i];
            for (int j=0;j<y;j++){
                d[f][i]+=a[f][j]*b[j][i];
            }
        }
    }
    return;
}
void reset(){ //intialising a matrix
    for (int i=0;i<1000;i++){
        for ( int j=0;j<1000;j++){
            a[i][j]=1;
            b[i][j]=1;
            c[i][j]=1;
            d[i][j]=1;
        }
    }
    return;
}
void time_print(){ // all the data required for plotting is written here
    double arr[10*4];
    double sd[10*4];
    double m[10*4];
    reset();
    x=300;
    z=300;
    y=20;
    int j = 0;
    double mean =0;
    double squared_sum=0;
    ofstream openblas("openblas.txt");
    for (int i=1;i<10;i++){//testing on 9 different sizes
        y=100*i;
        mean = 0;
        squared_sum=0;
        for (int k = 0; k < 100; k++)//running 50 iterations on each size
        {
            auto ob_t1=get_time();
            matMULopenblas();
            auto ob_t2=get_time();
            auto ob_time_span=duration_cast<duration<double>>(ob_t2-ob_t1);
            mean+= ob_time_span.count();
            squared_sum+= (ob_time_span.count()*ob_time_span.count());
            openblas<<ob_time_span.count()<<" "<<y<<"\n";//writing the runtime in the openblas.txt file
        }
        arr[j]=mean;
        mean/=50;
        squared_sum/=50;
        squared_sum-=(mean*mean);
        m[j]=mean;
        sd[j]=sqrt(squared_sum);
        j+=1;
    }
    openblas.close();

    ofstream mkl("mkl.txt");
    for (int i=1;i<10;i++){

        y=100*i;
        mean = 0;
        squared_sum=0;
        for (int k = 0; k < 100; k++)
        {
            auto mkl_t1=get_time();
            matMULmkl();
            auto mkl_t2=get_time();
            auto mkl_time_span=std::chrono::duration_cast<std::chrono::duration<double>>(mkl_t2-mkl_t1);
            mean+=mkl_time_span.count();
            squared_sum+= mkl_time_span.count()*mkl_time_span.count();
            mkl<<mkl_time_span.count()<<" "<<y<<"\n";
        }
        arr[j+1]=mean;
        mean/=50;
        squared_sum/=50;
        squared_sum-=mean*mean;
        m[j+1]=mean;
        sd[j+1]=sqrt(squared_sum/50);
        j+=1;
    }
    mkl.close();

    ofstream pthreads("pthread.txt");
    for (int i=1;i<10;i++){
        y=100*i;
        mean = 0;
        squared_sum=0;
        for (int k = 0; k < 100; k++)
        {
            auto p_t1=get_time();
            matMULpthread();
            auto p_t2=get_time();
            auto p_time_span=duration_cast<duration<double>>(p_t2-p_t1);
            mean+=p_time_span.count();
            squared_sum+= p_time_span.count()*p_time_span.count();
            pthreads<<p_time_span.count()<<" "<<y<<"\n";
        }
        arr[j+2]=mean;
        mean/=50;
        squared_sum/=50;
        squared_sum-=mean*mean;
        m[j+2]=mean;
        sd[j+2]=sqrt(squared_sum/50);
        j+=1;

    }
    pthreads.close();

    ofstream normal("normal.txt");
    for (int i=1;i<10;i++){
        y=100*i;
        mean = 0;
        squared_sum=0;
        for (int k = 0; k < 100; k++)
        {
            auto n_t1=get_time();
            matMUL();
            auto n_t2=get_time();
            auto n_time_span=duration_cast<duration<double>>(n_t2-n_t1);
            mean+=n_time_span.count();
            squared_sum+= n_time_span.count()*n_time_span.count();
            normal<<n_time_span.count()<<" "<<y<<"\n";
        }
        arr[j+3]=mean;
        mean/=50;
        squared_sum/=50;
        squared_sum-=mean*mean;
        m[j+3]=mean;
        sd[j+3]=sqrt(squared_sum/50);
        j+=1;
    }
    normal.close();

        string file="data.txt";
        ofstream destination(file);
        ofstream destination1("data1.txt");
        ofstream destination2("data2.txt");
        ofstream destination3("data3.txt");
        ofstream destination4("data4.txt");
        ofstream standarddev("sd.txt");
        ofstream mean_print("mean.txt");
        if (!destination.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return;
        }

        for (int i=0;i<9;i+=1){
            mean_print<<m[i]<<" "<<"openblas"<<"\n";
            standarddev<<sd[i]<<" "<<"openblas"<<"\n";
            destination<<arr[i]<<" "<<"openblas"<<"\n";
            destination1<<arr[i]<<" "<<"openblas"<<"\n";
            
            mean_print<<m[i+10]<<" "<<"mkl"<<"\n";
            standarddev<<sd[i+10]<<" "<<"mkl"<<"\n";
            destination<<arr[i+10]<<" "<<"mkl"<<"\n";
            destination2<<arr[i+10]<<" "<<"mkl"<<"\n";
            
            mean_print<<m[i+20]<<" "<<"pthread"<<"\n";
            standarddev<<sd[i+20]<<" "<<"pthread"<<"\n";
            destination<<arr[i+20]<<" "<<"pthread"<<"\n";
            destination3<<arr[i+20]<<" "<<"pthread"<<"\n";
            
            mean_print<<m[i+30]<<" "<<"normal"<<"\n";
            standarddev<<sd[i+30]<<" "<<"normal"<<"\n";
            destination<<arr[i+30]<<" "<<"normal"<<"\n";
            destination4<<arr[i+30]<<" "<<"normal"<<"\n";
        }
        mean_print.close();
        standarddev.close();
        destination.close();
        destination1.close();
        destination2.close();
        destination3.close();
        destination4.close();


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

// Old multiplication function
//returns out=M*N+B
// void matMulAdd(vector<vector<float>> &M,vector<vector<float>> &N,vector<vector<float>> &B,vector<vector<float>> &out){
//     int a=M.size();
//     int b=M[0].size();
//     int c=N[0].size();
//     for (int i=0;i<a;i++){
//         for (int j=0;j<c;j++){
//             out[i][j]=B[i][j];
//             for (int k=0;k<b;k++){
//                 out[i][j]+=M[i][k]*N[k][j];
//             }
//         }
//     }
//     return;
// }

int main(int argc,char *argv[]){

    //assert(argc>=2);
    if(string(argv[1])=="timeprint"){
        
        time_print();
        return 0;
    }
    else if ( string(argv[1])=="fullyconnected"){
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


        
        //output sequence to output file
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
// after this we have the old functions defined from the previous part
    else if (string(argv[1])=="activation"){
        //passing matrices through activation layers
        
        //reading input matrix
        assert(argc==5);
        string file=string(argv[3]);
        ifstream source(file);
        if (!source.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        int a,b;
        source>>b>>a;
        vector<vector<float>> input_matrix(a,vector<float> (b,0));
        for (int i=0;i<b;i++){
            for (int j=0;j<a;j++){
                source>>input_matrix[j][i];
            }
        }
        source.close();

        vector<vector<float>> output_matrix(a,vector<float> (b,0));

        //deciding which type of activation
        if ((string) argv[2]=="relu"){
            reLU(input_matrix,output_matrix);
        }
        else if ((string) argv[2]=="tanh"){
            tanh(input_matrix,output_matrix);
        }
        else {
            cerr<<"Command line argument 3 is invalid"<<endl;
            return 0;
        }

        //output sequence to output file
        file=string(argv[4]);
        ofstream destination(file);
        if (!destination.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        destination<<output_matrix[0].size()<<"\n";
        destination<<output_matrix.size()<<"\n";
        for (int i=0;i<b;i++){
            for (int j=0;j<a;j++){
                destination<<output_matrix[j][i];
                destination<<"\n";
            }
        }
        destination.close();
    }

    else if (string(argv[1])=="probability"){

        // converting vectors to probability vectors
        assert(argc==5);

        //reading input vector
        string file=string(argv[3]);
        ifstream source(file);
        if (!source.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        int a;
        source>>a;
        vector<float> input_vector(a);
        for (int i=0;i<a;i++){
            source>>input_vector[i];
        }
        source.close();

        vector<float> output_vector(a);

        //deciding the transformation
        if ((string) argv[2]=="softmax"){
            softmax(input_vector,output_vector);
        }
        else if ((string) argv[2]=="sigmoid"){
            sigmoid(input_vector,output_vector);
        }
        else {
            cerr<<"Command line argument 3 is invalid"<<endl;
            return 0;
        }

        //output sequence to output file
        file=string(argv[4]);
        ofstream destination(file);
        destination<<output_vector.size()<<"\n";
        if (!destination.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        for (int i=0;i<a;i++){
            destination<<output_vector[i];
            destination<<"\n";
        }
        destination.close();
    }
    else if(string(argv[1])=="pooling") {

        //Pooling case
        assert(argc==6);

        //reading input matrix
        string file=string(argv[3]);
        ifstream source(file);
        if (!source.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        int a,b;
        source>>b>>a;
        vector<vector<float>> input_matrix(a,vector<float> (b,0));
        for (int i=0;i<b;i++){
            for (int j=0;j<a;j++){
                source>>input_matrix[j][i];
            }
        }
        source.close();

        // taking stride as input
        assert(check_if_int(string(argv[4])));
        int stride=stoi(string(argv[4]));
        vector<vector<float>> output_matrix(a/stride,vector<float> (b/stride,0));

        assert(a%stride==0);
        assert(b%stride==0);

        // deciding type of pooling
        if (string(argv[2])=="max"){
            maxPool(input_matrix,output_matrix,stride);
        }
        else if (string(argv[2])=="average") {
            avgPool(input_matrix,output_matrix,stride);
        }
        else{
            cerr<<"Command line argument 3 is invalid"<<endl;
            return 0;
        }

        //output sequence to output file
        file=string(argv[5]);
        ofstream destination(file);
        if (!destination.is_open()) { 
            cerr<<"File could not be opened"<<"\n";
            return 0;
        }
        destination<<output_matrix[0].size()<<"\n";
        destination<<output_matrix.size()<<"\n";
        for (int i=0;i<b/stride;i++){
            for (int j=0;j<a/stride;j++){
                destination<<output_matrix[j][i];
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
