#include<algorithm>
#include<cstdio>
using namespace std;
const int BATCH_SIZE=20;
const int BATCH_COUNT=3000;
const int TRAIN_COUNT=60000;
const int TEST_COUNT=10000;
unsigned char tmp[784*TRAIN_COUNT];
FILE *traini,*trainl,*testi,*testl;
struct Data{
	float image[784];
	unsigned char label;
};
void _read(FILE *fi,FILE *fl,Data* data,int n){
	fread(tmp,sizeof(char),784*n,fi);
	for(int i=0;i<n;i++)for(int j=0;j<784;j++)data[i].image[j]=float(tmp[784*i+j])/256;
	fread(tmp,sizeof(char),n,fl);
	for(int i=0;i<n;i++)data[i].label=tmp[i];
}
struct Batch{Data data[BATCH_SIZE];};
struct Test{
	Data data[TEST_COUNT];
	void init(){_read(testi,testl,data,TEST_COUNT);fclose(testi);fclose(testl);}
};
struct MNIST{
	Data data[TRAIN_COUNT];
	void init(){_read(traini,trainl,data,TRAIN_COUNT);fclose(traini);fclose(trainl);}
	void reinit(){random_shuffle(data,data+TRAIN_COUNT);}
	Batch getbatch(int x){
		Batch r;
		for(int i=0;i<BATCH_SIZE;i++)r.data[i]=data[BATCH_SIZE*x+i];
		return r;
	}
};
void inittrain(){
	traini=fopen("mnist\\train-images.idx3-ubyte","rb");
	fread(tmp,sizeof(char),16,traini);
	trainl=fopen("mnist\\train-labels.idx1-ubyte","rb");
	fread(tmp,sizeof(char),8,trainl);
}
void inittest(){
	testi=fopen("mnist\\t10k-images.idx3-ubyte","rb");
	fread(tmp,sizeof(char),16,testi);
	testl=fopen("mnist\\t10k-labels.idx1-ubyte","rb");
	fread(tmp,sizeof(char),8,testl);
}