#include<cstdio>
#include<cmath>
#include<ctime>
#include "readdata.h"
using namespace std;
MNIST mnist;
const int L=3;
const int COUNT[L]={784,100,10};
const float eta=0.001;
FILE* fl;
Test t;
inline float reLU(float x){if(x>=0)return x;else return 0;}
inline float DreLU(float x){if(x>=0)return 1;else return 0;}
struct node{float a,z,parz,bias,*weight,*parweight;};
struct BP{
	node* net[L];
	float s[10];
	BP(){
		for(int i=0;i<L;i++)net[i]=new node[COUNT[i]];
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)net[i][j].weight=new float[COUNT[i-1]];
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)net[i][j].parweight=new float[COUNT[i-1]];
	}
	void newdata(){
		for(int i=1;i<L;i++){
			for(int j=0;j<COUNT[i];j++){
				for(int k=0;k<COUNT[i-1];k++)net[i][j].weight[k]=rand()*0.1/32767;
				net[i][j].bias=rand()*0.1/32767;
			}
		}
	}
	void save(char* name){
		FILE* f=fopen(name,"wb");
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)fwrite(net[i][j].weight,sizeof(float),COUNT[i-1],f);
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)fwrite(&net[i][j].bias,sizeof(float),1,f);
		fclose(f);
	}
	void load(char* name){
		FILE* f=fopen(name,"rb");
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)fread(net[i][j].weight,sizeof(float),COUNT[i-1],f);
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)fread(&net[i][j].bias,sizeof(float),1,f);
		fclose(f);
	}
	void eval(int l,int k){
		net[l][k].z=net[l][k].bias;
		for(int i=0;i<COUNT[l-1];i++)net[l][k].z+=net[l-1][i].a*net[l][k].weight[i];
		net[l][k].a=reLU(net[l][k].z);
	}
	void calc(float* data){
		for(int i=0;i<COUNT[0];i++)net[0][i].a=data[i];
		for(int i=1;i<L;i++)for(int j=0;j<COUNT[i];j++)eval(i,j);
		float expsum=0;
		for(int i=0;i<10;i++)expsum+=exp(net[L-1][i].a);
		for(int i=0;i<10;i++)s[i]=exp(net[L-1][i].a)/expsum;
	}
	void dodata(Data data){
		calc(data.image);
		for(int i=0;i<10;i++){
			if(i==data.label)net[L-1][i].parz=(s[i]-1)*DreLU(net[L-1][i].z);
			else net[L-1][i].parz=s[i]*DreLU(net[L-1][i].z);
		}
		for(int i=L-1;i>0;i--){
			for(int j=0;j<COUNT[i];j++)for(int k=0;k<COUNT[i-1];k++)net[i][j].parweight[k]+=net[i][j].parz*net[i-1][k].a;
			if(i>1)for(int j=0;j<COUNT[i-1];j++){
				net[i-1][j].parz=0;
				for(int k=0;k<COUNT[i];k++)net[i-1][j].parz+=net[i][k].parz*net[i][k].weight[j];
				net[i-1][j].parz*=DreLU(net[i-1][j].z);
			}
		}
	}
	void edit(){
		for(int i=1;i<L;i++){
			for(int j=0;j<COUNT[i];j++){
				for(int k=0;k<COUNT[i-1];k++){
					net[i][j].weight[k]-=eta*net[i][j].parweight[k]/BATCH_SIZE;
					net[i][j].parweight[k]=0;
				}
				net[i][j].bias-=eta*net[i][j].parz;
			}
		}
	}
	int answer(){
		float maxn=-1;
		int pos=-1;
		for(int i=0;i<10;i++){
			if(s[i]>maxn){
				maxn=s[i];
				pos=i;
			}
		}
		return pos;
	}
	void SGD(){
		mnist.reinit();
		int acc=0;
		for(int i=0;i<BATCH_COUNT;i++){
			Batch bat=mnist.getbatch(i);
			for(int j=0;j<BATCH_SIZE;j++){
				dodata(bat.data[j]);
				if(answer()==bat.data[j].label)acc++;
			}
			edit();
		}
		fprintf(fl,"ACC:%d\n",acc);
		printf("ACC:%d\n",acc);
	}
}bp;
int main(){
	srand(time(NULL));
	int type;
	scanf("%d",&type);
	if(type==0){
		bp.load("233.bp");
		fl=fopen("bp.txt","w");
		inittrain();
		mnist.init();
		printf("Reading completed.\n");
		int cnt;
		scanf("%d",&cnt);
		while(cnt--){
			printf("%d:",cnt);
			bp.SGD();
		}
		bp.save("233.bp");
		fclose(fl);
	}else if(type==1){
		bp.load("233.bp");
		inittest();
		t.init();
		int crt=0;
		for(int i=0;i<TEST_COUNT;i++){
			bp.calc(t.data[i].image);
			crt+=(bp.answer()==t.data[i].label);
		}
		printf("ACC:%f\n",float(crt)/TEST_COUNT);
	}else{
		bp.newdata();
		bp.save("233.bp");
	}
}