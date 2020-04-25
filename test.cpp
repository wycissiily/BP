#include<cstdio>
#include<cmath>
using namespace std;
const int L=3;
const int COUNT[L]={784,100,10};
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
}bp;
float num[784];
int main(){
    bp.load("233.bp");
    for(int i=0;i<784;i++)scanf("%f",&num[i]);
    bp.calc(num);
    printf("ANSWER:%d\n",bp.answer());
}