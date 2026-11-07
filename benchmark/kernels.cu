


struct V { float x,y,z; };
__device__ V v(float x,float y,float z) { V a={x,y,z}; return a; }
__device__ V add(V a,V b) { return v(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ V sub(V a,V b) { return v(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ V mul(V a,float b) { return v(a.x*b,a.y*b,a.z*b); }
__device__ float dot(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ V norm(V a) { return mul(a,rsqrtf(dot(a,a))); }
__device__ unsigned mix(unsigned x) { x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16); }
__device__ float unit(unsigned x) { return (mix(x)&65535u)/65535.0f; }
__device__ V center(int j,unsigned s,unsigned t) {
  float x=(j%3-1)*0.82f + (unit(s+j*17u)-0.5f)*0.28f;
  float y=(j/3-0.5f)*0.70f + (unit(t+j*31u)-0.5f)*0.20f;
  return v(x,y,0.35f+0.35f*unit(s+t+j*71u));
}
__device__ float sphere(V ro,V rd,V ce,float rad) {
  V oc=sub(ro,ce); float b=dot(oc,rd), c=dot(oc,oc)-rad*rad;
  float d=b*b-c; if(d<=0.0f) return 1e20f;
  float t=-b-sqrtf(d); return t>0.002f?t:1e20f;
}
__device__ V shade(float px,float py,unsigned s,unsigned t,unsigned g) {
  V ro=v((unit(g+79u)-0.5f)*0.18f,0.0f,-3.2f);
  V rd=norm(v(px,py,2.0f)); float best=1e20f; int obj=-1;
  for(int j=0;j<6;j++) { float h=sphere(ro,rd,center(j,s,t),0.30f); if(h<best){best=h;obj=j;} }
  if(obj<0) return v(0.07f+0.05f*(py+1.0f),0.09f+0.03f*(px+1.0f),0.15f);
  V p=add(ro,mul(rd,best)), n=norm(sub(p,center(obj,s,t)));
  V color=v(0.2f+0.7f*unit(s+obj*9u),0.2f+0.7f*unit(t+obj*33u),0.2f+0.7f*unit(s+t+obj*15u));
  float light=0.12f;
  for(int k=0;k<2;k++) {
    V lp=v(k?2.0f:-2.0f,2.1f,-1.8f+0.3f*unit(g+k));
    V delta=sub(lp,p); float distance=sqrtf(dot(delta,delta)); V ld=mul(delta,1.0f/distance);
    bool blocked=false; V origin=add(p,mul(n,0.005f));
    for(int j=0;j<6;j++) if(j!=obj && sphere(origin,ld,center(j,s,t),0.30f)<distance) blocked=true;
    if(!blocked) light+=0.46f*fmaxf(0.0f,dot(n,ld));
  }
  return mul(color,light);
}
extern "C" __global__ void validate(const unsigned* current,const unsigned* previous,unsigned* changed,int groups) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j<groups) changed[j]=(current[j]!=previous[j]);
}
extern "C" __global__ void render(unsigned* image,const unsigned* state,const unsigned* changed,
                                  const unsigned* mask,int width,int height,int mode) {
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=width || y>=height) return;
  int tile=(y/48)*((width+47)/48)+x/48;
  int a=1+tile%64, b=1+(tile*13+7)%64;
  if(mode==1 && !changed[0] && !changed[a] && !changed[b]) return;
  if(mode==2 && !mask[tile]) return;
  if(mode==3) return;
  V rgb=v(0,0,0);
  for(int sy=0;sy<2;sy++) for(int sx=0;sx<2;sx++) {
    float px=2.0f*((x%48)+(sx+0.5f)*0.5f)/48.0f-1.0f;
    float py=1.0f-2.0f*((y%48)+(sy+0.5f)*0.5f)/48.0f;
    rgb=add(rgb,shade(px,py,state[a],state[b],state[0]));
  }
  rgb=mul(rgb,0.25f);
  unsigned r=(unsigned)(255.0f*fminf(1.0f,fmaxf(0.0f,rgb.x)));
  unsigned g=(unsigned)(255.0f*fminf(1.0f,fmaxf(0.0f,rgb.y)));
  unsigned b8=(unsigned)(255.0f*fminf(1.0f,fmaxf(0.0f,rgb.z)));
  image[y*width+x]=r|(g<<8)|(b8<<16)|0xff000000u;
}


extern "C" __global__ void render_compact(unsigned* image, const unsigned* state,
                                          const unsigned* active, int width, int height) {
  int tiles_x=(width+47)/48;
  int tile=active[blockIdx.z];
  int x=(tile%tiles_x)*48+blockIdx.x*blockDim.x+threadIdx.x;
  int y=(tile/tiles_x)*48+blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=width || y>=height) return;
  int a=1+tile%64, b=1+(tile*13+7)%64;
  V rgb=v(0,0,0);
  for(int sy=0;sy<2;sy++) for(int sx=0;sx<2;sx++) {
    float px=2.0f*((x%48)+(sx+0.5f)*0.5f)/48.0f-1.0f;
    float py=1.0f-2.0f*((y%48)+(sy+0.5f)*0.5f)/48.0f;
    rgb=add(rgb,shade(px,py,state[a],state[b],state[0]));
  }
  rgb=mul(rgb,0.25f);
  unsigned r=(unsigned)(255.0f*fminf(1.0f,fmaxf(0.0f,rgb.x)));
  unsigned g=(unsigned)(255.0f*fminf(1.0f,fmaxf(0.0f,rgb.y)));
  unsigned b8=(unsigned)(255.0f*fminf(1.0f,fmaxf(0.0f,rgb.z)));
  image[y*width+x]=r|(g<<8)|(b8<<16)|0xff000000u;
}
