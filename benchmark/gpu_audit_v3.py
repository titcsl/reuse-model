import argparse
import base64
import collections
import csv
import ctypes as C
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics as st
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zlib
VERSION = 'gpu-frame-audit-3.0'
BASE = 825307
GROUPS = 65
TILE = 48
FPS = 60
WIDTH, HEIGHT = (960, 540)
SCENARIOS = ('static', 'sparse_changes', 'frequent_changes', 'camera_cuts', 'all_changed')
POLICIES = ('full_render', 'validate_reuse', 'known_dirty', 'known_compact')
FIELDS = ('record_type', 'utc', 'session', 'trial', 'block', 'phase', 'condition', 'scenario', 'policy', 'seed', 'frame', 't_start_ns', 't_end_ns', 'wall_ms', 'gpu_ms', 'upload_ms', 'validate_ms', 'render_ms', 'copy_ms', 'controller_ms', 'dirty_fraction', 'deadline_miss', 'lateness_ms', 'energy_j', 'power_w', 'metric', 'value', 'details')
PACK = struct.Struct('<65I')
P = C.c_void_p
U = C.c_uint
U64 = C.c_uint64
I = C.c_int
S = C.c_size_t

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def compact(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def ns():
    return time.perf_counter_ns()

def bind(lib, name, args):
    f = getattr(lib, name)
    f.argtypes = args
    f.restype = C.c_int
    return f

class Log:

    def __init__(self, path, append=True, session=''):
        self.path = Path(path)
        self.session = session
        self.f = self.path.open('a' if append else 'x', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.f, fieldnames=FIELDS)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
        self.f.flush()

    def row(self, kind, **kw):
        kw.setdefault('session', self.session)
        if kind not in ('frame', 'telemetry'):
            kw.setdefault('utc', utc())
        if 'details' in kw and (not isinstance(kw['details'], str)):
            kw['details'] = compact(kw['details'])
        self.writer.writerow(dict(record_type=kind, **kw))

    def meta(self, key, value):
        self.row('metadata', metric=key, value=value if isinstance(value, (str, int, float)) else compact(value))
        self.flush()

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

class CUDA:

    def __init__(self, log):
        candidates = ['nvcuda.dll'] if os.name == 'nt' else ['libcuda.so.1', '/usr/lib/wsl/lib/libcuda.so.1']
        for path in candidates:
            try:
                self.lib = C.CDLL(path)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('NVIDIA CUDA driver is unavailable. Use native Linux, working CUDA-enabled WSL, or native Windows with the NVIDIA driver installed. No CPU fallback was run.')
        signatures = {'cuInit': [U], 'cuDriverGetVersion': [C.POINTER(I)], 'cuDeviceGetCount': [C.POINTER(I)], 'cuDeviceGetName': [P, I, I], 'cuDeviceComputeCapability': [C.POINTER(I), C.POINTER(I), I], 'cuDeviceGetPCIBusId': [P, I, I], 'cuDeviceTotalMem_v2': [C.POINTER(S), I], 'cuDevicePrimaryCtxRetain': [C.POINTER(P), I], 'cuCtxSetCurrent': [P], 'cuMemAlloc_v2': [C.POINTER(U64), S], 'cuMemFree_v2': [U64], 'cuMemcpyHtoD_v2': [U64, P, S], 'cuMemcpyDtoH_v2': [P, U64, S], 'cuMemcpyDtoD_v2': [U64, U64, S], 'cuModuleLoadDataEx': [C.POINTER(P), P, U, P, P], 'cuModuleGetFunction': [C.POINTER(P), P, C.c_char_p], 'cuLaunchKernel': [P, U, U, U, U, U, U, U, P, C.POINTER(P), P], 'cuEventCreate': [C.POINTER(P), U], 'cuEventRecord': [P, P], 'cuEventSynchronize': [P], 'cuEventElapsedTime': [C.POINTER(C.c_float), P, P], 'cuEventDestroy_v2': [P], 'cuCtxSynchronize': [], 'cuModuleUnload': [P], 'cuDevicePrimaryCtxRelease_v2': [I]}
        for name, args in signatures.items():
            setattr(self, name, bind(self.lib, name, args))
        self.check(self.cuInit(0))
        n = I()
        self.check(self.cuDeviceGetCount(C.byref(n)))
        if n.value == 0:
            raise RuntimeError('No CUDA GPU is visible; no CPU fallback was run.')
        devices = []
        for dev in range(n.value):
            b = C.create_string_buffer(256)
            self.check(self.cuDeviceGetName(b, 256, dev))
            devices.append(b.value.decode())
        self.dev = int(os.environ.get('GPU_AUDIT_DEVICE', '0'))
        if not 0 <= self.dev < len(devices):
            raise RuntimeError('GPU_AUDIT_DEVICE is not a valid CUDA device index')
        self.name = devices[self.dev]
        self.count = n.value
        v = I()
        self.check(self.cuDriverGetVersion(C.byref(v)))
        self.driver_version = v.value
        if v.value < 12010:
            raise RuntimeError('The installed driver does not support CUDA 12.1. Update the NVIDIA driver and rerun; this script never changes drivers.')
        a = I()
        b = I()
        self.check(self.cuDeviceComputeCapability(C.byref(a), C.byref(b), self.dev))
        self.cc = (a.value, b.value)
        if a.value * 10 + b.value < 52:
            raise RuntimeError('The embedded GPU kernels require compute capability 5.2 or later.')
        bus = C.create_string_buffer(64)
        self.check(self.cuDeviceGetPCIBusId(bus, 64, self.dev))
        self.bus = bus.value
        mem = S()
        self.check(self.cuDeviceTotalMem_v2(C.byref(mem), self.dev))
        self.ctx = P()
        self.check(self.cuDevicePrimaryCtxRetain(C.byref(self.ctx), self.dev))
        self.check(self.cuCtxSetCurrent(self.ctx))
        log.meta('gpu', self.name)
        log.meta('cuda_device_count', n.value)
        log.meta('cuda_driver_api_version', v.value)
        log.meta('compute_capability', self.cc)
        log.meta('gpu_memory_bytes', mem.value)

    def check(self, code):
        if code:
            try:
                name = C.c_char_p()
                self.lib.cuGetErrorName(code, C.byref(name))
                detail = name.value.decode()
            except Exception:
                detail = str(code)
            raise RuntimeError('CUDA error: ' + detail)

    def alloc(self, n):
        p = U64()
        self.check(self.cuMemAlloc_v2(C.byref(p), n))
        return p

    def put(self, p, host):
        self.check(self.cuMemcpyHtoD_v2(p, C.cast(host, P), C.sizeof(host)))

    def copy(self, dst, src, n):
        self.check(self.cuMemcpyDtoD_v2(dst, src, n))

    def launch(self, fn, grid, block, args):
        params = (P * len(args))(*(C.cast(C.byref(a), P) for a in args))
        self.check(self.cuLaunchKernel(fn, *grid, *block, 0, None, params, None))

    def event(self):
        e = P()
        self.check(self.cuEventCreate(C.byref(e), 0))
        return e
CUDA_SOURCE = """


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
"""
KERNEL_BUILD = {'compiler': 'NVIDIA NVRTC', 'version_api': [12, 1], 'distribution_version': '12.1.105', 'compiler_wheel_sha256': '339b385f50c309763ca65456ec75e17bbefcbbf2893f462cb8b90584cd27a1c2', 'target': 'compute_52', 'options': ['--std=c++11', '--gpu-architecture=compute_52', '--fmad=false'], 'source_sha256': 'd821fd67098793d56129b43c748a604c987d3340adf90c1e4a0b8d166d40be99', 'ptx_sha256': '5633a0108bfdf55a58e15abe5dc04bee6e75170b1b58a4816b2b68399cb9963a', 'physical_gpu_test_performed_during_preparation': False}
PTX_ZLIB_BASE64 = (
    'eNq9fVuPHjeS5b5Kv6IetoEZoF2dyXu6vYP2ddHAbD8sFvMwhiCUXSW3AFnSVJU87R70f98veQ4zmeRJlrGQ1phVbX+MjGQy'
    'DoNkRDDiD394/oc/XP3Pu7d39zePd7dXP/x69Zd/+/M3f/7y8uff/tfV1+9+fv/6zd39hWilK//z6qsPr9/cXv35m8+vvv7X'
    'z6wJKU3RZIoPtzdXP2aym8fX795ePb579+bh91f3d2/ubh7urmZzPf/+6t/WP9fz5NdHvrr8fnt1Ic1vjNfT9by+7vn1L3f3'
    'DyuLdPnh+vHm/qe7x6uHn1968/z65vb2/u7h4eXD67/fXQX3/PmzC6Prn968++HNs19u3ry+vXzM8/y/b95cXV9++OnC5ur6'
    'Q3BXL//9Lza+/PNf/s+3//svX/7ry4n/vUwvbz7cvn58+eOHly//ff7l1atX5u0vAa8Nt3ePN6/fLDdv3rz867uHx4dvr/7H'
    '1fzHT/CGy/dZ//KH14/rG8ynekMsb3Cf6A1+Km9In+oNprxhDp/qFba8wn4qSYRtnMKnEkWYt4Eyn0oYYROG8Z9KGnEbKj9/'
    'KnHEHVST+VQCifv8ntynkkjaRstNy6cSSQrbLJ+XTyWTFPeJbtOnEkpa9rkew6eSyrJPd+/tpxDLPK9r1O3dL69/vMMqZecp'
    'TupNP6T/pxfF1w95Dfz2+/Tiwv+/Lsv51P/fPz7iG5fLG/FF5Z3Gu99fVI3//dX8Cd87Tzdvf32J///+Zv//4c3L23fVez8m'
    'THzeYGSc+4/ON2a+8WPz9VPW+tNH52syX/PR+drM135sviGPQ/jo4xDmzHf+6Hzz+IaPPr4xj0P86OMQc3/jx+9vnm/xo8+3'
    'lMchffRxSCHzDR+db9YP6aPrh7RkvsvH5rvk8V0u43s5i75+eP3Dm7ur67u3j/e/XpUz5j89f3b9/ub+5me8qfz8Mv/2cvr9'
    'sHkeN5u62Zq22T7/5+f/daG4v/vp6vr9/eUI/ex377+w//LH8tsPl2ee/e7+i6X66fKWy0+3X8zz5cfLqfnNLd6Q37+2XFa1'
    '79uvuCxAPaHpCWdJaHtCcyREPwVDu9L9/O6XQnJh9bsfL7uc2+u/HRsuu4HfvX3sf78s07/bfr65vX7z7voBLZfvBL/87P1l'
    'bj57uHt8f/3THSneZ4r8z0UhPPvT5YerH+5vrp799399+fKrr6aXl1+fP/vxl8eb68d3xN32zZnn7by+9cOb6/98fVu43t96'
    'sr1sZJ/d3N5eP/CRkB/Bg2tvTjjHTGCah1P+NW4PX8a2PIfXXrh/v9K9UI0RjeFFGYS32yCYlWXmG3Ljm/floVR2QBei8+4u'
    'uUu26e485Z+Xrb8Pj4cufZ9pXqzNl534823EP3/+7P7u8fLLP2p7z/3d29u7+26K4udmguLHk+l5aJxHjWbUaJt5e2h0o0Y/'
    'agxyvs/WVNN7Duv4PnwR9t9eZXG9+iJ4oRpsiEI5GHuiHC7o/P44SHLGh5bMSDLfklmlFuaOm5NkqSXzkiy2ZBn1J+jNs/82'
    'HHXKvJwoGzOdqSczj/VQfhKML7TNs6a87temYdeGbYvj637tXpfnc34SjO9XK0er+azdVN8cRHNWUvnp+TI9n727Bxxz26re'
    'wOBCB71pfa0455frhgWK8a+vt3558oyzTyamtKqMh7/eb5+U1eNKZWe2lCfj1lK0DBuyUmRzaNoWfNxFDcc/tn2xWRoryUln'
    'LBaPqe+MNVvL8YUYUjbP/SsdR/zsjXl4VqrujWFrad6YP5zNvgMC+gMg5IFaj20Hxokr5PbG0pc8dmtzO6pu2hZVu6xNb2+z'
    'psltecxWis9ChtyHH7bHzL4Wzw3H/OOlfe4+wFk+NV/+X9N3h+dE510eyLW963woz6xUbe/zYK4kfe9T9Vw88sTLErqfZ9Hd'
    'f2yzKL9v1Uczp0k4ThO/zpL2oVgesnwoqrnVLOIQ5foUdzI2NU8NtjL7pqLbyhSEtFuZpdqNJLXhcAt2HMsLMS756ZViYmeX'
    '/hOfXX65/vD29fETtpH7XLzTT3jnLHY5AO5KwXe6SQxQ9/3Yxrh+AGYsHPx3UkPgZ3ZH9ic/uZKU/sy/qT95InnRH1v3x8j+'
    'GPbHCplghq4kpT9GwG77H5+fL6jY3zqiaZ/Nnjro0vd0nF9+2RYjn7BUX9+/Xbc1pHiFtqXjGVzRbi3P4LcVLE/nnifaJOpn'
    'fyZ1bOH5jbN+OJ6JKB0ejnITz138nHdAlZ4J2KFfmuNy1Hchs13b80r+t8taXXRayANbCLrRixncK5GZXZjNPE3+yDvmgV/p'
    '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'
)

def kernel_bytes():
    if digest(CUDA_SOURCE.encode()) != KERNEL_BUILD['source_sha256']:
        raise RuntimeError('CUDA source checksum mismatch')
    p = zlib.decompress(base64.b64decode(PTX_ZLIB_BASE64, validate=True))
    if digest(p) != KERNEL_BUILD['ptx_sha256']:
        raise RuntimeError('PTX checksum mismatch')
    return p

def deps_for(w, h):
    return [(1 + i % 64, 1 + (13 * i + 7) % 64) for i in range(math.ceil(w / 48) * math.ceil(h / 48))]

class Engine:

    def __init__(self, cu, module, w=WIDTH, h=HEIGHT):
        self.cu = cu
        self.w = w
        self.h = h
        self.n = w * h
        self.deps = deps_for(w, h)
        self.tiles = len(self.deps)
        self.host = (U * 65)()
        self.mask = (U * self.tiles)()
        self.active = (U * self.tiles)()
        self.pixels = cu.alloc(self.n * 4)
        self.present = cu.alloc(self.n * 4)
        self.current = cu.alloc(260)
        self.previous = cu.alloc(260)
        self.changed = cu.alloc(260)
        self.maskdev = cu.alloc(self.tiles * 4)
        self.activedev = cu.alloc(self.tiles * 4)
        self.mode = I(0)
        self.render = P()
        self.validate = P()
        self.compact_fn = P()
        for obj, name in [(self.render, b'render'), (self.validate, b'validate'), (self.compact_fn, b'render_compact')]:
            cu.check(cu.cuModuleGetFunction(C.byref(obj), module, name))
        self.args = [self.pixels, self.current, self.changed, self.maskdev, I(w), I(h), self.mode]
        self.vargs = [self.current, self.previous, self.changed, I(65)]
        self.cargs = [self.pixels, self.current, self.activedev, I(w), I(h)]
        self.events = [cu.event() for _ in range(5)]

    def frame(self, state, changed, policy):
        cu = self.cu
        t = ns()
        C.memmove(self.host, state, 260)
        active_count = 0
        if policy == 'known_dirty':
            self.mask[:] = [int(changed[0] or changed[a] or changed[b]) for a, b in self.deps]
        elif policy == 'known_compact':
            active = [i for i, (a, b) in enumerate(self.deps) if changed[0] or changed[a] or changed[b]]
            active_count = len(active)
            for i, v in enumerate(active):
                self.active[i] = v
        elif policy == 'copy_only' and any(changed):
            raise RuntimeError('Copy-only is valid only for static input')
        controller = (ns() - t) / 1000000.0
        cu.check(cu.cuEventRecord(self.events[0], None))
        cu.put(self.current, self.host)
        if policy == 'known_dirty':
            cu.put(self.maskdev, self.mask)
        if policy == 'known_compact' and active_count:
            cu.check(cu.cuMemcpyHtoD_v2(self.activedev, C.cast(self.active, P), active_count * 4))
        cu.check(cu.cuEventRecord(self.events[1], None))
        if policy == 'validate_reuse':
            cu.launch(self.validate, (1, 1, 1), (128, 1, 1), self.vargs)
        cu.check(cu.cuEventRecord(self.events[2], None))
        if policy == 'known_compact':
            if active_count:
                cu.launch(self.compact_fn, (3, 3, active_count), (16, 16, 1), self.cargs)
        elif policy != 'copy_only':
            self.mode.value = {'full_render': 0, 'validate_reuse': 1, 'known_dirty': 2, 'blind': 3}[policy]
            cu.launch(self.render, ((self.w + 15) // 16, (self.h + 15) // 16, 1), (16, 16, 1), self.args)
        cu.check(cu.cuEventRecord(self.events[3], None))
        cu.copy(self.present, self.pixels, self.n * 4)
        cu.copy(self.previous, self.current, 260)
        cu.check(cu.cuEventRecord(self.events[4], None))
        cu.check(cu.cuEventSynchronize(self.events[4]))
        intervals = []
        for a, b in zip(self.events, self.events[1:]):
            val = C.c_float()
            cu.check(cu.cuEventElapsedTime(C.byref(val), a, b))
            intervals.append(val.value)
        total = C.c_float()
        cu.check(cu.cuEventElapsedTime(C.byref(total), self.events[0], self.events[-1]))
        return {'gpu_ms': total.value, 'upload_ms': intervals[0], 'validate_ms': intervals[1], 'render_ms': intervals[2], 'copy_ms': intervals[3], 'controller_ms': controller}

    def read(self):
        host = (U * self.n)()
        self.cu.check(self.cu.cuMemcpyDtoH_v2(C.cast(host, P), self.present, self.n * 4))
        return bytes(host)

    def initialize(self):
        self.frame(PACK.pack(*initial_state()), (True,) * 65, 'full_render')

    def close(self):
        for p in (self.pixels, self.present, self.current, self.previous, self.changed, self.maskdev, self.activedev):
            self.cu.check(self.cu.cuMemFree_v2(p))
        for e in self.events:
            self.cu.check(self.cu.cuEventDestroy_v2(e))

class Util(C.Structure):
    _fields_ = [('gpu', U), ('memory', U)]

class NVMLBase:

    def __init__(self, cu, log):
        paths = ['nvml.dll', 'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll'] if os.name == 'nt' else ['libnvidia-ml.so.1', '/usr/lib/wsl/lib/libnvidia-ml.so.1']
        for p in paths:
            try:
                self.lib = C.CDLL(p)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('NVIDIA NVML library is unavailable')
        if bind(self.lib, 'nvmlInit_v2', [])() != 0:
            raise RuntimeError('NVML initialization failed')
        self.handle = P()
        self.lock = threading.Lock()
        self.funcs = {}
        f = bind(self.lib, 'nvmlDeviceGetHandleByPciBusId_v2', [C.c_char_p, C.POINTER(P)])
        if f(cu.bus, C.byref(self.handle)):
            raise RuntimeError('Cannot match NVML and CUDA by PCI bus')
        for name, args in {'nvmlDeviceGetPowerUsage': [P, C.POINTER(U)], 'nvmlDeviceGetTotalEnergyConsumption': [P, C.POINTER(C.c_ulonglong)], 'nvmlDeviceGetTemperature': [P, U, C.POINTER(U)], 'nvmlDeviceGetClockInfo': [P, U, C.POINTER(U)], 'nvmlDeviceGetPerformanceState': [P, C.POINTER(U)], 'nvmlDeviceGetCurrentClocksThrottleReasons': [P, C.POINTER(C.c_ulonglong)], 'nvmlDeviceGetPowerManagementLimit': [P, C.POINTER(U)], 'nvmlDeviceGetUtilizationRates': [P, C.POINTER(Util)]}.items():
            try:
                self.funcs[name] = bind(self.lib, name, args)
            except AttributeError:
                pass
        for name, fn in [('driver', 'nvmlSystemGetDriverVersion'), ('nvml', 'nvmlSystemGetNVMLVersion')]:
            out = C.create_string_buffer(96)
            try:
                if bind(self.lib, fn, [P, U])(out, 96) == 0:
                    log.meta(name + '_version', out.value.decode())
            except AttributeError:
                pass
        out = C.create_string_buffer(96)
        try:
            if bind(self.lib, 'nvmlDeviceGetVbiosVersion', [P, P, U])(self.handle, out, 96) == 0:
                log.meta('vbios_version', out.value.decode())
        except AttributeError:
            pass
        log.meta('power_limit_mw', self.get('nvmlDeviceGetPowerManagementLimit')['value'])
        log.meta('cuda_nvml_pci_bus_match', cu.bus.decode())

    def get(self, name, *extra, large=False):
        with self.lock:
            out = C.c_ulonglong() if large else U()
            start = ns()
            code = -1
            if name in self.funcs:
                code = self.funcs[name](self.handle, *extra, C.byref(out))
            end = ns()
        return {'value': out.value if code == 0 else None, 'before_ns': start, 'after_ns': end, 'status': code}

    def energy(self):
        return self.get('nvmlDeviceGetTotalEnergyConsumption', large=True)

    def sample(self, continuous_counter=False):
        p = self.get('nvmlDeviceGetPowerUsage')
        r = {'power': p}
        if continuous_counter:
            r['counter'] = self.energy()
        for key, name, args, large in [('temperature_c', 'nvmlDeviceGetTemperature', (0,), False), ('sm_mhz', 'nvmlDeviceGetClockInfo', (1,), False), ('memory_mhz', 'nvmlDeviceGetClockInfo', (2,), False), ('pstate', 'nvmlDeviceGetPerformanceState', (), False), ('throttle_mask', 'nvmlDeviceGetCurrentClocksThrottleReasons', (), True)]:
            r[key] = self.get(name, *args, large=large)['value']
        with self.lock:
            u = Util()
            f = self.funcs.get('nvmlDeviceGetUtilizationRates')
            code = f(self.handle, C.byref(u)) if f else -1
        r['gpu_util_pct'] = u.gpu if code == 0 else None
        r['snapshot_end_ns'] = ns()
        return r

    def close(self):
        self.lib.nvmlShutdown()

def initial_state():
    return [1000 + 7919 * j for j in range(65)]

def next_state(state, scenario, frame, rng, fps=60):
    changed = [False] * 65
    p = {'static': 0.0, 'sparse_changes': 0.02, 'frequent_changes': 0.5, 'camera_cuts': 0.02, 'all_changed': 1.0}[scenario]
    for j in range(1, 65):
        if rng.random() < p:
            state[j] = state[j] + rng.randrange(1, 100000) & 4294967295
            changed[j] = True
    if scenario == 'all_changed' or (scenario == 'camera_cuts' and frame % (2 * fps) == 0):
        state[0] = state[0] + 100003 & 4294967295
        changed[0] = True
    return tuple(changed)

def trace(scenario, seed, frames, warm_frames=300, w=WIDTH, h=HEIGHT):
    state = initial_state()
    deps = deps_for(w, h)
    sets = []
    digests = []
    for count, rseed in [(warm_frames, seed ^ 29615), (frames, seed)]:
        part = []
        h = hashlib.sha256()
        rng = random.Random(rseed)
        for f in range(count):
            changed = next_state(state, scenario, f, rng)
            packed = PACK.pack(*state)
            fraction = sum((changed[0] or changed[a] or changed[b] for a, b in deps)) / len(deps)
            part.append((packed, changed, fraction))
            h.update(packed)
        sets.append(part)
        digests.append(h.hexdigest())
    return (sets[0], sets[1], digests)

def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    k = (len(a) - 1) * q
    i = int(k)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (k - i)

def midpoint(r):
    return (r['before_ns'] + r['after_ns']) // 2

def integrate(samples, a, b):
    points = sorted(((midpoint(x['power']), x['power']['value'] / 1000) for x in samples if x['power']['value'] is not None))
    if len(points) < 2 or points[0][0] > a or points[-1][0] < b or (b <= a):
        return None
    result = 0.0
    for (u, p), (v, q) in zip(points, points[1:]):
        lo = max(a, u)
        hi = min(b, v)
        if hi <= lo or v <= u:
            continue
        pl = p + (q - p) * (lo - u) / (v - u)
        ph = p + (q - p) * (hi - u) / (v - u)
        result += (pl + ph) * 0.5 * (hi - lo) / 1000000000.0
    return result

def sleep_to(deadline_ns):
    remaining = (deadline_ns - ns()) / 1000000000.0
    if remaining > 0:
        time.sleep(remaining)

def wait_seconds(seconds, label=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        left = end - time.monotonic()
        if label:
            print(f'{label}: {math.ceil(left)} seconds remaining', flush=True)
        time.sleep(min(30, max(0, left)))

def ac_power():
    if os.name != 'nt':
        return 'unknown'

    class Status(C.Structure):
        _fields_ = [('ACLineStatus', C.c_ubyte), ('BatteryFlag', C.c_ubyte), ('BatteryLifePercent', C.c_ubyte), ('SystemStatusFlag', C.c_ubyte), ('BatteryLifeTime', U), ('BatteryFullLifeTime', U)]
    s = Status()
    if C.windll.kernel32.GetSystemPowerStatus(C.byref(s)):
        return {0: 'battery', 1: 'ac', 255: 'unknown'}.get(s.ACLineStatus, 'unknown')
    return 'unknown'

def keep_awake(on):
    if os.name == 'nt':
        f = C.windll.kernel32.SetThreadExecutionState
        f.argtypes = [U]
        f.restype = U
        return int(f(2147483651 if on else 2147483648))
    return None

def inventory(log):
    log.meta('python', sys.version)
    log.meta('os', platform.platform())
    log.meta('cpu_logical_count', os.cpu_count())
    log.meta('power_source', ac_power())
    log.meta('python_experiment_mode', 'isolated standard library; input traces precomputed; no CPU/system energy measurement')
    if os.name == 'nt':
        script = '$c=Get-CimInstance Win32_ComputerSystem; $b=Get-CimInstance Win32_BIOS; [ordered]@{manufacturer=$c.Manufacturer;model=$c.Model;ram_bytes=$c.TotalPhysicalMemory;cpu=@(Get-CimInstance Win32_Processor|ForEach-Object {$_.Name});bios=$b.SMBIOSBIOSVersion;power_scheme=(powercfg /getactivescheme | Out-String).Trim();background_process_names=@(Get-Process|Select-Object -ExpandProperty ProcessName -Unique)}|ConvertTo-Json -Compress -Depth 4'
        try:
            r = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True, text=True, timeout=25)
            log.meta('platform_inventory', json.loads(r.stdout) if r.returncode == 0 else {'unavailable': r.stderr[-500:]})
        except Exception as e:
            log.meta('platform_inventory_unavailable', str(e))
    else:
        log.meta('platform_inventory_unavailable', 'Native Windows is the supported launcher; portable payload details may be incomplete on this OS')

def correctness_cases(short=False):
    s = initial_state()
    cases = []

    def add(label, new):
        nonlocal s
        ch = tuple((a != b for a, b in zip(s, new)))
        s = list(new)
        cases.append((label, PACK.pack(*s), ch))
    for i in range(2 if short else 16):
        add('unchanged', s)
    for j in range(65):
        if short and j not in [0, 1, 32, 64]:
            continue
        old = list(s)
        new = list(s)
        new[j] ^= 4294967295
        add(f'isolated-{j}', new)
        add(f'revert-{j}', old)
    rng = random.Random(BASE + 77)
    for f in range(8 if short else 96):
        old = list(s)
        new = list(s)
        next_state(new, SCENARIOS[1 + f // 24 % 4], f, rng)
        s = old
        add('random-sequence', new)
    for i in range(2 if short else 16):
        add('long-retention', s)
    return cases

def pixel_mismatches(a, b):
    if len(a) != len(b):
        raise RuntimeError('Image sizes differ')
    if a == b:
        return 0
    return sum((x != y for x, y in zip(memoryview(a).cast('I'), memoryview(b).cast('I'))))

def correctness(cu, module, log, short=False, dims=None):
    dims = dims or ([(960, 540)] if short else [(960, 540), (961, 541)])
    total = 0
    negative = 0
    for w, h in dims:
        engines = {p: Engine(cu, module, w, h) for p in POLICIES + ('blind',)}
        try:
            for e in engines.values():
                e.initialize()
            cases = correctness_cases(short)
            hh = hashlib.sha256()
            for i, (label, state, ch) in enumerate(cases):
                engines['full_render'].frame(state, ch, 'full_render')
                ref = engines['full_render'].read()
                hh.update(ref)
                for p in POLICIES[1:]:
                    engines[p].frame(state, ch, p)
                    bad = pixel_mismatches(engines[p].read(), ref)
                    total += w * h
                    log.row('correctness', policy=p, frame=i, metric='mismatch_pixels', value=bad, details={'width': w, 'height': h, 'case': label})
                    if bad:
                        raise RuntimeError(f'Exact pixel check failed: {p}, {w}x{h}, {label}, mismatches={bad}')
                if i in (0, 1, 2, 16, 32, len(cases) - 1):
                    engines['blind'].frame(state, ch, 'blind')
                    bad = pixel_mismatches(engines['blind'].read(), ref)
                    negative += bad
                    log.row('negative_control', policy='blind', frame=i, metric='mismatch_pixels', value=bad, details={'case': label, 'width': w, 'height': h})
                    if label == 'unchanged' and i < 2 and bad:
                        raise RuntimeError('Static negative control unexpectedly differs')
                if i % 65 == 0:
                    print(f'Correctness {w}x{h}: {i + 1}/{len(cases)} cases', flush=True)
            log.meta(f'correctness_reference_digest_{w}x{h}', hh.hexdigest())
        finally:
            for e in engines.values():
                e.close()
    if not negative:
        raise RuntimeError('Stale-output negative control did not detect an error')
    log.meta('correctness_summary', {'valid_pixel_comparisons': total, 'valid_mismatches': 0, 'negative_mismatches': negative, 'short_session_recheck': short})
    log.flush()
SESSIONS = 4
POLL_MS = (25, 50, 100, 250, 500, 1000)
MAIN_SCENARIOS = ('static', 'sparse_changes', 'all_changed')
WILLIAMS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
API_SOURCE = 'https://raw.githubusercontent.com/NVIDIA/go-nvml/main/pkg/nvml/nvml.h'

class FieldUnion(C.Union):
    _fields_ = [('dVal', C.c_double), ('uiVal', U), ('ulVal', C.c_ulong), ('ullVal', U64), ('sllVal', C.c_longlong), ('siVal', I), ('usVal', C.c_ushort)]

class FieldValue(C.Structure):
    _fields_ = [('fieldId', U), ('scopeId', U), ('timestamp', C.c_longlong), ('latencyUsec', C.c_longlong), ('valueType', I), ('nvmlReturn', I), ('value', FieldUnion)]

class NVML(NVMLBase):

    def __init__(self, cu, log):
        super().__init__(cu, log)
        try:
            self.field_fn = bind(self.lib, 'nvmlDeviceGetFieldValues', [P, I, C.POINTER(FieldValue)])
        except AttributeError:
            self.field_fn = None
        log.meta('field_query_api_available', self.field_fn is not None)
        log.meta('nvml_field_abi', {'header_version': '13.4.92', 'source': API_SOURCE, 'sizeof_field': C.sizeof(FieldValue), 'field_ids': [185, 186], 'supported_fields_determined_from_return_codes': True})

    def power_fields(self):
        arr = (FieldValue * 2)()
        for x, fid in zip(arr, (185, 186)):
            x.fieldId = fid
            x.scopeId = 0
            x.nvmlReturn = -1
        with self.lock:
            a = ns()
            code = self.field_fn(self.handle, 2, arr) if self.field_fn else -1
            b = ns()
        result = []
        attrs = {0: 'dVal', 1: 'uiVal', 2: 'ulVal', 3: 'ullVal', 4: 'sllVal', 5: 'siVal', 6: 'usVal'}
        for x in arr:
            value = getattr(x.value, attrs[x.valueType]) if code == 0 and x.nvmlReturn == 0 and (x.valueType in attrs) else None
            if isinstance(value, float) and (not math.isfinite(value)):
                value = None
            result.append({'field_id': x.fieldId, 'scope_id': x.scopeId, 'api_status': code, 'field_status': x.nvmlReturn, 'value_type': x.valueType, 'value_mw': value, 'source_timestamp_us': x.timestamp if code == 0 and x.nvmlReturn == 0 else None, 'latency_us': x.latencyUsec if code == 0 and x.nvmlReturn == 0 else None, 'before_ns': a, 'after_ns': b})
        return result

    def sample(self, continuous_counter=False, fields=False):
        r = {'sample_start_ns': ns(), 'sample_wall_time_ns': time.time_ns(), 'power': self.get('nvmlDeviceGetPowerUsage')}
        if continuous_counter:
            r['counter'] = self.energy()
        if fields:
            r['power_fields'] = self.power_fields()
        for key, name, args, large in [('temperature_c', 'nvmlDeviceGetTemperature', (0,), False), ('sm_mhz', 'nvmlDeviceGetClockInfo', (1,), False), ('memory_mhz', 'nvmlDeviceGetClockInfo', (2,), False), ('pstate', 'nvmlDeviceGetPerformanceState', (), False), ('throttle_mask', 'nvmlDeviceGetCurrentClocksThrottleReasons', (), True)]:
            q = self.get(name, *args, large=large)
            r[key] = q['value']
            r[key + '_query'] = q
        with self.lock:
            u = Util()
            f = self.funcs.get('nvmlDeviceGetUtilizationRates')
            a = ns()
            code = f(self.handle, C.byref(u)) if f else -1
            b = ns()
        r['gpu_util_pct'] = u.gpu if code == 0 else None
        r['utilization_query'] = {'status': code, 'before_ns': a, 'after_ns': b, 'gpu_pct': u.gpu if code == 0 else None, 'memory_pct': u.memory if code == 0 else None}
        r['snapshot_end_ns'] = ns()
        return r

class Monitor:

    def __init__(self, nvml, mode, interval_ms=100, seed=BASE, fields=False):
        self.nvml = nvml
        self.mode = mode
        self.interval = interval_ms / 1000
        self.rng = random.Random(seed)
        self.fields = fields
        self.samples = []
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.hot = False

    def delay_ns(self):
        return round(self.interval * self.rng.uniform(0.8, 1.2) * 1000000000.0)

    def take(self, scheduled=None):
        r = self.nvml.sample(self.mode == 'power_counter', fields=self.fields)
        r['scheduled_ns'] = scheduled
        r['schedule_lateness_ms'] = max(0, r['sample_start_ns'] - scheduled) / 1000000.0 if scheduled else None
        self.samples.append(r)
        if r['temperature_c'] is not None and r['temperature_c'] >= 85:
            self.hot = True
        if r['power']['status'] != 0:
            raise RuntimeError('NVML power query failed; raw status retained')
        if 'counter' in r and r['counter']['status'] != 0:
            raise RuntimeError('NVML counter query failed; raw status retained')

    def start(self):
        if self.mode == 'boundary_only':
            return
        self.take(ns())
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        scheduled = self.samples[-1]['sample_start_ns'] + self.delay_ns()
        try:
            while not self.stop.wait(max(0, (scheduled - ns()) / 1000000000.0)):
                self.take(scheduled)
                scheduled += self.delay_ns()
                if scheduled <= ns():
                    scheduled = ns() + self.delay_ns()
        except Exception:
            self.error = traceback.format_exc()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.hot:
            raise RuntimeError('GPU reached the 85 C stop threshold')

    def finish(self):
        if self.thread:
            self.stop.set()
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('Telemetry thread did not stop')
            self.take(ns())
        if self.error:
            raise RuntimeError(self.error)

class Guard:

    def __init__(self):
        self.stop = threading.Event()
        self.events = []
        self.reason = None
        self.thread = None
        self.phase = 'setup'

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        last = ns()
        wall = time.time_ns()
        while not self.stop.wait(1):
            now = ns()
            newwall = time.time_ns()
            gap = (now - last) / 1000000000.0
            wgap = (newwall - wall) / 1000000000.0
            event = {'utc': utc(), 'phase': self.phase, 'monotonic_gap_s': gap, 'wall_gap_s': wgap}
            if gap > 5 or wgap > 5:
                self.reason = 'Watchdog observed a gap exceeding 5 seconds; no automatic resume'
                self.events.append({**event, 'kind': 'long_gap'})
            elif abs(wgap - gap) > 2:
                self.reason = 'Wall-clock discontinuity exceeded 2 seconds'
                self.events.append({**event, 'kind': 'clock_discontinuity'})
            if ac_power() == 'battery':
                self.reason = 'AC power disconnected'
                self.events.append({**event, 'kind': 'AC_disconnected'})
            last = now
            wall = newwall

    def check(self):
        if self.reason:
            raise RuntimeError(self.reason)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)

class ConsoleMode:

    def __init__(self):
        self.handle = None
        self.old = None
        self.status = {'applied': False}

    def enter(self):
        if os.name != 'nt':
            return self.status
        k = C.windll.kernel32
        k.GetStdHandle.argtypes = [U]
        k.GetStdHandle.restype = P
        k.GetConsoleMode.argtypes = [P, C.POINTER(U)]
        k.GetConsoleMode.restype = I
        k.SetConsoleMode.argtypes = [P, U]
        k.SetConsoleMode.restype = I
        self.handle = k.GetStdHandle(U(-10 & 4294967295))
        old = U()
        if self.handle and k.GetConsoleMode(self.handle, C.byref(old)):
            self.old = old.value
            new = (self.old | 128) & ~64
            self.status = {'applied': bool(k.SetConsoleMode(self.handle, new)), 'before': self.old, 'requested': new}
        return self.status

    def restore(self):
        if self.old is not None:
            C.windll.kernel32.SetConsoleMode(self.handle, self.old)

def windows_build(log):
    if os.name != 'nt':
        return
    import winreg
    result = {}
    for key, names in [('SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion', ('CurrentBuild', 'CurrentBuildNumber', 'UBR', 'DisplayVersion', 'EditionID', 'BuildLabEx')), ('SOFTWARE\\Microsoft\\WindowsSelfHost\\Applicability', ('BranchName', 'ContentType', 'Ring'))]:
        values = {}
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as h:
                for name in names:
                    try:
                        values[name] = winreg.QueryValueEx(h, name)[0]
                    except OSError:
                        values[name] = None
        except OSError as e:
            values = {'unavailable': str(e)}
        result[key] = values
    result['channel_interpretation'] = 'Registry values only; missing flight data does not establish release-channel membership'
    log.meta('windows_build_registry', result)

def task(phase, block, scenario, seed, policy, seconds, condition=None, monitor='power', interval_ms=100, fields=False, w=960, h=540, paced=True):
    return dict(phase=phase, block=block, scenario=scenario, seed=seed, policy=policy, seconds=seconds, condition=condition or policy, monitor=monitor, interval_ms=interval_ms, fields=fields, width=w, height=h, paced=paced, poll_seed=BASE + session_seed(seed), jitter_fraction=0.2)

def session_seed(seed):
    return int(seed) * 17 + 11

def make_plan(session, smoke=False):
    rng = random.Random(BASE + 90000 + session)
    seed = BASE + 1009 * session
    blocks = []
    if smoke:
        return [task('smoke', 'smoke', 'sparse_changes', seed, p, 2) for p in POLICIES] + [task('smoke', 'idle-smoke', 'static', seed, 'no_launch', 2), task('smoke', 'counter-smoke', 'static', seed, 'full_render', 2, monitor='power_counter', interval_ms=25), task('smoke', 'fields-smoke', 'static', seed, 'full_render', 2, fields=True)]
    sweep = [task('poll_sweep', 'poll-sweep', 'static', seed, 'full_render', 15, condition=f'{m}_{ms}ms', monitor=m, interval_ms=ms) for ms in POLL_MS for m in ('power', 'power_counter')]
    sweep.append(task('poll_sweep', 'poll-sweep', 'static', seed, 'full_render', 15, condition='boundary_only', monitor='boundary_only'))
    rng.shuffle(sweep)
    blocks.append(sweep)
    controls = [task('controls', 'controls', 'static', seed, p, 20, condition=p + '_' + m, monitor=m) for p in ('no_launch', 'copy_only') for m in ('power', 'power_counter', 'boundary_only')]
    rng.shuffle(controls)
    blocks.append(controls)
    for si, scenario in enumerate(MAIN_SCENARIOS):
        labels = list(POLICIES)
        random.Random(BASE + si).shuffle(labels)
        order = [labels[i] for i in WILLIAMS[session % 4]]
        blocks.append([task('main', 'main-' + scenario, scenario, seed + 9973 * si, p, 20) for p in order])
    labels = ['A', 'B'] if session % 2 == 0 else ['B', 'A']
    blocks.append([task('aa', 'aa', 'static', seed, 'full_render', 20, condition=x) for x in labels])
    fieldpair = [task('power_fields', 'power-fields', 'static', seed, 'full_render', 20, condition='fields_on' if f else 'fields_off', fields=f) for f in (False, True)]
    if session % 2:
        fieldpair.reverse()
    blocks.append(fieldpair)
    rng.shuffle(blocks)
    blocks.append([task('sustained_load', 'sustained', 'static', seed, 'full_render', 15, condition='1080p_uncapped', w=1920, h=1080, paced=False)])
    return [t for b in blocks for t in b]

def do_frame(engine, state, changed, policy):
    if policy == 'no_launch':
        return {k: None for k in ('gpu_ms', 'upload_ms', 'validate_ms', 'render_ms', 'copy_ms', 'controller_ms')}
    return engine.frame(state, changed, policy)

def run_trial(engine, nvml, log, t, index, guard, smoke=False):
    ctx = {k: t[k] for k in ('block', 'phase', 'condition', 'scenario', 'policy', 'seed')}
    ctx['trial'] = index
    warm_s = 1 if smoke else 5
    idle_s = 0.2 if smoke else 10 if not t['paced'] else 2
    tail_s = max(1.5, t['interval_ms'] / 1000 * 1.3)
    count = round(t['seconds'] * FPS) if t['paced'] else None
    warm, measured, hashes = trace(t['scenario'], t['seed'], count or 1, round(warm_s * FPS), engine.w, engine.h)
    log.row('trial_plan', **ctx, details={**t, 'frames': count, 'warm_seconds': warm_s, 'idle_seconds': idle_s, 'tail_seconds': tail_s, 'state_trace_sha256': hashes[1] if count else None, 'warm_trace_sha256': hashes[0] if count else None})
    log.flush()
    guard.phase = f'trial {index} preparation'
    guard.check()
    wait_seconds(idle_s)
    guard.check()
    ac0 = ac_power()
    pre = nvml.sample()
    log.row('operating_state', **ctx, metric='before_trial', details=pre)
    if ac0 == 'battery':
        raise RuntimeError('Connect AC power')
    if pre['temperature_c'] is not None and pre['temperature_c'] >= 85:
        raise RuntimeError('GPU reached 85 C')
    engine.initialize()
    monitor = Monitor(nvml, t['monitor'], t['interval_ms'], t['poll_seed'], t['fields'])
    records = []
    start = end = e0 = e1 = cpu0 = cpu1 = warm_start = warm_end = None
    err = None
    last_state = PACK.pack(*initial_state())
    gc_was = gc.isenabled()
    state0 = last_state
    try:
        monitor.start()
        warm_start = ns()
        guard.phase = f'trial {index} warmup'
        if t['paced']:
            for i, (state, ch, _) in enumerate(warm):
                guard.check()
                monitor.check()
                do_frame(engine, state, ch, t['policy'])
                sleep_to(warm_start + round((i + 1) * 1000000000.0 / FPS))
        else:
            while ns() < warm_start + round(warm_s * 1000000000.0):
                guard.check()
                monitor.check()
                do_frame(engine, state0, (False,) * 65, t['policy'])
        warm_end = ns()
        gc.disable()
        e0 = nvml.energy()
        if e0['status'] != 0:
            raise RuntimeError('Start energy-counter read failed')
        cpu0 = time.process_time_ns()
        start = ns()
        guard.phase = f'trial {index} measurement'
        i = 0
        while i < count if t['paced'] else ns() < start + round(t['seconds'] * 1000000000.0):
            guard.check()
            monitor.check()
            state, ch, fraction = measured[i] if t['paced'] else (state0, (False,) * 65, 0.0)
            a = ns()
            timing = do_frame(engine, state, ch, t['policy'])
            b = ns()
            deadline = start + round((i + 1) * 1000000000.0 / FPS) if t['paced'] else None
            records.append(dict(frame=i, t_start_ns=a, t_end_ns=b, wall_ms=(b - a) / 1000000.0, **timing, dirty_fraction=fraction, deadline_miss=int(b > deadline) if deadline else None, lateness_ms=max(0, b - deadline) / 1000000.0 if deadline else None))
            last_state = state
            i += 1
            if b - a > 5000000000.0:
                raise RuntimeError('One frame/host step exceeded 5 seconds')
            if deadline:
                sleep_to(deadline)
        end = ns()
        cpu1 = time.process_time_ns()
        e1 = nvml.energy()
        if e1['status'] != 0:
            raise RuntimeError('End energy-counter read failed')
        guard.phase = f'trial {index} tail'
        if t['monitor'] != 'boundary_only':
            wait_seconds(tail_s)
        guard.check()
        monitor.check()
    except BaseException as e:
        err = e
        if start is not None and end is None:
            end = ns()
            cpu1 = time.process_time_ns()
        if e0 is not None and e1 is None:
            try:
                e1 = nvml.energy()
            except Exception:
                pass
    finally:
        if gc_was:
            gc.enable()
        try:
            monitor.finish()
        except BaseException as e:
            if err is None:
                err = e
        guard.phase = f'trial {index} writing'
        for r in records:
            log.row('frame', **ctx, **r)
        for r in monitor.samples:
            p = r['power']
            phase = 'warmup' if start is None or midpoint(p) < start else 'measurement' if end is not None and midpoint(p) <= end else 'tail'
            log.row('telemetry', **ctx, t_start_ns=p['before_ns'], t_end_ns=p['after_ns'], power_w=p['value'] / 1000 if p['value'] is not None else '', details={**r, 'sample_phase': phase})
        for name, e in [('start', e0), ('end', e1)]:
            if e:
                log.row('energy_boundary', **ctx, metric=name, t_start_ns=e['before_ns'], t_end_ns=e['after_ns'], value=e['value'] if e['value'] is not None else '', details={'status': e['status'], 'unit': 'millijoules'})
        log.flush()
    valid = e0 and e1 and (e0['value'] is not None) and (e1['value'] is not None) and (e1['value'] >= e0['value'])
    energy = (e1['value'] - e0['value']) / 1000 if valid else None
    work = integrate(monitor.samples, start, end) if start is not None and end is not None else None
    aligned = integrate(monitor.samples, midpoint(e0), midpoint(e1)) if e0 and e1 else None
    post = nvml.sample()
    ac1 = ac_power()
    flags = []
    if ac1 == 'battery':
        flags.append('AC_disconnected')
    if monitor.hot or (post['temperature_c'] is not None and post['temperature_c'] >= 85):
        flags.append('temperature_stop_threshold')
    if energy is None:
        flags.append('counter_unusable')
    if energy == 0:
        flags.append('zero_counter_delta')
    if t['monitor'] != 'boundary_only' and aligned is None:
        flags.append('power_not_bracketed')
    misses = sum((r['deadline_miss'] or 0 for r in records))
    if misses:
        flags.append('deadline_misses')
    duration = (end - start) / 1000000000.0 if start is not None and end is not None else None
    if duration is not None and duration > t['seconds'] + 2:
        flags.append('window_overrun_gt_2s')
    if t['paced'] and len(records) != count:
        flags.append('incomplete_frame_count')
    stop_flags = {'AC_disconnected', 'temperature_stop_threshold', 'counter_unusable', 'power_not_bracketed', 'window_overrun_gt_2s', 'incomplete_frame_count'}
    if err is None and stop_flags.intersection(flags):
        err = RuntimeError('Trial failed quality checks: ' + ', '.join(flags))
    wall = [r['wall_ms'] for r in records]
    gpu = [r['gpu_ms'] for r in records if r['gpu_ms'] is not None]
    pts = [midpoint(r['power']) for r in monitor.samples if start is not None and end is not None and (start <= midpoint(r['power']) <= end)]
    gaps = [(b - a) / 1000000.0 for a, b in zip(pts, pts[1:])]
    status = 'interrupted' if isinstance(err, KeyboardInterrupt) else 'failed' if err else 'completed'
    log.row('trial', **ctx, t_start_ns=start or '', t_end_ns=end or '', energy_j=energy if energy is not None else '', details={'status': status, 'flags': flags, 'planned_frames': count, 'completed_frames': len(records), 'duration_s': duration, 'width': engine.w, 'height': engine.h, 'paced': t['paced'], 'fps_target': FPS if t['paced'] else None, 'warm_start_ns': warm_start, 'warm_end_ns': warm_end, 'counter_query_midpoint_duration_s': (midpoint(e1) - midpoint(e0)) / 1000000000.0 if e0 and e1 else None, 'power_integral_work_j': work, 'power_integral_counter_midpoints_j': aligned, 'wall_p50_ms': percentile(wall, 0.5), 'wall_p95_ms': percentile(wall, 0.95), 'wall_p99_ms': percentile(wall, 0.99), 'wall_max_ms': max(wall) if wall else None, 'gpu_p50_ms': percentile(gpu, 0.5), 'deadline_misses': misses, 'cpu_process_s': (cpu1 - cpu0) / 1000000000.0 if cpu0 is not None and cpu1 is not None else None, 'state_trace_sha256': hashes[1] if count else None, 'warm_trace_sha256': hashes[0] if count else None, 'monitor': t['monitor'], 'nominal_poll_ms': t['interval_ms'], 'actual_poll_ms_p50': percentile(gaps, 0.5), 'actual_poll_ms_p95': percentile(gaps, 0.95), 'actual_poll_ms_max': max(gaps) if gaps else None, 'fields_enabled': t['fields'], 'ac_start': ac0, 'ac_end': ac1, 'error': str(err) if err else None, 'no_launch_has_no_CUDA_calls_in_window': t['policy'] == 'no_launch', 'gc_disabled_during_measurement': True})
    log.row('operating_state', **ctx, metric='after_trial', details=post)
    log.flush()
    if err:
        raise err
    actual = engine.read()
    engine.frame(last_state, (True,) * 65, 'full_render')
    reference = engine.read()
    bad = pixel_mismatches(actual, reference)
    log.row('final_image_check', **ctx, metric='mismatch_pixels', value=bad, details={'pixels': engine.n, 'outside_energy_window': True, 'reference_sha256': digest(reference)})
    log.flush()
    if bad:
        raise RuntimeError(f'Trial {index} final image differs in {bad} pixels')
    guard.check()

def protocol(smoke=False):
    plans = [make_plan(s, smoke) for s in range(1 if smoke else SESSIONS)]
    trials = [t for p in plans for t in p]
    nominal = sum((t['seconds'] + (1 if smoke else 5) + (0.2 if smoke else 10 if not t['paced'] else 2) + (max(1.5, t['interval_ms'] / 1000 * 1.3) if t['monitor'] != 'boundary_only' else 0) for t in trials))
    nominal += (len(plans) - 1) * 60 + (0 if smoke else len(plans) * 20)
    return {'version': VERSION, 'sessions': len(plans), 'trials_per_session': len(plans[0]), 'planned_trials': len(trials), 'phase_counts': dict(collections.Counter((t['phase'] for t in trials))), 'measurement_and_scheduled_wait_minutes': nominal / 60, 'runtime_note': 'Allow about 75-95 minutes for full run, including setup, correctness, file writing and slower hardware; this is an estimate, not a timeout', 'plan': plans, 'fps': 60, 'primary_dimensions': [960, 540], 'sustained_diagnostic_dimensions': [1920, 1080], 'primary_scenarios': list(MAIN_SCENARIOS), 'polling_intervals_ms': list(POLL_MS), 'polling_jitter_fraction': 0.2, 'polling': 'Start-to-start randomized U(0.8,1.2) times nominal interval; no catch-up bursts; achieved intervals logged', 'power_fields': '185 average and 186 instantaneous requested only in dedicated fields_on trials; raw API and per-field status retained; never substituted into the primary estimator', 'counter_windows': 'All modes read cumulative counter immediately before and after the window; only power_counter polls it inside the window', 'idle_control': 'Context and allocations resident. No CUDA launches, copies, events or synchronization during the no_launch warmup/measurement; 60-Hz host cadence and common watchdog remain', 'copy_control': 'Retains state uploads, CUDA event calls, output and previous-state device copies; not a pure memcpy-only kernel', 'sustained_control': 'Uncapped static full rendering, fixed elapsed duration. Diagnostic only, excluded from 60-Hz policy savings; P8 departure is not guaranteed', 'order': 'Random block order; four-period Williams ordering balances main policy position and first-order predecessors across four sessions. Fresh scenario seeds each session', 'inference_unit': 'Session/seed block, not individual frame or shared-baseline contrast. Sessions are one sitting, not independent days or devices', 'analysis': 'Retain every trial. Matched power-integral/counter differences at common counter-query endpoints. Report absolute J, W, J/frame, individual pairs, block spread and all flags. No assumed ground truth', 'stopping': 'Fixed plan; no stopping for desired results. Stop on API/correctness failure, AC loss, temperature >=85 C, watchdog gap >5 s, clock discontinuity >2 s, unbracketed primary power or window overrun >2 s', 'limitations': 'No external meter; no formal statistical power calculation; four within-sitting blocks cannot establish general energy equivalence or physical accuracy', 'software_changes': 'No driver/clock/power-limit/persistent power-plan changes. Temporary idle-sleep prevention and classic-console QuickEdit disable restored on exit', 'smoke_only': smoke, 'source_sha256': digest(Path(__file__).read_bytes()), 'kernel_build': KERNEL_BUILD, 'prior_v2_source_sha256': '393f60171d0ab6f640a402d9a188ff099d9f6791c4caa6e796c47fec1836f6ca'}

def self_test():
    assert C.sizeof(FieldValue) == 40 and FieldValue.value.offset == 32 and (C.sizeof(FieldUnion) == 8)
    ptx = kernel_bytes()
    assert b'.entry render_compact' in ptx and b'.entry validate' in ptx
    assert len(deps_for(960, 540)) == 240 and len(deps_for(1920, 1080)) == 920
    groups = collections.defaultdict(list)
    pred = collections.defaultdict(collections.Counter)
    for s in range(SESSIONS):
        plan = make_plan(s)
        assert len(plan) == 36 and plan[-1]['phase'] == 'sustained_load'
        for sc in MAIN_SCENARIOS:
            block = [t for t in plan if t['phase'] == 'main' and t['scenario'] == sc]
            assert len(block) == 4 and len({t['seed'] for t in block}) == 1
            for i, t in enumerate(block):
                groups[sc, t['policy']].append(i)
            for x, y in zip(block, block[1:]):
                pred[sc][x['policy'], y['policy']] += 1
        assert len([t for t in plan if t['phase'] == 'poll_sweep']) == 13
    assert all((sorted(v) == [0, 1, 2, 3] for v in groups.values()))
    assert all((len(v) == 12 and set(v.values()) == {1} for v in pred.values()))

    class NoGPU:

        def frame(self, *a):
            raise AssertionError('No-launch invoked GPU')
    assert do_frame(NoGPU(), b'', (), 'no_launch')['gpu_ms'] is None
    for sc in MAIN_SCENARIOS:
        assert trace(sc, BASE, 31, 7) == trace(sc, BASE, 31, 7)
        assert all((0 <= f <= 1 for part in trace(sc, BASE, 31, 7)[:2] for _, _, f in part))
    sample = lambda t, p: {'power': {'value': p * 1000, 'before_ns': int(t * 1000000000.0), 'after_ns': int(t * 1000000000.0)}}
    assert math.isclose(integrate([sample(0, 2), sample(1, 4), sample(2, 6)], 250000000, 1750000000), 6.0)
    assert integrate([sample(0, 2), sample(1, 4)], -1, 100) is None
    assert pixel_mismatches(struct.pack('<2I', 1, 2), struct.pack('<2I', 1, 3)) == 1
    print('Host checks PASS. This is not a GPU execution test.', flush=True)

def session_run(args):
    log = Log(args.csv, session=args.session)
    cu = nvml = None
    engines = {}
    module = P()
    code = 0
    guard = Guard()
    console = ConsoleMode()
    log.row('session_status', metric='started', value=args.session)
    try:
        log.meta('console_mode', console.enter())
        log.meta('keep_awake_previous_state', keep_awake(True))
        inventory(log)
        windows_build(log)
        if sys.byteorder != 'little' or C.sizeof(P) != 8:
            raise RuntimeError('Requires 64-bit little-endian Python')
        if ac_power() == 'battery':
            raise RuntimeError('Connect AC power before starting')
        cu = CUDA(log)
        nvml = NVML(cu, log)
        if nvml.energy()['value'] is None or nvml.sample()['power']['value'] is None:
            raise RuntimeError('Both NVML cumulative energy and power must be supported')
        ptx = C.create_string_buffer(kernel_bytes())
        cu.check(cu.cuModuleLoadDataEx(C.byref(module), C.cast(ptx, P), 0, None, None))
        print(f'GPU: {cu.name} | session {args.session + 1}/{(1 if args.smoke else SESSIONS)}', flush=True)
        correctness(cu, module, log, short=args.session > 0 or args.smoke)
        if args.session == 0 and (not args.smoke):
            correctness(cu, module, log, short=True, dims=[(1920, 1080)])
        for w, h in [(960, 540)] if args.smoke else [(960, 540), (1920, 1080)]:
            engines[w, h] = Engine(cu, module, w, h)
        e = engines[960, 540]
        e.initialize()
        state = PACK.pack(*initial_state())
        cal = []
        clock = ns()
        for i in range(120):
            a = ns()
            e.frame(state, (True,) * 65, 'full_render')
            b = ns()
            if i >= 60:
                cal.append((b - a) / 1000000.0)
            sleep_to(clock + round((i + 1) * 1000000000.0 / FPS))
        log.meta('cadence_preflight', {'samples': len(cal), 'p95_ms': percentile(cal, 0.95), 'criterion_ms': 0.8 * 1000 / FPS, 'paced': True})
        if percentile(cal, 0.95) > 0.8 * 1000 / FPS:
            raise RuntimeError('Insufficient 60-Hz headroom; no adaptive FPS substitution')
        plan = make_plan(args.session, args.smoke)
        log.meta('session_plan', plan)
        log.flush()
        guard.start()
        if not args.smoke:
            wait_seconds(20, 'Pre-session idle')
        for i, t in enumerate(plan, 1):
            guard.phase = f'before trial {i} progress output'
            guard.check()
            stamp = ns()
            wall = time.time_ns()
            log.row('progress_checkpoint', trial=i, metric='before_console', t_start_ns=stamp, value=wall)
            log.flush()
            print(f"Session {args.session + 1}: {i}/{len(plan)} | {t['phase']} | {t['scenario']} | {t['condition']} | {t['seconds']} s", flush=True)
            log.row('progress_checkpoint', trial=i, metric='after_console', t_start_ns=ns(), value=time.time_ns())
            if ns() - stamp > 5000000000.0:
                raise RuntimeError('Progress output blocked for more than 5 seconds')
            guard.check()
            run_trial(engines[t['width'], t['height']], nvml, log, t, i, guard, args.smoke)
        log.row('session_status', metric='completed', value=len(plan))
        log.flush()
    except BaseException as e:
        code = 130 if isinstance(e, KeyboardInterrupt) else 1
        log.row('session_status', metric='interrupted' if code == 130 else 'failed', value=str(e), details={'traceback': traceback.format_exc()})
        log.flush()
        print('Stopped:', str(e), flush=True)
    finally:
        guard.close()
        log.meta('watchdog_events', guard.events)
        for engine in engines.values():
            try:
                engine.close()
            except Exception as e:
                log.row('cleanup', metric='engine', value=str(e))
        for name, fn in [('module', lambda: cu.check(cu.cuModuleUnload(module)) if cu and module.value else None), ('context', lambda: cu.check(cu.cuDevicePrimaryCtxRelease_v2(cu.dev)) if cu else None), ('nvml', lambda: nvml.close() if nvml else None)]:
            try:
                fn()
            except Exception as e:
                log.row('cleanup', metric=name, value=str(e))
        keep_awake(False)
        console.restore()
        log.close()
    return code

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--session', type=int)
    p.add_argument('--csv')
    p.add_argument('--output-dir', default=str(Path.cwd()))
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.plan:
        print(json.dumps(protocol(args.smoke), indent=2))
        return 0
    if args.session is not None:
        if not args.csv:
            p.error('--csv is required with --session')
        if not 0 <= args.session < SESSIONS:
            p.error('invalid session')
        return session_run(args)
    self_test()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ('gpu_audit_v3_' + ('smoke_' if args.smoke else '') + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6] + '.csv')
    log = Log(path, append=False)
    pr = protocol(args.smoke)
    log.meta('protocol', pr)
    log.meta('protocol_sha256', digest(compact(pr).encode()))
    source = Path(__file__).read_bytes()
    log.meta('payload_source_zlib_base64', base64.b64encode(zlib.compress(source, 9)).decode())
    log.meta('payload_sha256', digest(source))
    log.meta('primary_sources', [API_SOURCE, 'https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html', 'https://learn.microsoft.com/en-us/windows/console/setconsolemode'])
    log.row('run_status', metric='started', value=VERSION)
    log.close()
    print('\nCSV TO SEND BACK:', path, flush=True)
    print('SMOKE TEST ONLY; do not use as research results' if args.smoke else 'Full run: 144 trials in 4 sessions. Allow about 75-95 minutes plus download. Keep AC connected.', flush=True)
    code = 0
    worker = None
    console = ConsoleMode()
    console.enter()
    keep_awake(True)
    try:
        for s in range(pr['sessions']):
            if s:
                stamp = ns()
                wall = time.time_ns()
                transition = Log(path)
                transition.row('session_transition', session=s, metric='pause_start', t_start_ns=stamp, value=wall, details={'expected_seconds': 60})
                transition.close()
                wait_seconds(60, 'Between-session pause')
                elapsed = (ns() - stamp) / 1000000000.0
                transition = Log(path)
                transition.row('session_transition', session=s, metric='pause_end', t_end_ns=ns(), value=time.time_ns(), details={'actual_seconds': elapsed})
                transition.close()
                if elapsed > 65:
                    raise RuntimeError('Between-session pause exceeded its scheduled duration by more than 5 seconds')
            cmd = [sys.executable, '-I', '-u', str(Path(__file__).resolve()), '--session', str(s), '--csv', str(path)]
            if args.smoke:
                cmd.append('--smoke')
            worker = subprocess.Popen(cmd)
            try:
                code = worker.wait()
            except KeyboardInterrupt:
                code = 130
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=5)
            if code:
                break
    except BaseException as e:
        code = 130 if isinstance(e, KeyboardInterrupt) else 1
        errorlog = Log(path)
        errorlog.row('run_error', metric='parent', value=str(e), details={'traceback': traceback.format_exc()})
        errorlog.close()
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
            code = 130
        keep_awake(False)
        console.restore()
        end = Log(path)
        end.row('run_status', metric='completed' if code == 0 else 'incomplete', value=code, details={'smoke_only': args.smoke, 'planned_trials': pr['planned_trials'], 'interpretation': 'No physical-energy accuracy, policy equivalence or publication outcome is inferred'})
        end.close()
    print('\n' + ('COMPLETE' if code == 0 else 'INCOMPLETE - send CSV including the error records') + '\nCSV TO SEND BACK: ' + str(path), flush=True)
    return code
if __name__ == '__main__':
    raise SystemExit(main())
