"""fp8-resident multi-layer conv3d chain — the amortized speedup measurement (sm120).

The single-layer fused kernel (fp8_conv3d_triton.py) is 2.9-4.1x faster than cuDNN fp16
at the KERNEL level, but a per-layer f32<->fp8 cast+permute round-trip drags the
end-to-end number to ~1.17x. This benchmark closes that gap: activations stay fp8
channels-last BETWEEN layers (the realistic inference pattern), with the output
requantize FUSED into the kernel epilogue (conv_f8f8 stores fp8 directly). Result on the
5060 Ti: 2.85x (C=320) - 4.13x (C=128) over a cuDNN fp16 chain, vs TRT fp16's 1.53x.

The out scale `so` is a constant here; production would calibrate per-layer amax (static
calibration pass, standard fp8 recipe). Usage: python fp8_conv3d_chain.py
"""
import time, statistics
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
E=448.0
dev='cuda'; torch.manual_seed(0)

# fp8-in fp8-out conv kernel: consumes fp8 [M,Cin] channels-last, produces fp8 [M,Cout].
# out scale so = amax_est/E passed in (per-tensor, precomputed/calibrated in real inference).
@triton.autotune(configs=[triton.Config({"BLOCK_M":bm,"BLOCK_N":bn,"BLOCK_K":bk},num_warps=w,num_stages=s)
    for bm in (64,128) for bn in (64,128) for bk in (32,64) for w in (4,8) for s in (3,4)],
    key=["Cin","Cout","D","H","W"])
@triton.jit
def conv_f8f8(x_ptr,w_ptr,y_ptr, sab, inv_so, N,D,H,W,Cin,Cout,
              BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M); offs_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    w_idx=offs_m%W; t=offs_m//W; h_idx=t%H; t=t//H; d_idx=t%D; n_idx=t//D
    m_valid=offs_m<(N*D*H*W)
    acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)
    for tap in range(27):
        dz=tap//9-1; dy=(tap%9)//3-1; dx=tap%3-1
        zz=d_idx+dz; yy=h_idx+dy; xx=w_idx+dx
        ib=m_valid&(zz>=0)&(zz<D)&(yy>=0)&(yy<H)&(xx>=0)&(xx<W)
        base=((n_idx*D+zz)*H+yy)*W+xx
        for k0 in range(0,Cin,BLOCK_K):
            ok=k0+tl.arange(0,BLOCK_K); kv=ok<Cin
            a=tl.load(x_ptr+base[:,None]*Cin+ok[None,:], mask=ib[:,None]&kv[None,:], other=0.0)
            b=tl.load(w_ptr+offs_n[:,None]*(27*Cin)+(tap*Cin+ok)[None,:], mask=(offs_n<Cout)[:,None]&kv[None,:], other=0.0)
            acc+=tl.dot(a,b.T)
    acc=acc*sab*inv_so                       # dequant then requant to out-fp8 scale
    acc=tl.minimum(tl.maximum(acc,-448.0),448.0)
    yo=offs_m[:,None]*Cout+offs_n[None,:]
    tl.store(y_ptr+yo, acc.to(tl.float8e4nv), mask=m_valid[:,None]&(offs_n<Cout)[None,:])

def bench(fn,it=30,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1e6)
    return statistics.median(ts)

P=64; NL=4
for C in (128,320):
    N,D,H,Wd=1,P,P,P; M=N*D*H*Wd
    # weights per layer, fp8
    ws=[(torch.randn(C,C,3,3,3,device=dev)*(1/(C*27)**0.5)) for _ in range(NL)]
    sb=[(wi.abs().amax()/E).to(torch.float32) for wi in ws]
    wc=[(wi/s).clamp(-E,E).to(torch.float8_e4m3fn).reshape(C,C,27).permute(0,2,1).reshape(C,27*C).contiguous() for wi,s in zip(ws,sb)]
    x0=torch.randn(1,C,P,P,P,device=dev)*0.5
    sa0=(x0.abs().amax()/E).to(torch.float32)
    xc=(x0/sa0).clamp(-E,E).to(torch.float8_e4m3fn).permute(0,2,3,4,1).contiguous().reshape(M,C)
    grid=lambda meta:(triton.cdiv(M,meta["BLOCK_M"]),triton.cdiv(C,meta["BLOCK_N"]))
    so=torch.tensor(0.1,device=dev)  # calibrated output scale (constant here)
    bufs=[torch.empty(M,C,dtype=torch.float8_e4m3fn,device=dev) for _ in range(NL)]
    def fp8_chain():
        cur=xc; sin=sa0
        for i in range(NL):
            conv_f8f8[grid](cur,wc[i],bufs[i], (sin*sb[i]).item(), (1.0/so).item(), N,D,H,Wd,C,C)
            cur=bufs[i]; sin=so
        return cur
    t_fp8=bench(fp8_chain)
    # cuDNN fp16 chain
    xh=x0.half(); whs=[wi.half() for wi in ws]
    def f16_chain():
        c=xh
        for i in range(NL): c=F.conv3d(c,whs[i],padding=1)
        return c
    t16=bench(f16_chain)
    print(f"C={C} {NL} layers: fp8-resident {t_fp8:.0f}us | cuDNN fp16 {t16:.0f}us | speedup {t16/t_fp8:.2f}x")
