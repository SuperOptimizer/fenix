"""Validate the fp8_conv3d_op variants (3x3x3 s1/s2, 1x1x1, +LeakyReLU) vs torch
reference. Expect corr >= 0.998 per op. Run: python fp8_op_validate.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from fp8_conv3d_op import Fp8Tensor, pack_weight_fp8, fp8_conv3d
dev='cuda'; torch.manual_seed(0)
def corr(a,b): return torch.corrcoef(torch.stack([a.flatten().float(),b.flatten().float()]))[0,1].item()
P,C=32,64
print(f"{'case':>18} | {'corr':>8} {'max rel err':>11}")
print("-"*45)
for name,(k,stride) in {"3x3x3 s1":(3,1),"3x3x3 s2":(3,2),"1x1x1 s1":(1,1)}.items():
    x=torch.randn(1,C,P,P,P,device=dev)*0.5
    w=torch.randn(C,C,k,k,k,device=dev)*(1/(C*k**3)**0.5)
    ref=F.conv3d(x.float(),w.float(),stride=stride,padding=(k-1)//2)
    xf=Fp8Tensor.from_nchw(x); wf,ws=pack_weight_fp8(w)
    # calibrate out scale from ref amax (static calib stand-in)
    so=(ref.abs().amax()/448.0).to(torch.float32)
    y=fp8_conv3d(xf,wf,ws,C,k,stride,so)
    yr=y.to_nchw_f32()
    c=corr(yr,ref); e=(yr-ref).abs().amax().item()/(ref.abs().amax().item()+1e-8)
    print(f"{name:>18} | {c:>8.4f} {e:>11.4f}")
# with act
x=torch.randn(1,C,P,P,P,device=dev)*0.5
w=torch.randn(C,C,3,3,3,device=dev)*(1/(C*27)**0.5)
ref=F.leaky_relu(F.conv3d(x.float(),w.float(),padding=1),0.01)
xf=Fp8Tensor.from_nchw(x); wf,ws=pack_weight_fp8(w)
so=(ref.abs().amax()/448.0).to(torch.float32)
y=fp8_conv3d(xf,wf,ws,C,3,1,so,act=True)
yr=y.to_nchw_f32()
print(f"{'3x3x3 s1 +lrelu':>18} | {corr(yr,ref):>8.4f} {(yr-ref).abs().amax().item()/(ref.abs().amax().item()+1e-8):>11.4f}")
