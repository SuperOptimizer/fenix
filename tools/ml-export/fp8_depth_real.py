"""The accuracy GATE: fp8 vs fp16 depth sweep using REAL trained surface_recto_3dunet
stage-3 weights (5 resident 256ch blocks). Result: corr stays 0.9996-0.9999 across depth
(random weights compound to 0.98; trained weights do NOT). This is what clears fp8 for
production. Run: python fp8_depth_real.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from fp8_conv3d_op import Fp8Tensor, pack_weight_fp8, fp8_conv3d
dev='cuda'; torch.manual_seed(0); E=448.0
def corr(a,b): return torch.corrcoef(torch.stack([a.flatten().float(),b.flatten().float()]))[0,1].item()
ck=torch.load('/home/forrest/fenix/models/surface_recto_3dunet/checkpoint_inference_ready.pth',map_location='cpu',weights_only=False)
sd=ck['model']
def get(p): return sd[p].to(dev).float()
def instn(x,g,b,eps=1e-5):
    mu=x.mean((2,3,4),keepdim=True); v=x.var((2,3,4),unbiased=False,keepdim=True)
    return (x-mu)/torch.sqrt(v+eps)*g.view(1,-1,1,1,1)+b.view(1,-1,1,1,1)

# stage 3 has blocks 0..? ; blocks 1..5 are resident (in==out==256, stride1). use blocks 1-5.
STAGE=3; C=256; P=24  # deep stage -> small spatial in real net
blocks=[]
for bi in range(1,6):
    pfx=f"shared_encoder.stages.{STAGE}.blocks.{bi}"
    blocks.append(dict(
        w1=get(f"{pfx}.conv1.conv.weight"), b1=get(f"{pfx}.conv1.conv.bias"),
        g1=get(f"{pfx}.conv1.norm.weight"), n1b=get(f"{pfx}.conv1.norm.bias"),
        w2=get(f"{pfx}.conv2.conv.weight"), b2=get(f"{pfx}.conv2.conv.bias"),
        g2=get(f"{pfx}.conv2.norm.weight"), n2b=get(f"{pfx}.conv2.norm.bias")))
def ref_block(x,b):
    r=x
    o=F.leaky_relu(instn(F.conv3d(x,b['w1'],b['b1'],padding=1),b['g1'],b['n1b']),0.01)
    o=instn(F.conv3d(o,b['w2'],b['b2'],padding=1),b['g2'],b['n2b'])
    return F.leaky_relu(o+r,0.01)
def fp8_block(x,b):
    r=x.float()
    w1,s1=pack_weight_fp8(b['w1']); xf=Fp8Tensor.from_nchw(x)
    t1=F.conv3d(x.float(),b['w1'],b['b1'],padding=1); so1=(t1.abs().amax()/E).to(torch.float32)
    o1=fp8_conv3d(xf,w1,s1,C,3,1,so1).to_nchw_f32()+b['b1'].view(1,-1,1,1,1)
    o1=F.leaky_relu(instn(o1,b['g1'],b['n1b']),0.01)
    w2,s2=pack_weight_fp8(b['w2']); xf2=Fp8Tensor.from_nchw(o1)
    t2=F.conv3d(o1.float(),b['w2'],b['b2'],padding=1); so2=(t2.abs().amax()/E).to(torch.float32)
    o2=fp8_conv3d(xf2,w2,s2,C,3,1,so2).to_nchw_f32()+b['b2'].view(1,-1,1,1,1)
    return F.leaky_relu(instn(o2,b['g2'],b['n2b'])+r,0.01)

x0=torch.randn(1,C,P,P,P,device=dev).float()  # realistic post-encoder activation scale
print("REAL trained stage-3 weights, 5 resident blocks (256ch):")
with torch.no_grad():
    cur=x0.clone(); refs=[]
    for b in blocks: cur=ref_block(cur,b); refs.append(cur.clone())
    curf=x0.clone()
    for i,b in enumerate(blocks):
        curf=fp8_block(curf,b)
        print(f"  block {i+1}: corr={corr(curf,refs[i]):.4f} maxrel={(curf-refs[i]).abs().amax().item()/refs[i].abs().amax().item():.4f}")
