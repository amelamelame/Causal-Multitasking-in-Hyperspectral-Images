"""
models/baseline/model.py
=========================
Full Baseline Multitask Model for TAIGA:
  SharedEncoder (ResNet + DenseASPP)
  → SpectralSpatialAttention
  → 3 ClassificationDecoders + 10 RegressionDecoders
Optimized for RTX 3060 6GB with gradient checkpointing + FP16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


# ─────────────────── Building Blocks ───────────────────────

class ConvBNLeaky(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, k, stride=s, padding=p*d, dilation=d, bias=False),
            nn.BatchNorm2d(oc), nn.LeakyReLU(0.1, inplace=True))
    def forward(self, x): return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ic, oc, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.c1   = nn.Conv2d(ic, oc, 3, padding=1, bias=False)
        self.b1   = nn.BatchNorm2d(oc)
        self.c2   = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.b2   = nn.BatchNorm2d(oc)
        self.act  = nn.LeakyReLU(0.1, inplace=True)
        self.skip = nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),
                                   nn.BatchNorm2d(oc)) if ic!=oc else nn.Identity()

    def _fwd(self, x):
        return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + self.skip(x))

    def forward(self, x):
        return grad_ckpt(self._fwd, x, use_reentrant=False) if (self.ckpt and self.training) else self._fwd(x)


# ─────────────────── Dense ASPP ────────────────────────────

class DenseASPP(nn.Module):
    def __init__(self, ic, rates=(6,12,18)):
        super().__init__()
        mc = ic // 2
        self.reduce = nn.ModuleList([ConvBNLeaky(ic+i*mc,mc,1,p=0) for i in range(3)])
        self.atrous  = nn.ModuleList([ConvBNLeaky(mc,mc,3,d=r,p=r) for r in rates])
        self.final   = ConvBNLeaky(ic+3*mc, ic*2, 1, p=0)
        self.out_ch  = ic*2

    def forward(self, x):
        feats = [x]
        for red, atr in zip(self.reduce, self.atrous):
            feats.append(atr(red(torch.cat(feats,1))))
        out = torch.cat(feats, 1)
        out = self.final(out)
        H,W = out.shape[2:]
        if H>4: out = F.adaptive_avg_pool2d(out,(H//2,W//2))
        return out


# ─────────────────── Attention ─────────────────────────────

class SpectralAttn(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c,c//r,False),nn.ReLU(),nn.Linear(c//r,c,False))
    def forward(self, x):
        B,C,H,W = x.shape
        pa = x.mean([2,3]); pm = x.amax([2,3])
        return torch.sigmoid(self.mlp(pa)+F.relu(self.mlp(pm))).view(B,C,1,1)

class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.BatchNorm2d(1))
    def forward(self, x):
        return torch.sigmoid(self.conv(torch.cat([x.mean(1,True),x.amax(1,True)],1)))

class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.spec = SpectralAttn(c, r)
        self.spat = SpatialAttn()
        self.pool = ConvBNLeaky(c, c, 1, p=0)
    def forward(self, x):
        f = self.pool(x)
        f = f * self.spec(f)
        f = f * self.spat(f)
        return x + f


# ─────────────────── Shared Encoder ────────────────────────

class SharedEncoder(nn.Module):
    def __init__(self, in_ch=128, channels=None, ckpt=True, drop=0.3):
        super().__init__()
        ch = channels or [64,128,128,256]
        self.stem   = ConvBNLeaky(in_ch, ch[0])
        self.s1     = ResBlock(ch[0], ch[1], ckpt)
        self.dn1    = nn.MaxPool2d(2,2)
        self.s2     = ResBlock(ch[1], ch[2], ckpt)
        self.dn2    = nn.MaxPool2d(2,2)
        self.s3     = ResBlock(ch[2], ch[3], ckpt)
        self.aspp   = DenseASPP(ch[3])
        self.drop   = nn.Dropout2d(drop)
        self.out_ch = self.aspp.out_ch
        self.chs    = ch

    def forward(self, x):
        skips = {}
        s0 = self.stem(x);  skips['s0']=s0
        s1 = self.s1(s0);   skips['s1']=s1; s1=self.dn1(s1)
        s2 = self.s2(s1);   skips['s2']=s2; s2=self.dn2(s2)
        s3 = self.s3(s2);   skips['s3']=s3
        return self.aspp(self.drop(s3)), skips


# ─────────────────── Decoder Blocks ────────────────────────

class DecBlock(nn.Module):
    def __init__(self, ic, sc, oc, ckpt=False):
        super().__init__()
        self.red = ConvBNLeaky(ic+sc, oc, 1, p=0)
        self.res = ResBlock(oc, oc, ckpt)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.res(self.red(torch.cat([x,skip],1)))

class ClsDecoder(nn.Module):
    def __init__(self, enc_ch, skips, dc, n_cls, ckpt=False):
        super().__init__()
        self.d3 = DecBlock(enc_ch,    skips['s3'], dc,    ckpt)
        self.d2 = DecBlock(dc,        skips['s2'], dc//2, ckpt)
        self.d1 = DecBlock(dc//2,     skips['s1'], dc//4, ckpt)
        self.d0 = DecBlock(dc//4,     skips['s0'], dc//8, ckpt)
        fc = dc//8
        self.attn = CBAM(fc)
        self.head = nn.Conv2d(fc, n_cls, 1)

    def forward(self, e, sk, tsz):
        x = self.d3(e,sk['s3']); x=self.d2(x,sk['s2'])
        x = self.d1(x,sk['s1']); x=self.d0(x,sk['s0'])
        x = self.attn(x)
        return self.head(F.interpolate(x,tsz,mode='bilinear',align_corners=False))

class RegDecoder(nn.Module):
    def __init__(self, enc_ch, skips, dc, ckpt=False):
        super().__init__()
        self.d3 = DecBlock(enc_ch,    skips['s3'], dc,    ckpt)
        self.d2 = DecBlock(dc,        skips['s2'], dc//2, ckpt)
        self.d1 = DecBlock(dc//2,     skips['s1'], dc//4, ckpt)
        self.d0 = DecBlock(dc//4,     skips['s0'], dc//8, ckpt)
        fc = dc//8
        self.attn = CBAM(fc)
        self.head = nn.Sequential(nn.Conv2d(fc,1,1), nn.Sigmoid())

    def forward(self, e, sk, tsz):
        x = self.d3(e,sk['s3']); x=self.d2(x,sk['s2'])
        x = self.d1(x,sk['s1']); x=self.d0(x,sk['s0'])
        x = self.attn(x)
        return self.head(F.interpolate(x,tsz,mode='bilinear',align_corners=False))


# ─────────────────── Full Model ────────────────────────────

from data.envi_reader import CAT_VARIABLES, REG_VARIABLES

class MultitaskModel(nn.Module):
    """Full baseline multitask model — Phase 1."""
    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg["model"]
        ch  = m["encoder_channels"]
        dc  = m["decoder_channels"]
        ckpt = m["use_checkpoint"]

        self.encoder = SharedEncoder(in_ch=cfg["data"]["num_bands"],
                                     channels=ch, ckpt=ckpt, drop=m["dropout"])
        self.bridge  = CBAM(self.encoder.out_ch, m["attn_reduction"])

        # Build skip_sizes dict from encoder channel list
        skips = {'s0':ch[0],'s1':ch[1],'s2':ch[2],'s3':ch[3]}
        enc_ch = self.encoder.out_ch

        self.cls_decoders = nn.ModuleDict({
            v: ClsDecoder(enc_ch, skips, dc, n, ckpt)
            for v, n in CAT_VARIABLES.items()
        })
        self.reg_decoders = nn.ModuleDict({
            v: RegDecoder(enc_ch, skips, dc, ckpt)
            for v in REG_VARIABLES
        })

    def forward(self, x: torch.Tensor) -> dict:
        tsz = x.shape[2:]
        enc, skips = self.encoder(x)
        enc = self.bridge(enc)
        return {
            "cls": {v: d(enc, skips, tsz) for v, d in self.cls_decoders.items()},
            "reg": {v: d(enc, skips, tsz) for v, d in self.reg_decoders.items()},
        }

    def param_count(self):
        return sum(p.numel() for p in self.parameters()) / 1e6
