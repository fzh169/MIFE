import matplotlib.pyplot as plt
import torch
import sys
from torch.nn import functional as F
import torch.nn as nn
from einops import rearrange, repeat
from utility import moduleNormalize


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args

        self.fe = FeatureExtractor(dim=(32, 64, 128, 256))
        self.sn = ScoreNet(dim=(16, 32, 64, 96), in_ch=27, out_ch=9, kernel=3, dilation=(2, 4, 8))

        self.flow0 = RefineFlow(dim=64,  part_size=8,  kernel=3, dilation=2)
        self.flow1 = RefineFlow(dim=128, part_size=16, kernel=3, dilation=4)
        self.flow2 = RefineFlow(dim=256, part_size=20, kernel=3, dilation=8)

        self.fn = FusionNet(dim=(16, 32, 64, 96), in_ch=2, out_ch=2)
        self.mn = MaskNet(dim=32, in_ch=1)

        self.ce = ContextExtractor(dim=(16, 32, 64, 96))
        self.sy = SynthesisNet(dim=((16, 32, 64, 96), (96, 192, 288)), in_ch=17, out_ch=3)

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        x = 8
        if h0 % x != 0:
            pad_h = x - (h0 % x)
            frame0 = F.pad(frame0, [0, 0, 0, pad_h])
            frame2 = F.pad(frame2, [0, 0, 0, pad_h])

        if w0 % x != 0:
            pad_w = x - (w0 % x)
            frame0 = F.pad(frame0, [0, pad_w, 0, 0])
            frame2 = F.pad(frame2, [0, pad_w, 0, 0])

        nf0 = moduleNormalize(frame0)
        nf2 = moduleNormalize(frame2)

        feat00, feat10, feat20 = self.fe(nf0)
        feat02, feat12, feat22 = self.fe(nf2)

        score00, score10, score20 = self.sn(nf0)
        score02, score12, score22 = self.sn(nf2)

        flow10_0, flow12_0, flow02_0, flow20_0, range0_0, range1_0 = self.flow0(feat00, feat02, score00, score02)
        flow10_1, flow12_1, flow02_1, flow20_1, range0_1, range1_1 = self.flow1(feat10, feat12, score10, score12)
        flow10_2, flow12_2, flow02_2, flow20_2, range0_2, range1_2 = self.flow2(feat20, feat22, score20, score22)

        flow10 = self.fn(flow10_0, flow10_1, flow10_2, range1_0, range1_1)
        flow12 = self.fn(flow12_0, flow12_1, flow12_2, range1_0, range1_1)
        flow02 = self.fn(flow02_0, flow02_1, flow02_2, range0_0, range0_1)
        flow20 = self.fn(flow20_0, flow20_1, flow20_2, range0_0, range0_1)

        # frame1_hat = self.generate(flow10, flow12, flow02, flow20, frame0, frame2)[:, :, 0:h0, 0:w0]

        f10 = self.backwarp(flow10, frame0)
        f12 = self.backwarp(flow12, frame2)

        rad02 = (flow02[:, 0] ** 2 + flow02[:, 1] ** 2 + 1e-6) ** 0.5
        rad20 = (flow20[:, 0] ** 2 + flow20[:, 1] ** 2 + 1e-6) ** 0.5

        alpha = self.mn((rad20 - rad02).unsqueeze(1))
        frame1_hat = alpha * f10 + (1 - alpha) * f12

        c0 = self.ce(frame0, flow10)
        c1 = self.ce(frame2, flow12)
        res = self.sy(torch.cat([frame0, frame2, f10, f12, flow10, flow12, alpha], dim=1), c0, c1) * 2 - 1
        frame1_hat = torch.clamp(frame1_hat + res, 0, 1)[:, :, 0:h0, 0:w0]

        if self.training:

            g0 = self.generate(flow10_0, flow12_0, flow02_0, flow20_0, frame0, frame2)[:, :, 0:h0, 0:w0]
            g1 = self.generate(flow10_1, flow12_1, flow02_1, flow20_1, frame0, frame2)[:, :, 0:h0, 0:w0]
            g2 = self.generate(flow10_2, flow12_2, flow02_2, flow20_2, frame0, frame2)[:, :, 0:h0, 0:w0]

            return {'frame1': frame1_hat, 'g0': g0, 'g1': g1, 'g2': g2}
        else:
            return frame1_hat

    def generate(self, flow10, flow12, flow02, flow20, frame0, frame2):

        f10 = self.backwarp(flow10, frame0)
        f12 = self.backwarp(flow12, frame2)

        rad02 = (flow02[:, 0] ** 2 + flow02[:, 1] ** 2 + 1e-6) ** 0.5
        rad20 = (flow20[:, 0] ** 2 + flow20[:, 1] ** 2 + 1e-6) ** 0.5

        alpha = self.mn((rad20 - rad02).unsqueeze(1))
        frame1_hat = alpha * f10 + (1 - alpha) * f12
        
        return frame1_hat

    def backwarp(self, flow, frame):

        B, _, H, W = flow.shape

        bx = torch.arange(0.0, W, device="cuda")
        by = torch.arange(0.0, H, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)

        coords = base + flow

        x = coords[:, 0]
        y = coords[:, 1]

        x = 2 * (x / (W - 1.0) - 0.5)
        y = 2 * (y / (H - 1.0) - 0.5)
        grid = torch.stack((x, y), dim=3)

        frame_hat = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return frame_hat


class BasicFlow(nn.Module):
    def __init__(self, dim, part_size):
        super(BasicFlow, self).__init__()

        self.scale = dim ** -0.5
        self.part_h = part_size
        self.part_w = part_size

        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, feat0, feat2, score0, score2):

        B, C, H, W = feat0.shape

        feat0 = self.proj(feat0)
        feat2 = self.proj(feat2)

        pad_h, pad_w = 0, 0
        if H % self.part_h != 0:
            pad_h = self.part_h - (H % self.part_h)
            feat0 = F.pad(feat0, [0, 0, 0, pad_h])
            feat2 = F.pad(feat2, [0, 0, 0, pad_h])

        if W % self.part_w != 0:
            pad_w = self.part_w - (W % self.part_w)
            feat0 = F.pad(feat0, [0, pad_w, 0, 0])
            feat2 = F.pad(feat2, [0, pad_w, 0, 0])

        s1 = int((H + pad_h) / self.part_h)
        s2 = int((W + pad_w) / self.part_w)

        flow_00, _, flow02_00, _, flow20_00, _, range0_00, range1_00 = self.part(feat0, feat2, (False, False))
        flow_01, _, flow02_01, _, flow20_01, _, range0_01, range1_01 = self.part(feat0, feat2, (False, True))
        flow_10, _, flow02_10, _, flow20_10, _, range0_10, range1_10 = self.part(feat0, feat2, (True, False))
        flow_11, _, flow02_11, _, flow20_11, _, range0_11, range1_11 = self.part(feat0, feat2, (True, True))

        flow = self.splice(flow_00, flow_01, flow_10, flow_11, 2, s1, s2, H, W)

        flow02 = self.splice(flow02_00, flow02_01, flow02_10, flow02_11, 1, s1, s2, H, W)
        flow20 = self.splice(flow20_00, flow20_01, flow20_10, flow20_11, 1, s1, s2, H, W)

        range0 = self.splice(range0_00, range0_01, range0_10, range0_11, 1, s1, s2, H, W)
        range1 = self.splice(range1_00, range1_01, range1_10, range1_11, 2, s1, s2, H, W)

        flow10 = flow
        flow12 = flow * -1

        flow02 = repeat(flow02, 'b f h w -> b f (h n1) (w n2)', n1=2, n2=2) * 2
        flow20 = repeat(flow20, 'b f h w -> b f (h n1) (w n2)', n1=2, n2=2) * 2
        range0 = repeat(range0, 'b f h w -> b f (h n1) (w n2)', n1=2, n2=2) * 2

        return flow10, flow12, flow02, flow20, range0, range1

    def part(self, feat0, feat2, shift):

        B, C, H, W = feat0.shape

        shift_h, shift_w = 0, 0
        if shift[0]:
            shift_h = int(self.part_h / 2)
        if shift[1]:
            shift_w = int(self.part_w / 2)

        mask = None
        if shift[0] or shift[1]:
            feat0 = torch.roll(feat0, shifts=(-shift_h, -shift_w), dims=(2, 3))
            feat2 = torch.roll(feat2, shifts=(-shift_h, -shift_w), dims=(2, 3))

            mask = torch.zeros((H, W)).type_as(feat0)
            mask = self.mask(mask, shift_h, shift_w)
            mask = repeat(mask, 's t1 t2 -> (b s) t1 t2', b=B)

        q = rearrange(feat0, 'b c (s1 h) (s2 w) -> (b s1 s2) (h w) c', h=self.part_h, w=self.part_w)
        k = rearrange(feat2, 'b c (s1 h) (s2 w) -> (b s1 s2) (h w) c', h=self.part_h, w=self.part_w)

        corr = torch.einsum('blk,btk->blt', q, k) * self.scale

        if mask is not None:
            corr = corr + mask

        dx = torch.arange(0.0, self.part_w, device="cuda")
        dy = torch.arange(0.0, self.part_h, device="cuda")
        meshy, meshx = torch.meshgrid(dy, dx)
        pos = torch.stack((meshx, meshy), dim=0).unsqueeze(0)
        pos = rearrange(pos, 'b f h w -> b f (h w)')

        flow02, corr02, range0 = self.flow(corr, pos, False)
        flow20, corr20, range0 = self.flow(corr.permute(0, 2, 1), pos, False)

        corr, base = self.select(corr)
        flow, corr, range1 = self.flow(corr, pos, True, base)

        range0 = range0.repeat(flow.size()[0], 1, 1, 1)
        range1 = range1.repeat(flow.size()[0], 1, 1, 1)

        return flow, corr, flow02, corr02, flow20, corr20, range0, range1

    def mask(self, mask, h, w):

        if h != 0:
            h_slices = (slice(0, -h * 2), slice(-h * 2, -h), slice(-h, None))
        else:
            h_slices = (slice(0, None),)
        if w != 0:
            w_slices = (slice(0, -w * 2), slice(-w * 2, -w), slice(-w, None))
        else:
            w_slices = (slice(0, None),)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[h, w] = cnt
                cnt += 1
        mask_windows = rearrange(mask, '(s1 h) (s2 w) -> (s1 s2) (h w)', h=self.part_h, w=self.part_w)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (s1 s2) (hw) (hw)
        mask = mask.masked_fill(mask != 0, float(-10000.0)).masked_fill(mask == 0, float(0.0))

        return mask

    def flow(self, corr, pos, mid, base=None):

        if base is None:
            cut = (slice(int(self.part_h / 4), int(-self.part_h / 4)),
                   slice(int(self.part_w / 4), int(-self.part_w / 4)))

            corr = rearrange(corr, 'b (h w) t -> b h w t', h=self.part_h)[:, cut[0], cut[1], :]
            corr = rearrange(corr, 'b h w t -> b (h w) t')

            bx = torch.arange(0.0, self.part_w, device="cuda")
            by = torch.arange(0.0, self.part_h, device="cuda")
            meshy, meshx = torch.meshgrid(by, bx)
            base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)[:, :, cut[0], cut[1]]

        bh = base.size()[2]
        base = rearrange(base, 'b f h w -> b f (h w)')
        flow = pos.unsqueeze(2) - base.unsqueeze(3)

        if mid:
            maxx = torch.max(flow.clone().detach(), dim=3)[0]
            minn = torch.min(flow.clone().detach(), dim=3)[0]
            range = torch.min(torch.stack([maxx, -minn], dim=3), dim=3)[0]

            corr = rearrange(corr, 'b (h w) t -> b h w t', h=bh).unsqueeze(1)
            flow = rearrange(flow, 'b f (h w) t -> b f h w t', h=bh)
            range = rearrange(range, 'b f (h w) -> b f h w', h=bh)
            bh = bh - 1

            corr = torch.cat([corr[:, :, :-1, :-1], corr[:, :, :-1, 1:], corr[:, :, 1:, :-1], corr[:, :, 1:, 1:]], dim=1)
            flow = torch.cat([flow[:, :, :-1, :-1], flow[:, :, :-1, 1:], flow[:, :, 1:, :-1], flow[:, :, 1:, 1:]], dim=1)
            corr = rearrange(corr, 'b n h w t -> b (h w) (n t)')
            flow = rearrange(flow, 'b (n f) h w t -> b f (h w) (n t)', f=2) * 2

            range = torch.cat([range[:, :, :-1, :-1], range[:, :, :-1, 1:], range[:, :, 1:, :-1], range[:, :, 1:, 1:]], dim=1)
            range = rearrange(range, 'b (n f) h w -> b f n h w', f=2)

            range = torch.min(range, dim=2)[0]
            range = torch.cat([-range, range], dim=1) * 2
        else:
            maxx = torch.max(flow.clone().detach(), dim=3)[0]
            minn = torch.min(flow.clone().detach(), dim=3)[0]
            range = torch.cat([minn, maxx], dim=1)
            range = rearrange(range, 'b f (h w) -> b f h w', h=bh)

        smax = torch.softmax(corr, dim=2)
        corr = torch.sum(corr * smax, dim=2)
        flow = torch.sum(flow * smax.unsqueeze(1), dim=3)

        corr = rearrange(corr, 'b (h w) -> b h w', h=bh).unsqueeze(1)
        flow = rearrange(flow, 'b f (h w) -> b f h w', h=bh)

        return flow, corr, range

    def select(self, score):

        ch, cw = int(self.part_h / 2) - 1, int(self.part_w / 2) - 1
        cut_h, cut_w = (slice(ch, -ch), slice(cw, -cw))

        score = rearrange(score, 'b (h0 w0) (h2 w2) -> h0 w0 b h2 w2', h0=self.part_h, h2=self.part_h)

        temp = []
        for i in range(0, self.part_h):
            s = F.pad(score[i], [0, 0, i, self.part_h - i - 1])[:, :, cut_h, :]
            temp.append(s.unsqueeze(0))
        score = torch.cat(temp, dim=0)

        temp = []
        for i in range(0, self.part_w):
            s = F.pad(score[:, i], [i, self.part_w - i - 1, 0, 0])[:, :, :, cut_w]
            temp.append(s.unsqueeze(1))
        score = torch.cat(temp, dim=1)

        score = rearrange(score, 'h0 w0 b h1 w1 -> b (h1 w1) (h0 w0)')

        bx = torch.arange(0.0, self.part_w - 0.5, step=0.5, device="cuda")
        by = torch.arange(0.0, self.part_h - 0.5, step=0.5, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)[:, :, cut_h, cut_w]

        return score, base

    def splice(self, feat00, feat01, feat10, feat11, factor, s1, s2, H, W):

        feat0 = torch.cat([feat00, feat01], dim=3)
        feat1 = torch.cat([feat10, feat11], dim=3)
        feat = torch.cat([feat0, feat1], dim=2)
        feat = rearrange(feat, '(b s1 s2) k h w -> b k (s1 h) (s2 w)', s1=s1, s2=s2)

        shift_h = int(self.part_h / 4 * factor)
        shift_w = int(self.part_w / 4 * factor)

        feat = torch.roll(feat, shifts=(shift_h, shift_w), dims=(2, 3))
        feat = feat[:, :, 0:H * factor, 0:W * factor]

        return feat


class RefineFlow(BasicFlow):
    def __init__(self, dim, part_size, kernel, dilation):
        super(RefineFlow, self).__init__(dim, part_size)

        self.kernel_size = kernel
        self.dilation = dilation
        self.padding = int(self.kernel_size / 2) * self.dilation

        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, feat0, feat2, score0, score2):

        B, C, H, W = feat0.shape

        feat0 = self.proj(feat0)
        feat2 = self.proj(feat2)

        pad_h, pad_w = 0, 0
        if H % self.part_h != 0:
            pad_h = self.part_h - (H % self.part_h)
            feat0 = F.pad(feat0, [0, 0, 0, pad_h])
            feat2 = F.pad(feat2, [0, 0, 0, pad_h])

        if W % self.part_w != 0:
            pad_w = self.part_w - (W % self.part_w)
            feat0 = F.pad(feat0, [0, pad_w, 0, 0])
            feat2 = F.pad(feat2, [0, pad_w, 0, 0])

        s1 = int((H + pad_h) / self.part_h)
        s2 = int((W + pad_w) / self.part_w)

        flow_00, corr_00, flow02_00, corr02_00, flow20_00, corr20_00, range0_00, range1_00 = self.part(feat0, feat2, (False, False))
        flow_01, corr_01, flow02_01, corr02_01, flow20_01, corr20_01, range0_01, range1_01 = self.part(feat0, feat2, (False, True))
        flow_10, corr_10, flow02_10, corr02_10, flow20_10, corr20_10, range0_10, range1_10 = self.part(feat0, feat2, (True, False))
        flow_11, corr_11, flow02_11, corr02_11, flow20_11, corr20_11, range0_11, range1_11 = self.part(feat0, feat2, (True, True))

        flow = self.splice(flow_00, flow_01, flow_10, flow_11, 2, s1, s2, H, W)
        corr = self.splice(corr_00, corr_01, corr_10, corr_11, 2, s1, s2, H, W)

        flow02 = self.splice(flow02_00, flow02_01, flow02_10, flow02_11, 1, s1, s2, H, W)
        flow20 = self.splice(flow20_00, flow20_01, flow20_10, flow20_11, 1, s1, s2, H, W)
        corr02 = self.splice(corr02_00, corr02_01, corr02_10, corr02_11, 1, s1, s2, H, W)
        corr20 = self.splice(corr20_00, corr20_01, corr20_10, corr20_11, 1, s1, s2, H, W)

        range0 = self.splice(range0_00, range0_01, range0_10, range0_11, 1, s1, s2, H, W)
        range1 = self.splice(range1_00, range1_01, range1_10, range1_11, 2, s1, s2, H, W)

        corr10, flow10 = corr, flow
        corr12, flow12 = corr, flow * -1

        factor1 = int(self.dilation / 2)
        factor0 = self.dilation

        flow10 = self.refine(flow10, corr10, score0, factor1, False)
        flow12 = self.refine(flow12, corr12, score2, factor1, False)
        flow02 = self.refine(flow02, corr02, score0, factor0, True)
        flow20 = self.refine(flow20, corr20, score2, factor0, True)

        if factor0 != 1:
            range0 = repeat(range0, 'b f h w -> b f (h n1) (w n2)', n1=factor0, n2=factor0) * factor0
        if factor1 != 1:
            range1 = repeat(range1, 'b f h w -> b f (h n1) (w n2)', n1=factor1, n2=factor1) * factor1

        return flow10, flow12, flow02, flow20, range0, range1

    def refine(self, flow, corr, score, factor, target):

        if factor != 1:
            flow = repeat(flow, 'b f h w -> b f (h n1) (w n2)', n1=factor, n2=factor) * factor
            corr = repeat(corr, 'b n h w -> b n (h n1) (w n2)', n1=factor, n2=factor)

        B, _, H, W = flow.shape

        flow = F.unfold(flow, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding)
        corr = F.unfold(corr, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding)
        flow = rearrange(flow, 'b (f k) (h w) -> b f k h w', f=2, h=H)
        corr = rearrange(corr, 'b k (h w) -> b k h w', h=H)

        if not target:
            score = self.sample(flow, score)

        smax = torch.softmax(score * corr, dim=1)
        flow = torch.sum(flow * smax.unsqueeze(1), dim=2)

        return flow

    def sample(self, flow, score):

        B, _, H, W = score.shape

        bx = torch.arange(0.0, W, device="cuda")
        by = torch.arange(0.0, H, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0)

        coords = base.unsqueeze(0).unsqueeze(2) + flow
        coords = rearrange(coords, 'b f k h w -> (b k) f h w')

        x = coords[:, 0]
        y = coords[:, 1]

        x = 2 * (x / (W - 1.0) - 0.5)
        y = 2 * (y / (H - 1.0) - 0.5)
        grid = torch.stack((x, y), dim=3)

        score = rearrange(score, 'b k h w -> (b k) h w').unsqueeze(1)
        score = F.grid_sample(score, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(1)
        score = rearrange(score, '(b k) h w -> b k h w', k=self.kernel_size ** 2)

        return score


class FeatureExtractor(nn.Module):
    def __init__(self, dim):
        super(FeatureExtractor, self).__init__()

        self.emb = nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=3, stride=1, padding=1)
        self.fe0 = FEBlock(dim[0], dim[1])
        self.fe1 = FEBlock(dim[1], dim[2])
        self.fe2 = FEBlock(dim[2], dim[3])

    def forward(self, frame):
        feat0 = self.fe0(self.emb(frame))
        feat1 = self.fe1(feat0)
        feat2 = self.fe2(feat1)

        return feat0, feat1, feat2


class ScoreNet(nn.Module):
    def __init__(self, dim, in_ch, out_ch, kernel, dilation):
        super(ScoreNet, self).__init__()

        self.kernel_size = kernel
        self.dilation = dilation
        self.padding = int(self.kernel_size / 2) * self.dilation

        self.init = ResBlock(in_ch, dim[0])

        self.sd0 = DownBlock(dim[0], dim[1], True)
        self.sd1 = DownBlock(dim[1], dim[2], True)
        self.sd2 = DownBlock(dim[2], dim[3], True)

        self.su2 = UpBlock(dim[3], dim[2], True)
        self.su1 = UpBlock(dim[2], dim[1], True)
        self.su0 = UpBlock(dim[1], dim[0], True)

        self.last = nn.Conv2d(in_channels=dim[0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, frame):
        B, _, H, W = frame.shape

        feat0 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[0], padding=self.padding[0])
        feat1 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[1], padding=self.padding[1])
        feat2 = F.unfold(frame, kernel_size=self.kernel_size, dilation=self.dilation[2], padding=self.padding[2])

        feat0 = rearrange(feat0, 'b k (h w) -> b k h w', h=H)
        feat1 = rearrange(feat1, 'b k (h w) -> b k h w', h=H)
        feat2 = rearrange(feat2, 'b k (h w) -> b k h w', h=H)

        score0 = self.snet(feat0)
        score1 = self.snet(feat1)
        score2 = self.snet(feat2)

        return score0, score1, score2

    def snet(self, frame):

        feat00 = self.init(frame)

        feat01 = self.sd0(feat00)
        feat02 = self.sd1(feat01)
        feat03 = self.sd2(feat02)

        feat12 = self.su2(feat03, feat02)
        feat11 = self.su1(feat12, feat01)
        feat10 = self.su0(feat11, feat00)

        return self.last(feat10)


class FusionNet(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super(FusionNet, self).__init__()

        self.init = ResBlock(in_ch, dim[0])

        self.fd0 = DownBlock(dim[0], dim[1])
        self.fd1 = DownBlock(dim[1], dim[2])
        self.fd2 = DownBlock(dim[2], dim[3])

        self.fu2 = UpBlock(dim[3], dim[2])
        self.fu1 = UpBlock(dim[2], dim[1])
        self.fu0 = UpBlock(dim[1], dim[0])

        self.last = nn.Conv2d(in_channels=dim[0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, flow0, flow1, flow2, range0, range1):

        _, _, H, W = flow0.shape

        range0 = rearrange(range0, 'b (m f) h w -> b m f h w', f=2)
        range1 = rearrange(range1, 'b (m f) h w -> b m f h w', f=2)

        mask2 = flow2.clone().detach()
        mask2 = torch.cat([range1[:, 0] - mask2, mask2 - range1[:, 1]], dim=1) / 2
        mask2 = self.conv(mask2)
        flow1 = flow2 * mask2 + flow1 * (1 - mask2)

        mask1 = flow1.clone().detach()
        mask1 = torch.cat([range0[:, 0] - mask1, mask1 - range0[:, 1]], dim=1)
        mask1 = self.conv(mask1)
        flow0 = flow1 * mask1 + flow0 * (1 - mask1)

        return self.fnet(flow0)

    def fnet(self, flow):

        feat00 = self.init(flow)

        feat01 = self.fd0(feat00)
        feat02 = self.fd1(feat01)
        feat03 = self.fd2(feat02)

        feat12 = self.fu2(feat03, feat02)
        feat11 = self.fu1(feat12, feat01)
        feat10 = self.fu0(feat11, feat00)

        return self.last(feat10)


class MaskNet(nn.Module):
    def __init__(self, dim, in_ch):
        super(MaskNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = torch.sigmoid(out + x)

        return out


class ContextExtractor(nn.Module):
    def __init__(self, dim):
        super(ContextExtractor, self).__init__()

        self.emb = nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=3, stride=1, padding=1)
        self.ce0 = CEBlock(dim[0], dim[1])
        self.ce1 = CEBlock(dim[1], dim[2])
        self.ce2 = CEBlock(dim[2], dim[3])

    def forward(self, frame, flow):

        feat0 = self.ce0(self.emb(frame))
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f0 = self.backwarp(flow, feat0)

        feat1 = self.ce1(feat0)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = self.backwarp(flow, feat1)

        feat2 = self.ce2(feat1)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = self.backwarp(flow, feat2)

        return [f0, f1, f2]

    def backwarp(self, flow, feat):

        B, _, H, W = flow.shape

        bx = torch.arange(0.0, W, device="cuda")
        by = torch.arange(0.0, H, device="cuda")
        meshy, meshx = torch.meshgrid(by, bx)
        base = torch.stack((meshx, meshy), dim=0).unsqueeze(0)

        coords = base + flow

        x = coords[:, 0]
        y = coords[:, 1]

        x = 2 * (x / (W - 1.0) - 0.5)
        y = 2 * (y / (H - 1.0) - 0.5)
        grid = torch.stack((x, y), dim=3)

        feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return feat


class SynthesisNet(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super(SynthesisNet, self).__init__()

        self.init = ResBlock(in_ch, dim[0][0])

        self.sd0 = DownBlock(dim[0][0], dim[0][1])
        self.sd1 = DownBlock(dim[1][0], dim[0][2])
        self.sd2 = DownBlock(dim[1][1], dim[0][3])

        self.su2 = UpBlock(dim[1][2], dim[0][2])
        self.su1 = UpBlock(dim[0][2], dim[0][1])
        self.su0 = UpBlock(dim[0][1], dim[0][0])

        self.last = nn.Conv2d(in_channels=dim[0][0], out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, feat, c0, c1):

        feat00 = self.init(feat)

        feat01 = self.sd0(feat00)
        feat02 = self.sd1(torch.cat([feat01, c0[0], c1[0]], dim=1))
        feat03 = self.sd2(torch.cat([feat02, c0[1], c1[1]], dim=1))

        feat12 = self.su2(torch.cat([feat03, c0[2], c1[2]], dim=1), feat02)
        feat11 = self.su1(feat12, feat01)
        feat10 = self.su0(feat11, feat00)

        return torch.sigmoid(self.last(feat10))


class FEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FEBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = nn.Sequential(ResBlock(out_ch, out_ch), ResBlock(out_ch, out_ch))
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class CEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CEBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)

        return feat


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False):
        super(DownBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)

        if norm:
            self.norm = nn.LayerNorm(out_ch)
        else:
            self.norm = lambda x: x

    def forward(self, feat):
        feat = self.down(feat)
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False):
        super(UpBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv = ResBlock(out_ch, out_ch)

        if norm:
            self.norm = nn.LayerNorm(out_ch)
        else:
            self.norm = lambda x: x

    def forward(self, feat1, feat0):
        feat = self.up(feat1) + feat0
        feat = self.conv(feat)
        feat = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        # self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        # self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu2 = nn.PReLU()

        if in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1))
        else:
            self.downsample = lambda x: x

    def forward(self, x):

        out = self.conv1(x)
        # out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        # out = self.norm2(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu2(out)

        return out
