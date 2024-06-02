import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
import matplotlib.pyplot as plt


class MixQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def gumbel_softmax(self, logits, tau=1, hard=False, dim=-1, gumbel_noise=1.0):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels*gumbel_noise) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def argmax_softmax(self, logits, tau=1, dim=-1):
        logits = logits / tau
        y_soft = logits.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def forward(self, z=None, temp=None, return_logits=False, provided_idx=None):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work

        if provided_idx == None:

            hard = self.straight_through if self.training else True
            temp = self.temperature if temp is None else temp

            logits = self.proj(z)
            if self.remap is not None:
                # continue only with used logits
                full_zeros = torch.zeros_like(logits)
                logits = logits[:, self.used, ...]

            if self.training:
                mask = self.random_masking(logits, mask_ratio=0.5)  # b, 1, h, w
                # soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
                stoch_one_hot = self.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
                deter_one_hot = self.argmax_softmax(logits, tau=temp, dim=1)
                # print (stoch_one_hot.shape, mask.shape)
                # 1/0
                soft_one_hot = stoch_one_hot * mask + deter_one_hot * (1-mask)
            else:
                # print('*******test********')
                y_soft = logits.softmax(dim=1)
                index = y_soft.max(dim=1, keepdim=True)[1]
                soft_one_hot = torch.zeros_like(logits).scatter_(1, index, 1.0)

            if self.remap is not None:
                # go back to all entries but unused set to zero
                full_zeros[:, self.used, ...] = soft_one_hot
                soft_one_hot = full_zeros
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

            # + kl divergence to the prior loss
            qy = F.softmax(logits, dim=1)
            prior_loss = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

            ind = soft_one_hot.argmax(dim=1)
            if self.remap is not None:
                ind = self.remap_to_used(ind)
            if self.use_vqinterface:
                if return_logits:
                    return z_q, diff, (None, None, ind), logits
                return z_q, diff, (None, None, ind)
            return z_q, prior_loss, ind

        else:
            b = provided_idx.shape[0]
            one_hot = torch.nn.functional.one_hot(provided_idx.to(torch.int64), 1024)
            # print(one_hot.shape)
            one_hot = one_hot.view(b, 1024, 1, 1)
            soft_one_hot = one_hot.type(torch.FloatTensor).cuda()
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

            return z_q

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        B, _, _, _ = x.shape
        # patch_size = 16
        # n_patch = H // patch_size
        vq_size = 16
        L = vq_size * vq_size

        # output = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # _, 3*16*16, 16*16
        # output = output.permute(0, 2, 1)

        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask = mask.view(B, 1, vq_size, vq_size).to(memory_format=torch.contiguous_format)
        # output = output.permute(0, 2, 1)
        # output = F.fold(output, output_size=(H, W), kernel_size=(patch_size, patch_size), stride=patch_size)

        return mask





class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def gumbel_softmax(self, logits, tau=1, hard=False, dim=-1, gumbel_noise=1.0):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)

        # def log(t, eps=1e-6):
        #     return torch.log(t + eps)
        # eps = 1e-6
        # u = torch.empty_like(logits).uniform_(0, 1)
        # gumbels2 = -log(-log(u, eps), eps)
        # # print (gumbels)
        #
        # N = 1024
        # x = np.arange(1024)
        # y1 = logits[0, :, 2, 2].cpu().detach().numpy()
        # y2 = gumbels[0, :, 2, 2].cpu().detach().numpy()
        # y3 = gumbels2[0, :, 2, 2].cpu().detach().numpy()
        # # y = np.random.rand(N)
        # plt.scatter(x, y1)
        # plt.scatter(x, y2)
        # plt.scatter(x, y3)
        # plt.savefig('gumbel.png')
        # 1/0
        #
        #
        # print (gumbels[0, :, 2, 2].min(), gumbels[0, :, 2, 2].max(), logits[0, :, 2, 2].min(), logits[0, :, 2, 2].max())
        # 1/0

        gumbels = (logits + gumbels*gumbel_noise) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, z=None, temp=None, return_logits=False, provided_idx=None):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work

        if provided_idx == None:

            hard = self.straight_through if self.training else True
            temp = self.temperature if temp is None else temp

            logits = self.proj(z)
            if self.remap is not None:
                # continue only with used logits
                full_zeros = torch.zeros_like(logits)
                logits = logits[:, self.used, ...]

            # soft_one_hot = self.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
            # 1/0


            if self.training:
                soft_one_hot = self.gumbel_softmax(logits, tau=temp, dim=1, hard=hard, gumbel_noise=1.0)
                # logits = logits.softmax(dim=1)  #  gumbel_norm
                # soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
            else:
                # print('*******test********')
                y_soft = logits.softmax(dim=1)
                index = y_soft.max(dim=1, keepdim=True)[1]
                soft_one_hot = torch.zeros_like(logits).scatter_(1, index, 1.0)

            if self.remap is not None:
                # go back to all entries but unused set to zero
                full_zeros[:, self.used, ...] = soft_one_hot
                soft_one_hot = full_zeros
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

            # + kl divergence to the prior loss
            qy = F.softmax(logits, dim=1)
            diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

            ind = soft_one_hot.argmax(dim=1)
            if self.remap is not None:
                ind = self.remap_to_used(ind)
            if self.use_vqinterface:
                if return_logits:
                    return z_q, diff, (None, None, ind), logits
                return z_q, diff, (None, None, ind)
            return z_q, diff, ind

        else:
            b = provided_idx.shape[0]
            one_hot = torch.nn.functional.one_hot(provided_idx.to(torch.int64), 1024)
            # print(one_hot.shape)
            one_hot = one_hot.view(b, 1024, 1, 1)
            soft_one_hot = one_hot.type(torch.FloatTensor).cuda()
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

            return z_q

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q




class ArgmaxQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def argmax_softmax(self, logits, tau=1, dim=-1):
        logits = logits / tau
        y_soft = logits.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def forward(self, z=None, temp=None, return_logits=False, provided_idx=None):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work

        if provided_idx == None:

            hard = self.straight_through if self.training else True
            temp = self.temperature if temp is None else temp

            logits = self.proj(z)
            if self.remap is not None:
                # continue only with used logits
                full_zeros = torch.zeros_like(logits)
                logits = logits[:, self.used, ...]

            if self.training:
                # soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
                soft_one_hot = self.argmax_softmax(logits, tau=temp, dim=1)
            else:
                # print('*******test********')
                y_soft = logits.softmax(dim=1)
                index = y_soft.max(dim=1, keepdim=True)[1]
                soft_one_hot = torch.zeros_like(logits).scatter_(1, index, 1.0)

            if self.remap is not None:
                # go back to all entries but unused set to zero
                full_zeros[:, self.used, ...] = soft_one_hot
                soft_one_hot = full_zeros
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

            # + kl divergence to the prior loss
            # qy = F.softmax(logits, dim=1)
            # diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
            diff = torch.zeros(1)

            ind = soft_one_hot.argmax(dim=1)
            if self.remap is not None:
                ind = self.remap_to_used(ind)
            if self.use_vqinterface:
                if return_logits:
                    return z_q, diff, (None, None, ind), logits
                return z_q, diff, (None, None, ind)
            return z_q, diff, ind

        else:
            b = provided_idx.shape[0]
            one_hot = torch.nn.functional.one_hot(provided_idx.to(torch.int64), 1024)
            # print(one_hot.shape)
            one_hot = one_hot.view(b, 1024, 1, 1)
            soft_one_hot = one_hot.type(torch.FloatTensor).cuda()
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

            return z_q

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q



