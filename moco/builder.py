
import torch
import torch.nn as nn
from resnet_output import resnet50
import numpy as np


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = resnet50(pretrained=True, num_classes=1000)
        # self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = resnet50(pretrained=True, num_classes=1000)
        # self.encoder_k = base_encoder(num_classes=dim)

        self.encoder_q.fc = nn.Linear(2048, dim)
        self.encoder_k.fc = nn.Linear(2048, dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim,
                                                  K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue_max
        self.register_buffer("queue_max", torch.randn(dim,
                                                  K))
        self.queue_max = nn.functional.normalize(self.queue_max, dim=0)

        self.register_buffer("queue_ptr_max", torch.zeros(1, dtype=torch.long))
        # add projection

        self.bilinear = 32

        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)

        self.bn16 = nn.BatchNorm2d(self.bilinear)

        self.conv16_2 = nn.Conv2d(self.bilinear, 1, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16_2 = nn.BatchNorm2d(1)


        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(2048, dim)
        self.qmax = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), self.fc)
        self.kmax = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), self.fc)



    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.qmax.parameters(), self.kmax.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _dequeue_and_enqueue_max(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_max)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_max[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_max[0] = ptr

    def max_mask(self, featmap, indx):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)



        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        img = att_max[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)



        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat_max = nn.functional.normalize(bp_out_feat, dim=1)



        return bp_out_feat_max, att_max

    def feat_bilinear(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        feat_matrix = torch.zeros(featcov16.size(0), self.bilinear, 2048)
        for i in range(self.bilinear):
            matrix = featcov16[:, i, :, :]
            matrix = matrix[:, None, :, :]
            matrix = matrix.repeat(1, 2048, 1, 1)
            PFM = featmap * matrix
            aa = self.avgpool(PFM)

            feat_matrix[:, i, :] = aa.view(aa.size(0), -1)

        bp_out_feat = feat_matrix.view(feat_matrix.size(0), -1)

        return bp_out_feat

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, epoch, iter, indx):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, _, featmap = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)


        # max bilinear q
        q_max, att_max = self.max_mask(featmap, indx)
        embedding_q = self.qmax(q_max.cuda())
        q_max_proj = nn.functional.normalize(embedding_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, _, featmap_k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # max bilinear k
            k_max, _ = self.max_mask(featmap_k, indx)
            embedding_k = self.kmax(k_max.cuda())
            k_max_proj = nn.functional.normalize(embedding_k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle).detach()

            k_max_proj = self._batch_unshuffle_ddp(k_max_proj, idx_unshuffle).detach()

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        #max
        l_pos_max = torch.einsum('nc,nc->n', [q_max_proj, k_max_proj]).unsqueeze(-1)
        l_neg_max = torch.einsum('nc,ck->nk', [q_max_proj, self.queue_max.clone().detach()])

        logits_max = torch.cat([l_pos_max, l_neg_max], dim=1) / self.T
        labels_max = torch.zeros(logits_max.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        self._dequeue_and_enqueue_max(k_max_proj)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        criterion = nn.CrossEntropyLoss()
        CE = criterion(logits, labels)

        grad_wrt_act1 = torch.autograd.grad(outputs=CE, inputs=featmap,
                                            grad_outputs=torch.ones_like(CE), retain_graph=True,
                                            allow_unused=True)[0]

        gradcam = torch.relu((featmap * grad_wrt_act1).sum(dim=1))


        return logits, labels, logits_max, labels_max, self.encoder_q, featcov16, featmap, att_max, gradcam

    def inference(self, img):
        projfeat, feat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)



        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        feat = nn.functional.normalize(feat, dim=1)

        return projfeat, feat, featcov16, bp_out_feat


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
