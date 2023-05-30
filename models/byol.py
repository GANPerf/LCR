import copy
import random 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import transforms 
from math import pi, cos
from torchvision.models import resnet50
from collections import OrderedDict
HPS = dict(
    max_steps=int(1000. * 1281167 / 4096), # 1000 epochs * 1281167 samples / batch size = 100 epochs * N of step/epoch
    # = total_epochs * len(dataloader) 
    mlp_hidden_size=4096,
    projection_size=256,
    base_target_ema=4e-3,
    optimizer_config=dict(
        optimizer_name='lars', 
        beta=0.9, 
        trust_coef=1e-3, 
        weight_decay=1.5e-6,
        exclude_bias_from_adaption=True),
    learning_rate_schedule=dict(
        base_learning_rate=0.2,
        warmup_steps=int(10.0 * 1281167 / 4096), # 10 epochs * N of steps/epoch = 10 epochs * len(dataloader)
        anneal_schedule='cosine'),
    batchnorm_kwargs=dict(
        decay_rate=0.9,
        eps=1e-5), 
    seed=1337,
)



def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, HPS['mlp_hidden_size']),
            nn.BatchNorm1d(HPS['mlp_hidden_size'], eps=HPS['batchnorm_kwargs']['eps'], momentum=1-HPS['batchnorm_kwargs']['decay_rate']),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(HPS['mlp_hidden_size'], HPS['projection_size'])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projector = MLP(backbone.output_dim)
        self.online_encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(HPS['projection_size'])
        #raise NotImplementedError('Please put update_moving_average to training')


        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool14 = nn.AvgPool2d(14, stride=1)



    def target_ema(self, k, K, base_ema=HPS['base_target_ema']):
        # tau_base = 0.996 
        # base_ema = 1 - tau_base = 0.996 
        return 1 - base_ema * (cos(pi*k/K)+1)/2 
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self): #, global_step, max_steps
        #tau = self.target_ema(global_step, max_steps)
        tau = 0.996
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, x1, x2):
        f_o, h_o = self.online_encoder, self.online_predictor
        f_t      = self.target_encoder


        x = self.online_encoder[0].conv1(x1)
        x = self.online_encoder[0].bn1(x)
        x = self.online_encoder[0].relu(x)
        x = self.online_encoder[0].layer1(x)
        x = self.online_encoder[0].layer2(x)
        x = self.online_encoder[0].layer3(x)
        feat_map1 = self.online_encoder[0].layer4(x)
        x = self.online_encoder[0].avgpool(feat_map1)
        x = x.reshape(x.size(0), -1)
        z1_o = self.online_encoder[1](x)

        x = self.online_encoder[0].conv1(x2)
        x = self.online_encoder[0].bn1(x)
        x = self.online_encoder[0].relu(x)
        x = self.online_encoder[0].layer1(x)
        x = self.online_encoder[0].layer2(x)
        x = self.online_encoder[0].layer3(x)
        feat_map2 = self.online_encoder[0].layer4(x)
        x = self.online_encoder[0].avgpool(feat_map2)
        x = x.reshape(x.size(0), -1)
        z2_o = self.online_encoder[1](x)



        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            self.update_moving_average()
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2


        grad_wrt_act1 = torch.autograd.grad(outputs=L, inputs=feat_map1,
                                           grad_outputs=torch.ones_like(L), retain_graph=True,
                                           allow_unused=True)[0]

        gradcam = torch.relu((feat_map1 * grad_wrt_act1).sum(dim=1))

        featcov16 = self.conv16(feat_map1)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)
        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img)) #batch*7*7

        return {'loss': L, 'attmap': att_max, 'gradcam': gradcam}



    def inference(self, img):
        x = self.online_encoder[0].conv1(img)
        x = self.online_encoder[0].bn1(x)
        x = self.online_encoder[0].relu(x)
        x = self.online_encoder[0].layer1(x)
        x = self.online_encoder[0].layer2(x)
        x = self.online_encoder[0].layer3(x)
        feat_map1 = self.online_encoder[0].layer4(x)
        x = self.online_encoder[0].avgpool(feat_map1)
        x = x.reshape(x.size(0), -1)

        featcov16 = self.conv16(feat_map1)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)



        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        img = att_max[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = feat_map1.cuda() * img.cuda()
        aa = self.avgpool14(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)


        return x, bp_out_feat

    

if __name__ == "__main__":
    pass
