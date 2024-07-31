import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive, i0e  # 导入 i0e 函数
import numpy as np

def miller_recurrence(nu, x):
    I_n = torch.ones(1, dtype=torch.float64).cuda()
    I_n1 = torch.zeros(1, dtype=torch.float64).cuda()

    Estimat_n = [nu, nu+1]
    scale0 = 0 
    scale1 = 0 
    scale = 0

    for i in range(2*nu, 0, -1):
        I_n_tem, I_n1_tem = 2*i/x*I_n + I_n1, I_n
        if torch.isinf(I_n_tem).any():
            I_n1 /= I_n
            scale += torch.log(I_n)
            if i >= (nu+1):
                scale0 += torch.log(I_n)
                scale1 += torch.log(I_n)
            elif i == nu:
                scale0 += torch.log(I_n)

            I_n = torch.ones(1, dtype=torch.float64).cuda()
            I_n, I_n1 = 2*i/x*I_n + I_n1, I_n
        else:
            I_n, I_n1 = I_n_tem, I_n1_tem

        if i == nu:
            Estimat_n[0] = I_n1
        elif i == (nu+1):
            Estimat_n[1] = I_n1

        # 使用 scipy.special.i0e 替代 torch.special.i0e
    ive0 = torch.from_numpy(i0e(x.cpu().numpy())).cuda()

    Estimat_n[0] = torch.log(ive0) + torch.log(Estimat_n[0]) - torch.log(I_n) + scale0 - scale
    Estimat_n[1] = torch.log(ive0) + torch.log(Estimat_n[1]) - torch.log(I_n) + scale1 - scale

    return Estimat_n[0], Estimat_n[1]

class LogRatioC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, p, logc):
        nu, nu1 = miller_recurrence((p/2 - 1).int(), k.double())
        tensor = nu + k - (p/2 - 1) * torch.log(k+1e-20) - logc
        ctx.save_for_backward(torch.exp(nu1 - nu))
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[0]
        grad[grad > 1.0] = 1.0
        grad *= grad_output
        return grad, None, None

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        self.Ave = F.normalize(torch.randn(class_num, feature_num), dim=1) * 0.9
        self.Amount = torch.zeros(class_num)
        self.kappa = torch.ones(class_num) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num/2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(self.kappa.device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)

        if torch.cuda.is_available():
            self.Ave = self.Ave.cuda()
            self.Amount = self.Amount.cuda()
            self.kappa = self.kappa.cuda()
            self.logc = self.logc.cuda()

    def reset(self):
        self.Ave = F.normalize(torch.randn(self.class_num, self.feature_num), dim=1) * 0.9
        self.Amount = torch.zeros(self.class_num)
        self.kappa = torch.ones(self.class_num) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num/2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(self.kappa.device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)
        if torch.cuda.is_available():
            self.Ave = self.Ave.cuda()
            self.Amount = self.Amount.cuda()
            self.kappa = self.kappa.cuda()
            self.logc = self.logc.cuda()

    def reload_memory(self):
        if torch.cuda.is_available():
            self.Ave = self.Ave.cuda()
            self.Amount = self.Amount.cuda()
            self.kappa = self.kappa.cuda()
            self.logc = self.logc.cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        onehot = torch.zeros(N, C)
        if torch.cuda.is_available():
            onehot = onehot.cuda()

        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA
        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)
        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)

    def update_kappa(self, kappa_inf=False):
        R = torch.linalg.norm(self.Ave, dim=1)
        self.kappa = self.feature_num * R / ( 1 - R**2)
        self.kappa[self.kappa > 1e5] = 1e5
        self.kappa[self.kappa < 0] = 1e5

        nu, _ = miller_recurrence(torch.tensor(self.feature_num/2 - 1).int(), self.kappa.double())
        self.logc = nu + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-20)

class ProCoLoss(nn.Module):
    def __init__(self, contrast_dim, temperature=1.0, num_classes=1000):
        super(ProCoLoss, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_num = contrast_dim
        self.estimator_old = EstimatorCV(self.feature_num, num_classes)
        self.estimator = EstimatorCV(self.feature_num, num_classes)

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        if torch.cuda.is_available():
            self.weight = self.weight.to(torch.device('cuda'))

    def reload_memory(self):
        self.estimator_old.reload_memory()
        self.estimator.reload_memory()

    def _hook_before_epoch(self, epoch, epochs):
        self.estimator_old.Ave = self.estimator.Ave
        self.estimator_old.Amount = self.estimator.Amount
        self.estimator_old.kappa = self.estimator.kappa
        self.estimator_old.logc = self.estimator.logc
        self.estimator.reset()

    def forward(self, features, labels=None, sup_logits=None, args=None):
        batch_size = features.size(0)
        N = batch_size

        if labels is not None:
            self.estimator_old.update_CV(features.detach(), labels)
            self.estimator.update_CV(features.detach(), labels)
            self.estimator_old.update_kappa()

        Ave = self.estimator_old.Ave.detach()
        Ave_norm = F.normalize(Ave, dim=1)
        logc = self.estimator_old.logc.detach()
        kappa = self.estimator_old.kappa.detach()

        tem = kappa.reshape(-1, 1) * Ave_norm
        tem = tem.unsqueeze(0) + features[:N].unsqueeze(1) / self.temperature
        kappa_new = torch.linalg.norm(tem, dim=2)

        contrast_logits = LogRatioC.apply(kappa_new, torch.tensor(self.estimator.feature_num), logc)

        return contrast_logits
