'''
author: lxr  Li Xinrui
data: 2019-10-20
task: complete islandloss
department: UESTC IN CHINA

Lisland=Lc+λ1∑cj∈N∑ck∈N,ck≠cjck⋅cj||ck||2||cj||2+1

'''

import torch
import torch.nn as nn
from torch.autograd.function import Function


class IsLandLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lamda=0.5, size_average=True):
        super(IsLandLoss, self).__init__()
        # 初始化中心,初始化参数, 2个点 2维
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # 初始化island损失函数
        self.islandlossfunc = IslandlossFunc.apply
        # 特征点数
        self.feat_dim = feat_dim
        # lamda
        self.lamda = lamda
        # 是否对batch中心损失进行平均，通常都要true
        self.size_average = size_average


    def forward(self, label, feat):
        # 获取batch_size
        batch_size = feat.size(0)
        # 重新规整特征点的形状
        feat = feat.view(batch_size, -1)
        # 检查中心和特征维度
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        # 清楚内存么  返回的是和feat 相同的 dtype 和 device 但不同形状 填充 32
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)

        lamda_tensor = feat.new_empty(1).fill_(self.lamda)

        loss = self.islandlossfunc(feat, lamda_tensor, label, self.centers, batch_size_tensor)

        return loss


class IslandlossFunc(Function):

    @staticmethod
    def forward(ctx, feature, lamda, label, centers, batch_size):
        ctx.save_for_backward(feature, lamda, label, centers, batch_size)
        # 在相应的维度去筛选，重组获取batch中心,写的好
        # center 是(7, 2)  label (32)根据标签获取每个 样本的中心
        # label tensor([0, 1, 2, 3, ... , 10])
        centers_batch = centers.index_select(0, label.long())
        center_loss = (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

        N = centers.size(0)
        island_loss = centers.new_zeros(1)
        for j in range(N):
            for k in range(N):
                if k != j:
                    cj = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(j)).squeeze()
                    ck = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(k)).squeeze()
                    # 求余弦距离
                    cos_distance = torch.cosine_similarity(cj, ck, dim=0) + centers.new_ones(1)
                    # cos_distance = cos_distance.index_select(0)
                    island_loss.add_(cos_distance)

        return center_loss + lamda * island_loss


    @staticmethod
    def backward(ctx, grad_output):

        # 获取反向传播值
        feature, lamda, label, centers, batch_size = ctx.saved_tensors
        #求每个batch 对应的中心
        centers_batch = centers.index_select(0, label.long())

        # 求解特征梯度
        diff = centers_batch - feature

        # 求解centerloss各类中心参数的梯度
        # init every iteration
        # [1, 1, 1, 1, 1, 1, 1]
        counts = centers.new_ones(centers.size(0))
        # [1, 1, 1, ... , 1, 1]
        ones = centers.new_ones(label.size(0))
        # 梯度 [7, 2]
        grad_centers = centers.new_zeros(centers.size())
        # 统计每个类在这个batch的个数
        counts = counts.scatter_add_(0, label.long(), ones)
        # 统计每个类的距离和
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        # count.view [1, 2, 3, 4]   [[1], [2], [3], [4]]
        grad_centers = grad_centers/counts.view(-1, 1)

        # 求解island中心参数的梯度
        N = centers.size(0)
        # 求L2范数
        l2_centers = torch.norm(centers, 2, 1).view(N, -1)
        # 梯度
        grad_centers_il = torch.zeros_like(centers)
        for j in range(N):
            for k in range(N):
                if k != j:
                    ck = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(k)).squeeze()
                    cj = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(j)).squeeze()
                    l2ck = l2_centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(k)).squeeze()
                    l2cj = l2_centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(j)).squeeze()

                    val = ck / (l2ck * l2cj) - (ck.mul(cj) / (l2ck * l2cj.pow(3))).mul(cj)

                    grad_centers_il[j, :].add_(val)

        return - grad_output * diff / batch_size, None, None, grad_centers / batch_size + grad_centers_il * lamda / (N -1), None




# featrues = torch.randn(32, 2)
# lables = torch.ones(32)
# islandloss = IsLandLoss(7, 2, lamda=0.5, size_average=True)
# loss = islandloss(lables, featrues)
# loss.backward()
# a = 0
# j = [1.0, 2.0]
# a = torch.tensor(j)
# b = torch.tensor([1.0, 1.0])
# cos_distance = torch.cosine_similarity(a, b, dim=0)
# print(cos_distance)



