import torch
import torch.nn as nn
import torch.nn.functional as F
#from nystrom_attention import NystromAttention
#from torch.autograd import Variable


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class) #输出各个示例的预测概率

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class MHSA(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        #x = torch.unsqueeze(x,dim=-1)
        #print(x.shape)
        B, N = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)

        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N)
        return v

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):  # K, L, N
        super(BClassifier, self).__init__()

        self.L = 64
        self.num_heads = 1
        self.k = 1

        self.DPL = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.L)
        )
        self.enc = nn.Sequential(
            nn.Linear(input_size, self.L),
        )
        self.attn = MHSA(self.num_heads, self.L)
        #self.classifier = nn.Sequential(nn.Linear(self.L , output_class))
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(self.L, output_class, kernel_size=self.L)

    def recalib(self, inputs):
        #c = self.enc(inputs.view(inputs.shape[0], -1))
        #a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
        # print(a1.shape)
        #_, m_indices = torch.sort(c, 0, descending=True)
        #m_feats = torch.index_select(inputs, dim=0,
                                     #index=m_indices[0, :])
        Q = torch.mean(inputs, dim=1, keepdim=True) # N*1
        v = torch.std(inputs, dim=1, unbiased=False, keepdim=True)

        return  Q, v


    def forward(self, feats, c):
        device = feats.device
        #print("原始feats：",feats)
        feats1 = self.enc(feats)
        Q,v = self.recalib(feats)
        feats = torch.relu((feats - Q)/v)# N*1024
        #print("校准后feats:", feats)
        feats = self.DPL(feats)  # feats维度 N*64 N示例数量
        #print("feats的大小",feats)
        A_unnorm = self.attn(feats)  # 未归一化的A维度：N * K，即 N*1
        # print("A_unmorm的大小：", A_unnorm.shape)
        A = torch.transpose(A_unnorm, 1, 0)  # 维度交换后的A维度：K * N
        A = F.softmax(A, dim=1)
        # print("A的大小", A.shape)
        M = torch.mm(A, feats1)  # 得分和特征相乘得到融合的包特征
        M = torch.unsqueeze(M,dim = 0)
        #print("M.shape:",M.shape)
        Y_prob = self.fcc(M)

        return Y_prob, A, M


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B
