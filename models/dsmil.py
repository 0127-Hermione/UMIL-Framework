import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1),c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU()) #模型采用非线性的情况
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # feats：N x 512, c: N x 1 ?  (N:示例数量，K: 特征输出维度，c:类别得分，由于二分类，设置c的维度为N*1)
        device = feats.device
        feats = self.lin(feats) #feats经过Linear(512,512), ReLU();  feats维度仍为N*512
        V = self.v(feats) # feats经过Dropout(), Linear(512,512);  N x V (V维度：N*512), unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # feats经过Linear(512,128), Tanh(); 维度：N x Q (Q维度：N*128), unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # instance维度上对类别分数排序，从大到小, m_indices in shape N x C (N*1?)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # 挑选关键instance，即得分最高的instance， m_feats维度 C x K (1 * 512?)
        q_max = self.q(m_feats) # 计算关键instance的查询向量q_max, q_max维度： C x Q( 1 * 128？)
        A = torch.mm(Q, q_max.transpose(0, 1)) # 计算每个instance到关键instance的内积, A维度：N x 1, 每列是未经过归一化的注意力分数
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # 归一化注意力分数
        B = torch.mm(A.transpose(0, 1), V) # 计算bag表示，compute bag representation, B：1 x 512
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1 计算bag得分
        C = C.view(1, -1)

        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B
        