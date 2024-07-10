import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import math
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, opt,dim, alpha, num_node,dropout=0., name=None, kernel_type='exp*-1'):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.batch_size = opt.batch_size
        # self.position_embed = PositionalEncoding(self.dropout, )
        self.num_node = num_node
        self.embedding = nn.Embedding(self.num_node, self.dim)
        self.reset_parameters()

        self.kernel_num = self.num_node
        parts = kernel_type.split('-')
        kernel_types = []
        self.params = []
        for i in range(len(parts)):
            pi = parts[i]
            if pi in {'exp', 'exp*', 'log', 'lin', 'exp^', 'exp*^', 'log^', 'lin^', 'ind', 'const', 'thres'}:
                if pi.endswith('^'):
                    var = (nn.Parameter(torch.rand(1, requires_grad=True) * 5 + 10),
                           nn.Parameter(torch.rand(1, requires_grad=True)))
                    kernel_types.append(pi[:-1])
                else:
                    var = (nn.Parameter(torch.rand(1, requires_grad=True) * 0.01),
                           nn.Parameter(torch.rand(1, requires_grad=True)))
                    kernel_types.append(pi)

                self.register_parameter(pi + str(len(self.params)) + '_0', var[0])
                self.register_parameter(pi + str(len(self.params)) + '_1', var[1])

                self.params.append(var)

            elif pi.isdigit():
                val = int(pi)
                if val > 1:
                    pi = parts[i - 1]
                    for j in range(val - 1):
                        if pi.endswith('^'):
                            var = (nn.Parameter(torch.rand(1, requires_grad=True) * 5 + 10),
                                   nn.Parameter(torch.rand(1, requires_grad=True)))
                            kernel_types.append(pi[:-1])
                        else:
                            var = (nn.Parameter(torch.rand(1, requires_grad=True) * 0.01),
                                   nn.Parameter(torch.rand(1, requires_grad=True)))
                            kernel_types.append(pi)

                        self.register_parameter(pi + str(len(self.params)) + '_0', var[0])
                        self.register_parameter(pi + str(len(self.params)) + '_1', var[1])

                        self.params.append(var)


            else:
                print('no matching kernel ' + pi)
        self.kernel_num = len(kernel_types)
        print(kernel_types, self.params)

        def decay_constructor(t):
            kernels = []
            for i in range(self.kernel_num):
                pi = kernel_types[i]
                if pi == 'log':
                    kernels.append(torch.mul(self.params[i][0], torch.log1p(t)) + self.params[i][1])
                elif pi == 'exp':
                    kernels.append(1000 * torch.exp(torch.mul(self.params[i][0], torch.neg(t))) + self.params[i][1])
                elif pi == 'exp*':
                    kernels.append(torch.mul(self.params[i][0], torch.exp(torch.neg(t))) + self.params[i][1])
                elif pi == 'lin':
                    kernels.append(self.params[i][0] * t + self.params[i][1])
                elif pi == 'ind':
                    kernels.append(t)
                elif pi == 'const':
                    kernels.append(torch.ones(t.size()))
                elif pi == 'thres':
                    kernels.append(torch.reciprocal(1 + torch.exp(-self.params[i][0] * t + self.params[i][1])))

            return torch.stack(kernels, dim=2)

        self.decay = decay_constructor

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hidden, adj, input_times,mask_item=None):
        time_embedding = self.decay(input_times)
        # time_embedding = time_embedding * math.sqrt(self.batch_size) + self.position_embed(time_embedding,1000)
        # time_embedding = time_embedding * math.sqrt(self.batch_size)

        # time_embedding = time_embedding.view(len(input_times), len(input_times[0]), 1)


        # hidden = torch.cat([hidden, time_embedding], 2)
        # hidden = hidden * time_embedding
        hidden = hidden[:, :, :-1]


        hidden1 = torch.cat((hidden, time_embedding), dim=2)

        #
        # hidden = hidden[:, :, :-1]

        h = hidden1
        batch_size = h.shape[0]
        N = h.shape[1]

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self,dim, dropout, act=torch.relu, name=None,kernel_type='exp-1'):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim
        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))
    ######
    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output

# class PositionalEncoding(nn.Module):
#     def __init__(self, dropout,):
#         super(PositionalEncoding, self).__init__()
#
#     def forward(self, X, max_seq_len=None):
#         if max_seq_len is None:
#             max_seq_len = X.size()[1]
#         # X为wordEmbedding的输入,PositionalEncoding与batch没有关系
#         # max_seq_len越大,sin()或者cos()的周期越小,同样维度
#         # 的X,针对不同的max_seq_len就可以得到不同的positionalEncoding
#         assert X.size()[1] <= max_seq_len
#         # X的维度为: [batch_size, seq_len, embed_size]
#         # 其中: seq_len = l, embed_size = d
#         l, d = X.size()[1], X.size()[-1]
#         # P_{i,2j}   = sin(i/10000^{2j/d})
#         # P_{i,2j+1} = cos(i/10000^{2j/d})
#         # for i=0,1,...,l-1 and j=0,1,2,...,[(d-2)/2]
#         max_seq_len = int((max_seq_len//l)*l)
#         P = np.zeros([1, l, d])
#         # T = i/10000^{2j/d}
#         T = [i*1.0/10000**(2*j*1.0/d) for i in range(0, max_seq_len, max_seq_len//l) for j in range((d+1)//2)]
#         T = np.array(T).reshape([l, (d+1)//2])
#         if d % 2 != 0:
#             P[0, :, 1::2] = np.cos(T[:, :-1])
#         else:
#             P[0, :, 1::2] = np.cos(T)
#         P[0, :, 0::2] = np.sin(T)
#         return torch.tensor(P, dtype=torch.float, device=X.device)