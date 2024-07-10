import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CombineGraph(Module):
    #adj 相邻
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.nonhybrid = opt.nonhybrid
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        #n_iter迭代次数
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.beta = opt.conbeta
        # Aggregator
        self.local_agg = LocalAggregator(opt,self.dim, self.opt.alpha,self.num_node ,dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        # self.pos_embedding = nn.Embedding(200, self.dim)
        self.pos_embedding = nn.Embedding(3000, self.dim)

        # Parameters
        self.linear_one = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_two = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_three = nn.Linear(self.dim, self.dim, bias=False)

        #local
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)
        #global
        self.w_11 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_22 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu11 = nn.Linear(self.dim, self.dim)
        self.glu22 = nn.Linear(self.dim, self.dim, bias=False)
        self.bias_list = nn.Parameter(torch.empty(size=(1, self.dim), dtype=torch.float), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        # pos = trans_to_cuda(score(sess_emb_hgnn, sess_emb_lgcn))
        # neg1 = trans_to_cuda(score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn)))
        # one  = trans_to_cuda(torch.ones(neg1.shape[0]))
        # con_loss = trans_to_cuda(torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1)))))

        pos = trans_to_cuda(score(sess_emb_hgnn, sess_emb_lgcn))
        neg1 = trans_to_cuda(score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn)))
        one = trans_to_cuda(torch.ones(neg1.shape[0]))
        con_loss = trans_to_cuda(
            torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1)))))

        # margin = 0.2  # Contrastive损失的边界值
        # neg2 = trans_to_cuda(score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn)))  # 计算第二个负样本
        # con_loss = torch.relu(pos - neg1 + margin).mean() + torch.relu(pos - neg2 + margin).mean()
        # con_loss = trans_to_cuda(con_loss)

        # hinge_pos = torch.clamp(1 - pos, min=0)
        # hinge_neg = torch.clamp(1 + neg1, min=0)
        # con_loss = hinge_pos.mean() + hinge_neg.mean()

        # logits = pos - neg1
        # eps = 1e-6
        # logits_clamped = torch.clamp(logits, min=eps)
        #
        # # 计算N-Pairs Loss

        # loss_per_pair = -torch.log(logits_clamped)  # 对每个样本对应用负对数似然损失
        # con_loss = loss_per_pair.mean()  # 计算平均损失

        return self.beta*con_loss
        # return 0.02*con_loss


    def compute_scores(self, local_hidden,global_hidden, mask):
        # local
        local_hidden = F.dropout(local_hidden, self.dropout_local, training=self.training)
        mask = mask.float().unsqueeze(-1)
        batch_size = local_hidden.shape[0]
        len = local_hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]

        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(local_hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        nh = torch.matmul(torch.cat([pos_emb, local_hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        # select = torch.sum(beta * local_hidden, 1)
        h_local = torch.sum(beta * local_hidden, 1)
        # select = torch.sum(beta * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # if not self.nonhybrid:
        #     select = self.linear_transform(torch.cat([select, nh], 1))


        # global
        global_hidden = F.dropout(global_hidden, self.dropout_global, training=self.training)
        # global_hidden_channel = torch.chunk(global_hidden, self.channels, dim=2)
        # mask = mask.float().unsqueeze(-1)
        # batch_size = global_hidden.shape[0]
        len_ = global_hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len_]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        # session_infos = []
        # for global_seq in global_hidden_channel:
        #     hs = torch.sum(global_seq * mask, -2) / torch.sum(mask, 1)
        #     hs = hs.unsqueeze(-2).repeat(1, len_, 1)
        #     nh = torch.matmul(torch.cat([pos_emb, global_seq], -1), self.w_11)
        #     nh = torch.tanh(nh)
        #     nh = torch.sigmoid(self.glu11(nh) + self.glu22(hs))
        #     beta = torch.matmul(nh, self.w_22)
        #     beta = beta * mask
        #     h_global = torch.sum(beta * global_seq, 1)
        #     session_infos.append(h_global)
        # session_infos = torch.cat([session_infos[i] for i in range(len(session_infos))], dim=1)
        # score

        hs = torch.sum(global_hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len_, 1)
        nh = torch.matmul(torch.cat([pos_emb, global_hidden], -1), self.w_11)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu11(nh) + self.glu22(hs))
        beta = torch.matmul(nh, self.w_22)
        beta = beta * mask
        h_global = torch.sum(beta * global_hidden, 1)


        b = self.embedding.weight[1:]  # n_nodes x latent_size
        conloss = self.SSL(h_local, h_global)
        output = h_local + h_global + self.bias_list

        # output = output.unsqueeze(1).expand(-1, len, -1)
        # mask = mask.squeeze(2)
        # ht = output[torch.arange(mask.shape[0]).long(), (torch.sum(mask, 1) - 1).long()]
        #
        # q1 = self.linear_one(output)
        # q2 = self.linear_two(ht).view(ht.shape[0], 1, ht.shape[1])
        # alpha = self.linear_three(torch.sigmoid(q1 + q2))
        # a = torch.sum(alpha * output * mask.view(mask.shape[0], -1, 1).float(), 1)


        scores = torch.matmul(output, b.transpose(1, 0))
        return scores,conloss

    def minmaxscaler(self, time_data, max, min):

        return (time_data - min) / (max - min)

    def forward(self, inputs, adj, mask_item, item,input_times):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        # h_local = self.local_agg(h, adj, input_times, mask_item)
        h_local = self.local_agg(h, adj, input_times, mask_item)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        # output = h_local + h_global
        output = [h_local, h_global]
        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# def forward(model, data):
#
#     alias_inputs, adj, items, mask, targets, inputs = data
#     alias_inputs = trans_to_cuda(alias_inputs).long()
#     items = trans_to_cuda(items).long()
#     adj = trans_to_cuda(adj).float()
#     mask = trans_to_cuda(mask).long()
#     inputs = trans_to_cuda(inputs).long()
#
#     hidden = model(items, adj, mask, inputs)
#     get = lambda index: hidden[index][alias_inputs[index]]
#     seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
#     return targets, model.compute_scores(seq_hidden, mask)
def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs, input_times, max_time, min_time = data
    # length_=len(items[1])
    # time=(input_times-min_time)/(max_time-min_time)

    max_time_list=max_time.tolist()
    max_time_int = int(max_time_list[0])
    input_times = input_times / max_time_int


    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).long()
    mask = trans_to_cuda(mask).long()
    input_times = trans_to_cuda(torch.Tensor(input_times).float())
    inputs = trans_to_cuda(inputs).long()


    hidden= model.forward(items, adj, mask,inputs,input_times)
    get_local = lambda index: hidden[0][index][alias_inputs[index]]
    seq_hidden_local = torch.stack([get_local(i) for i in torch.arange(len(alias_inputs)).long()])
    get_global = lambda index: hidden[1][index][alias_inputs[index]]
    seq_hidden_global = torch.stack([get_global(i) for i in torch.arange(len(alias_inputs)).long()])
    scores,conloss=model.compute_scores(seq_hidden_local,seq_hidden_global, mask)
    return targets, scores,conloss

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()

    total_loss = 0.0
    total_conloss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores,conloss = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + conloss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        total_conloss += conloss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tconloss:\t%.3f' % total_conloss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores ,conloss = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result
