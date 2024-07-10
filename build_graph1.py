import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='diginetica/Tmall/Nowplaying/yoochoose1_4/yoochoose1_64')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

# seq = pickle.load(open('diginetica/all_train_seq.txt', 'rb'))
seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
elif dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 61212
elif dataset == "yoochoose1_4":
    num = 29618
elif dataset == "yoochoose1_64":
    num = 55898
else:
    num = 3

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]


for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

max_num = 0
num_set = set()
for tup in relation:
    if tup[0] not in num_set:
        num_set.add(tup[0])
    max_num = max(max_num, max(tup))
    tup[1] = str(tup[1])
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

print(len(num_set), max_num,num)
weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

# pickle.dump(adj, open('diginetica/adj_' + str(sample_num) + '.pkl', 'wb'))
# pickle.dump(weight, open('diginetica/num_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))