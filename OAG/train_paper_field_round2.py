import sys
import torch

print('cuda status {}'.format(torch.cuda.is_available()))
from pyHGT.data_r2 import *
from pyHGT.model_r2 import *
from warnings import filterwarnings

filterwarnings("ignore")

import argparse

fopParent = '../../dataPapers/'
# fopInputDir=fopParent+'HGT_data/MAG_0919_CS/'
fopOutputDir = fopParent + 'HGT_data/bag_output/'
fopModelDir = fopParent + 'HGT_data/bag_model/'

parser = argparse.ArgumentParser(description='Training GNN on Paper-Field (L2) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default=fopOutputDir,
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default=fopModelDir,
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PF',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=5,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')

args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))

# train_range = {t: True for t in graph.times if t != None and t < 2015}
train_range = {t: True for t in graph.times if t != None and t <2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
test_range = {t: True for t in graph.times if t != None and t > 2016}
print('train valid test {} \n {} \n {} \n'.format(train_range,valid_range,test_range))
types = graph.get_types()
'''
    cand_list stores all the L2 fields, which is the classification domain.
'''
cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())
print('type of candlist {} {}'.format(type(cand_list),len(cand_list)))
print('field paper pf {}'.format(len(graph.edge_list['field']['paper']['PF_in_L2'].keys())))
# print(cand_list)
input('end cand list')
#
# cand_list_2 = graph.edge_list['field']['paper']['PF_in_L2']
# print('type of candlist 2 {} {}'.format(type(cand_list_2),len(cand_list_2)))
# print(cand_list_2)
# input('end cand list 2')


'''
Use KL Divergence here, since each paper can be associated with multiple fields.
Thus this task is a multi-label classification.
'''
criterion = nn.KLDivLoss(reduction='batchmean')


def node_classification_sample(seed, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''
    np.random.seed(seed)
    # print('go to here for classification {}'.format(len(pairs)))
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace=True)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, \
                                                      inp={'paper': np.array(target_info)}, \
                                                      sampled_depth=args.sample_depth, sampled_number=args.sample_width)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field']['rev_PF_in_L2']:
        if i[0] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['paper']['field']['rev_PF_in_L2'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper']['PF_in_L2']:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['field']['paper']['PF_in_L2'] = masked_edge_list

    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    # print('feature \nedge_list{}'.format(edge_list))
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = np.zeros([args.batch_size, len(cand_list)])
    # print('len batch {} and len cand_list {} len yLabel {}'.format(args.batch_size,len(cand_list),len(ylabel)))

    # input('check style of label')
    # print('pair {}'.format(pairs))
    # print('cand_list {}'.format(cand_list))
    for x_id, target_id in enumerate(target_ids):
        for source_id in pairs[target_id][0]:
            # print('source_id {} target {}'.format(source_id,target_id))
            ylabel[x_id][cand_list.index(source_id)] = 1
            # print('go here')
    # print('ylaabel sum {}'.format(ylabel.sum(axis=1).reshape(-1, 1)))
    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
    # print('x_ids {}\nnode_dict[paper] {}'.format(x_ids,node_dict))
    # print('end classification')
    # print('objCand 2 {}'.format(len(objCand)))
    # print('yLabel2 2 {}'.format(len(ylabel)))
    #
    # print('x_ids {}\nyLabel {}\nedge_type {}'.format(1,ylabel.shape,edge_type.shape))
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), \
                                                               sel_train_pairs, train_range))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), \
                                                           sel_valid_pairs, valid_range))
    # print('set valid pairs')
    # print(sel_valid_pairs)
    # input('end of prepare data')
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}
'''
    Prepare all the souce nodes (L2 field) associated with each target node (paper) as dict
'''
objCand = graph.edge_list['paper']['field']['rev_PF_in_L2']
# print('Type of graph.edge_list {}'.format(type(objCand)))
print('objCand {}'.format(len(objCand)))
input('rev_PF_in_L2')
for target_id in graph.edge_list['paper']['field']['rev_PF_in_L2']:
    for source_id in graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id]:
        _time = graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id][source_id]
        # print('_time {} sourceid {} targetid {}'.format(_time,source_id,target_id))
        # input('sample triple')
        if _time in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [[], _time]
            train_pairs[target_id][0] += [source_id]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [[], _time]
            valid_pairs[target_id][0] += [source_id]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [[], _time]
            test_pairs[target_id][0] += [source_id]
# print('test_pairs {}'.format(test_pairs))
np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
# print('valid pair keys {}'.format(list(valid_pairs.keys())))
sel_train_pairs = {p: train_pairs[p] for p in
                   np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                    replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in
                   np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                    replace=False)}
# print('set train and valid pairs')
# print('len train {}'.format(len(sel_train_pairs)))
# print(' train {}'.format(sel_train_pairs))
# print('len valid {}'.format(len(sel_valid_pairs)))
# print(' valid {}'.format(sel_valid_pairs))

# input('sel train {}'.format(args.data_percentage))

'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
# print('type GNN {}'.format(type(graph.node_feature['paper']['emb'].values[0])))
# print(' GNN {}'.format(len(graph.node_feature['paper']['emb'].values[0])))
# input('before GNN')
gnn = GNN(conv_name=args.conv_name, in_dim=len(graph.node_feature['paper']['emb'].values[0]) + 401, \
          n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, \
          num_types=len(graph.get_types()), num_relations=len(graph.get_meta_graph()) + 1).to(device)
classifier = Classifier(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val = 0
train_step = 50

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

print('n_pool {} lenJobs {}'.format(args.n_pool, len(jobs)))
print('job {} type job[0] {}'.format(type(jobs), type(jobs[0])))

for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:0]]
    valid_data = jobs[-1].get()

    # print('type of train data {}'.format(type(train_data)))
    # print('len train {}'.format(len(train_data)))
    # for tupItem in train_data:
    #     print('type tupItem {}'.format(type(tupItem)))
    #     if isinstance(tupItem, torch.Tensor):
    #         print('shape {}'.format(tupItem.shape))
    #     else:
    #         print('len of ndarray {}'.format(len(tupItem)))

    # print('type of valid data {}'.format(type(valid_data)))
    # for tupItem in valid_data:
    #     print('type tupItem {}'.format(type(tupItem)))
    #     if isinstance(tupItem,torch.Tensor):
    #         print('shape {}'.format(tupItem.shape))
    #     else:
    #         print('len of ndarray {}'.format(len(tupItem)))
    # input('data of valid')
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))

    '''
        Train (time < 2015)
    '''
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            print('x_ids {}\nedge_type {}\nylabel {}'.format(x_ids,edge_type,ylabel))
            res = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            # print('res type {}\n {}'.format(type(res),res))
            # print('loss type {}\n {}'.format(type(loss), loss))
            input('train batch')

            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res = classifier.forward(node_rep[x_ids])
        loss = criterion(res, torch.FloatTensor(ylabel).to(device))
        # print('res type {}\n {}'.format(type(res), res))
        # print('label type {}\n {}'.format(type(ylabel), ylabel))
        # input('valid batch')

        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            valid_res += [ai[bi.cpu().numpy()]]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])

        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data

'''
    Evaluate the trained model via test set (time > 2016)
'''

with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
            node_classification_sample(randint(), test_pairs, test_range)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Last Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Last Test MRR:  %.4f' % np.average(test_mrr))

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
            node_classification_sample(randint(), test_pairs, test_range)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]

        res = classifier.forward(paper_rep)
        # print('paper {}\nres {}'.format(type(paper_rep),type(res)))
        # print('paper {}\nres {}'.format(paper_rep.shape, res.shape))
        for ind1 in range(0,len(res)):
            lstGoodInds=[]
            for ind2 in range(0,len(res[ind1])):
                if res[ind1][ind2]>0:
                    strItem='({}:{})'.format(ind2,res[ind1][ind2])
                    lstGoodInds.append(strItem)
            if len(lstGoodInds)>0:
                print('res ind {} {}'.format(ind1,' , '.join(lstGoodInds)))
        for ind1 in range(0,len(ylabel)):
            lstGoodInds=[]
            for ind2 in range(0,len(ylabel[ind1])):
                if ylabel[ind1][ind2]>0:
                    strItem='({}:{})'.format(ind2,ylabel[ind1][ind2])
                    lstGoodInds.append(strItem)
            if len(lstGoodInds)>0:
                print('label ind {} {}'.format(ind1,' , '.join(lstGoodInds)))
        # input('test  ')
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Best Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Best Test MRR:  %.4f' % np.average(test_mrr))
