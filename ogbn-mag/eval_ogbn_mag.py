import argparse
from tqdm import tqdm
import sys

from sklearn.metrics import f1_score
from data import *
from utils import *
from model import *
from warnings import filterwarnings
filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


parser = argparse.ArgumentParser(description='Training GNN on ogbn-mag benchmark')



parser.add_argument('--input_dir', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--model_dir', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--task_type', type=str, default='ensemble,
                    help='Whether to use ensemble evaluation or sequential evaluation')

args = parser.parse_args()
args_print(args)

graph = dill.load(open(args.input_dir, 'rb'))


def ogbn_mag_sample(seed, samp_nodes):
    np.random.seed(seed)
    ylabel      = torch.LongTensor(graph.y[samp_nodes])
    feature, times, edge_list, indxs, _ = sample_subgraph(graph, \
                inp = {'paper': np.concatenate([samp_nodes, graph.years[samp_nodes]]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_MAG)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    train_mask = graph.train_mask[indxs['paper']]
    valid_mask = graph.valid_mask[indxs['paper']]
    test_mask  = graph.test_mask[indxs['paper']]
    ylabel     = graph.y[indxs['paper']]
    return node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel
    
def prepare_data(pool, task_type = 'train', s_idx = 0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    if task_type == 'train':
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(node_classification_sample, args=([randint(), \
                            np.random.choice(graph.train_paper, args.batch_size, replace = False)]))
            jobs.append(p)
    elif task_type == 'ensemble':
        target_papers = graph.test_paper[s_idx * args.batch_size : (s_idx + 1) * args.batch_size]
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(node_classification_sample, args=([randint(), target_papers]))
            jobs.append(p)
    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(node_classification_sample, args=([randint(), target_papers]))
            jobs.append(p)
    return jobs


device = torch.device("cuda:%d" % args.cuda)
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['paper'][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE)
classifier = Classifier(args.n_hid, graph.y.max()+1)

model = nn.Sequential(gnn, classifier)
model.load_state_dict(torch.load(args.model_dir))
model.to(device)
criterion = nn.NLLLoss()

model.eval()
with torch.no_grad():
    if args.task_type == 'ensemble':
        y_pred = []
        y_true = []
        pool = mp.Pool(args.n_pool)
        jobs = prepare_data(pool, task_type = 'ensemble', s_idx = 0, n_batch = 8)
        with tqdm(np.arange(len(graph.test_paper) // args.batch_size), desc='eval') as monitor:
            for s_idx in monitor:
                ress = []
                test_data = [job.get() for job in jobs]
                pool.close()
                pool.join()
                pool = mp.Pool(args.n_pool)
                jobs = prepare_data(pool, task_type = 'ensemble', s_idx = s_idx, n_batch = 8)

                for node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel in test_data:
                    node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
                    res  = classifier.forward(node_rep[:args.batch_size])
                    ress += [res]

                y_pred += torch.stack(ress).mean(dim=0).argmax(dim=1).tolist()
                y_true += list(ylabel[:args.batch_size])

                test_acc = evaluator.eval({
                        'y_true': torch.LongTensor(y_true).unsqueeze(-1),
                        'y_pred': torch.LongTensor(y_pred).unsqueeze(-1)
                    })['acc']
                monitor.set_postfix(accuracy = test_acc)
                
    elif args.task_type == 'sequential':
        y_pred = []
        y_true = []
        pool = mp.Pool(args.n_pool)
        jobs = prepare_data(pool, task_type = 'sequential', s_idx = 0, n_batch = args.n_batch, batch_size=args.batch_size)
        with tqdm(np.arange(len(graph.test_paper) // args.batch_size), desc='eval') as monitor:
            for s_idx in monitor:
                test_data = [job.get() for job in jobs]
                pool.close()
                pool.join()
                pool = mp.Pool(args.n_pool)
                jobs = prepare_data(pool, is_test = True, s_idx = int(s_idx * args.n_batch), batch_size=args.batch_size)

                for node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel in test_data:
                    ylabel = ylabel[:args.batch_size]
                    node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
                    res  = classifier.forward(node_rep[:args.batch_size])
                    pred  = res.argmax(dim=1)
                    
                    y_pred += pred.tolist()
                    y_true += ylabel.tolist()
                    
                test_acc = evaluator.eval({
                                'y_true': torch.FloatTensor(y_true).unsqueeze(-1),
                                'y_pred': torch.FloatTensor(y_pred).unsqueeze(-1)
                            })['acc']
                monitor.set_postfix(accuracy = test_acc)