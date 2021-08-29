from pyHGT.data_spoc import *
from pyHGT.utils_spoc import *
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator
from UtilFunctions import createDirIfNotExist
from tree_sitter import Language, Parser
import random
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../../')))

lstRelations = ['isProgramOf', 'isNLRootOf', 'isProgramOfNLRoot', 'isPLFatherOfPL', 'isPLFatherOfNLRoot','isNLFatherOfNL']
# lstYears = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
dictSourceTargetType = {}
dictSourceTargetType['isProgramOf'] = ('ProgramRoot', 'ProgramElement')
dictSourceTargetType['isNLRootOf'] = ('NLRoot', 'NLElement')
dictSourceTargetType['isProgramOfNLRoot'] = ('ProgramRoot', 'NLRoot')
dictSourceTargetType['isPLFatherOfPL'] = ('ProgramElement', 'ProgramElement')
dictSourceTargetType['isPLFatherOfNLRoot'] = ('ProgramElement', 'NLRoot')
dictSourceTargetType['isNLFatherOfNL'] = ('NLElement', 'NLElement')


def generateSPOCDataset(fopDataset):
    # generate ds for ProgramRoot
    createDirIfNotExist(fopDataset)
    fpProgramRoot=fopDataset+'ProgramRoot_raw.txt'
    fpProgramElement = fopDataset + 'ProgramElement_raw.txt'
    fpNLRoot = fopDataset + 'NLRoot_raw.txt'
    fpNLElement = fopDataset + 'NLElement_raw.txt'
    fpEdgeList = fopDataset + 'edgeLists_id.txt'
    fpLabelList = fopDataset + 'labels_id.txt'


    numId=-1
    lstPR=[]
    f1=open(fpProgramRoot,'r')
    arrPRs=f1.read().strip().split('\n')
    f1.close()

    for i in range(0,len(arrPRs)):
        lstPR.append(i)

    lstPE=[]
    f1 = open(fpProgramElement, 'r')
    arrPEs = f1.read().strip().split('\n')
    f1.close()

    for i in range(0, len(arrPEs)):
        lstPE.append(i)


    lstNLR=[]
    f1 = open(fpNLRoot, 'r')
    arrNLRs = f1.read().strip().split('\n')
    f1.close()
    for i in range(0, len(arrNLRs)):
        lstNLR.append(i)

    lstNLE = []
    f1 = open(fpNLElement, 'r')
    arrNLEs = f1.read().strip().split('\n')
    f1.close()
    for i in range(0, len(arrNLEs)):
        lstNLE.append(i)


    lstLabels = []
    f1=open(fpLabelList,'r')
    arrLabels=f1.read().strip().split('\n')
    f1.close()

    for item in arrLabels:
        lstLabels.append(int(item))

    return lstPR,lstPE,lstNLR,lstNLE,lstLabels


import argparse

parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--output_dir', type=str, default='dataset_spoc/spoc.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()

fopRandomDataset='dataset_spoc/'
lstPR,lstPE,lstNLR,lstNLE,lstLabels=generateSPOCDataset(fopRandomDataset)
print('finish random ds generation')
fpProgramRoot = fopRandomDataset + 'ProgramRoot_raw.txt'
fpProgramElement = fopRandomDataset + 'ProgramElement_raw.txt'
fpNLRoot = fopRandomDataset + 'NLRoot_raw.txt'
fpNLElement = fopRandomDataset + 'NLElement_raw.txt'
fpEdgeList = fopRandomDataset + 'edgeLists_id.txt'
fpLabelList = fopRandomDataset + 'labels_id.txt'


dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
evaluator = Evaluator(name='ogbn-mag')
edge_index_dict = data.edge_index_dict
graph = Graph()
edg   = graph.edge_list
# print('datta object {}'.format(data))
f1=open(fpNLRoot,'r')
arrNLRs=f1.read().split('\n')
f1.close()
lstYear1=[]
for i in range(0,len(arrNLRs)):
    lstYear1.append(i)
years=np.asarray(lstYear1)
# years = data.node_year['paper'].t().numpy()[0]
# print('year 1 {} {}'.format(len(years),years))

# input('next ')
graph = Graph()
edg   = graph.edge_list

# years = data.node_year['paper'].t().numpy()[0]
# print('edg {} {}'.format(len(edg),edg))
# print('year 2 {} {}'.format(len(years),years))
# input('year')


# # step1
# for key in edge_index_dict:
#     print('key {}'.format(key))
#     edges = edge_index_dict[key]
#     s_type, r_type, t_type = key[0], key[1], key[2]
#     elist = edg[t_type][s_type][r_type]
#     rlist = edg[s_type][t_type]['rev_' + r_type]
#     # print('type {}'.format(type(elist)))
#     # print('data {}'.format(elist))
#     idx=0
#     for s_id, t_id in edges.t().tolist():
#         year = None
#         if s_type == 'paper':
#             year = years[s_id]
#         elif t_type == 'paper':
#             year = years[t_id]
#         elist[t_id][s_id] = year
#         rlist[s_id][t_id] = year
#         idx=idx+1
#         # print('s_id {} t_id {} year {}'.format(s_id,t_id,year))
#         # input('s_t_id ')
#         # if idx==numLimit:
#         #     break

# step1 customize
# elist = defaultdict(Graph)
# rlist = defaultdict(Graph)
# elist = edg['paper']['author']['writes']
# rlist = edg['author']['paper']['rev_' + 'writes']
# elist = edg['paper']['author']['writesaba']
# rlist = edg['author']['paper']['rev_' + 'writes']
f1=open(fpEdgeList,'r')
arrEdges=f1.read().split('\n')
f1.close()
for line in arrEdges:
    arrTabsItem=line.split('\t')
    if len(arrTabsItem)>=4:
        s_type=dictSourceTargetType[arrTabsItem[0]][0]
        t_type=dictSourceTargetType[arrTabsItem[0]][1]
        r_type=arrTabsItem[0]
        elist = edg[t_type][s_type][r_type]
        rlist = edg[s_type][t_type]['rev_' + r_type]
        s_id=int(arrTabsItem[1])
        t_id=int(arrTabsItem[2])
        year=int(arrTabsItem[3])
        elist[t_id][s_id]=year
        rlist[s_id][t_id]=year

edg = {}
# deg = {key : np.zeros(data.num_nodes[key]) for key in data.num_nodes}
deg={}
deg['ProgramRoot']= np.zeros(len(lstPR))
deg['ProgramElement']= np.zeros(len(lstPE))
deg['NLRoot']= np.zeros(len(lstNLR))
deg['NLElement']= np.zeros(len(lstNLE))
print('deg {}\n{}\n{}'.format(deg.keys(),type(deg),len(deg['ProgramElement'])))
# input('abc ')
# # step2
# for k1 in graph.edge_list:
#     if k1 not in edg:
#         edg[k1] = {}
#     for k2 in graph.edge_list[k1]:
#         if k2 not in edg[k1]:
#             edg[k1][k2] = {}
#         for k3 in graph.edge_list[k1][k2]:
#             if k3 not in edg[k1][k2]:
#                 edg[k1][k2][k3] = {}
#             for e1 in graph.edge_list[k1][k2][k3]:
#                 if len(graph.edge_list[k1][k2][k3][e1]) == 0:
#                     continue
#
#                 edg[k1][k2][k3][e1] = {}
#                 for e2 in graph.edge_list[k1][k2][k3][e1]:
#                     edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
#                 deg[k1][e1] += len(edg[k1][k2][k3][e1])
#             print('item inside k1 k2 k3 ',k1, k2, k3, len(edg[k1][k2][k3]),type(edg[k1][k2][k3]))
# graph.edge_list = edg

# custom step2
for k1 in graph.edge_list:
    if k1 not in edg:
        edg[k1] = {}
    for k2 in graph.edge_list[k1]:
        if k2 not in edg[k1]:
            edg[k1][k2] = {}
        for k3 in graph.edge_list[k1][k2]:
            if k3 not in edg[k1][k2]:
                edg[k1][k2][k3] = {}
            # print('begin inside k1 k2 k3 ', k1, k2, k3, len(edg[k1][k2][k3]), type(edg[k1][k2][k3]))
            for e1 in graph.edge_list[k1][k2][k3]:
                if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                    continue

                edg[k1][k2][k3][e1] = {}
                for e2 in graph.edge_list[k1][k2][k3][e1]:
                    edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
                deg[k1][e1] += len(edg[k1][k2][k3][e1])
            print('item inside k1 k2 k3 ', k1, k2, k3, len(edg[k1][k2][k3]), type(edg[k1][k2][k3]))
graph.edge_list = edg

# cv = data.x_dict['paper'].numpy()
# print('cv dddd {}'.format(cv.shape))
# input('cv ')
cv=torch.zeros(len(lstPR),128)
graph.node_feature['ProgramRoot'] = np.concatenate((cv, np.log10(deg['ProgramRoot'].reshape(-1, 1))), axis=-1)
cv=torch.zeros(len(lstPE),128)
graph.node_feature['ProgramElement'] = np.concatenate((cv, np.log10(deg['ProgramElement'].reshape(-1, 1))), axis=-1)
cv=torch.zeros(len(lstNLR),128)
graph.node_feature['NLRoot'] = np.concatenate((cv, np.log10(deg['NLRoot'].reshape(-1, 1))), axis=-1)
cv=torch.zeros(len(lstNLE),128)
graph.node_feature['NLElement'] = np.concatenate((cv, np.log10(deg['NLElement'].reshape(-1, 1))), axis=-1)
# for _type in data.num_nodes:
#     if _type not in ['paper', 'institution']:
#         i = []
#         for _rel in graph.edge_list[_type]['paper']:
#             for t in graph.edge_list[_type]['paper'][_rel]:
#                 for s in graph.edge_list[_type]['paper'][_rel][t]:
#                     i += [[t, s]]
#         if len(i) == 0:
#             continue
#         i = np.array(i).T
#         v = np.ones(i.shape[1])
#         m = normalize(sp.coo_matrix((v, i), \
#             shape=(data.num_nodes[_type], data.num_nodes['paper'])))
#         out = m.dot(cv)
#         graph.node_feature[_type] = np.concatenate((out, np.log10(deg[_type].reshape(-1, 1))), axis=-1)
#
# for key in graph.node_feature.keys():
#     print('node features type {} {}'.format(key,graph.node_feature[key].shape))
#
# input('node ft')
# cv = graph.node_feature['author'][:, :-1]
# # print('author feature {}'.format(cv.shape))
# i = []
# for _rel in graph.edge_list['institution']['author']:
#     for j in graph.edge_list['institution']['author'][_rel]:
#         for t in graph.edge_list['institution']['author'][_rel][j]:
#             i += [[j, t]]
# i = np.array(i).T
# v = np.ones(i.shape[1])
# m = normalize(sp.coo_matrix((v, i), \
#     shape=(data.num_nodes['institution'], data.num_nodes['author'])))
# out = m.dot(cv)
# graph.node_feature['institution'] = np.concatenate((out, np.log10(deg['institution'].reshape(-1, 1))), axis=-1)
#
# # print('node_feature dddd {}'.format(len(graph.node_feature['field_of_study'])))
# # input('cv ')
#
#
# # print('node features {} \n{}'.format(graph.node_feature.keys(),type(graph.node_feature['paper'])))
# # input('input ')


# y = data.y_dict['paper'].t().numpy()[0]
y=np.asarray(lstLabels)
# print('type {}'.format(type(y)))
split_idx = dataset.get_idx_split()

# train_paper = split_idx['train']['paper'].numpy()
# valid_paper = split_idx['valid']['paper'].numpy()
# test_paper  = split_idx['test']['paper'].numpy()
# print('type train_paper {} {}'.format(valid_paper[10],type(train_paper)))

train_paper = np.asarray(lstNLR[:115749])
valid_paper = np.asarray(lstNLR[115749:136624])
test_paper  = np.asarray(lstNLR[136624:])


graph.y = y
graph.train_paper = train_paper
graph.valid_paper = valid_paper
graph.test_paper  = test_paper
graph.years       = years
# print('test paper {}'.format(test_paper))
# print('yshape {}\nsampple y{}'.format(y.shape,y[0]))
# input('test ')

graph.train_mask = np.zeros(len(graph.node_feature['NLRoot']), dtype=bool)
graph.train_mask[graph.train_paper] = True

graph.valid_mask = np.zeros(len(graph.node_feature['NLRoot']), dtype=bool)
graph.valid_mask[graph.valid_paper] = True

graph.test_mask = np.zeros(len(graph.node_feature['NLRoot']),  dtype=bool)
graph.test_mask[graph.test_paper] = True

# print('test mask {}'.format(graph.test_mask))
# input(' abc ')

dill.dump(graph, open(args.output_dir, 'wb'))
print('finish')