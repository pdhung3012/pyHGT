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

lstRelations = ['isProgramOf', 'isNLRootOf', 'isProgramOfNLRoot', 'isPLFatherOf', 'isNLFatherOf']
lstYears = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
dictSourceTargetType = {}
dictSourceTargetType['isProgramOf'] = ('ProgramRoot', 'ProgramElement')
dictSourceTargetType['isNLRootOf'] = ('NLRoot', 'NLElement')
dictSourceTargetType['isProgramOfNLRoot'] = ('ProgramRoot', 'NLRoot')
dictSourceTargetType['isPLFatherOf'] = ('ProgramElement', 'ProgramElement')
dictSourceTargetType['isNLFatherOf'] = ('NLElement', 'NLElement')

def generateRandomDataset(fopDataset):
    # generate ds for ProgramRoot
    createDirIfNotExist(fopDataset)
    fpProgramRoot=fopDataset+'ProgramRoot.txt'
    fpProgramElement = fopDataset + 'ProgramElement.txt'
    fpNLRoot = fopDataset + 'NLRoot.txt'
    fpNLElement = fopDataset + 'NLElement.txt'
    fpEdgeList = fopDataset + 'edgeLists.txt'
    fpLabelList = fopDataset + 'labelLists.txt'


    numId=-1
    lenProgramRoot=100000
    lstPR=[]
    lstPRStr=[]
    for i in range(0,lenProgramRoot):
        numId=numId+1
        lstPR.append(numId)
        indexYear = random.randint(0, len(lstYears) - 1)
        year=lstYears[indexYear]
        lstPRStr.append('{}\t{}'.format(numId,year))

    lenProgramElement = 10000
    lstPE=[]
    lstPEStr=[]
    numId=-1
    for i in range(0,lenProgramElement):
        numId=numId+1
        lstPE.append(numId)
        indexYear = random.randint(0, len(lstYears) - 1)
        year=lstYears[indexYear]
        lstPEStr.append('{}\t{}'.format(numId,year))

    lenNLRoot = 100000
    lstNLR=[]
    lstNLRStr=[]
    numId=-1
    for i in range(0,lenNLRoot):
        numId=numId+1
        lstNLR.append(numId)
        indexYear = random.randint(0, len(lstYears) - 1)
        year = lstYears[indexYear]
        lstNLRStr.append('{}\t{}'.format(numId, year))

    lenNLElement = 20000
    lstNLE = []
    lstNLEStr=[]
    numId=-1
    for i in range(0, lenNLElement):
        numId = numId + 1
        lstNLE.append(numId)
        indexYear = random.randint(0, len(lstYears) - 1)
        year = lstYears[indexYear]
        lstNLEStr.append('{}\t{}'.format(numId, year))
    f1=open(fpProgramRoot,'w')
    f1.write('\n'.join(lstPRStr))
    f1.close()
    f1 = open(fpProgramElement, 'w')
    f1.write('\n'.join(lstPEStr))
    f1.close()
    f1=open(fpNLRoot,'w')
    f1.write('\n'.join(lstNLRStr))
    f1.close()
    f1 = open(fpNLElement, 'w')
    f1.write('\n'.join(lstNLEStr))
    f1.close()

    lenEdge=10
    lstEdgesAdd=[]
    for i in range(0,lenProgramRoot):
        sourceNode=lstPR[i]
        randEdgeNum=random.randint(1,lenEdge)
        dictEdges={}
        for j in range(0,randEdgeNum):
            indexTarget=random.randint(0,lenProgramElement-1)
            indexYear=random.randint(0,len(lstYears)-1)
            randTargetNode=lstPE[indexTarget]
            # print('ind year {}'.format(indexYear))
            randYear=lstYears[indexYear]
            strKey='{} {}'.format(randTargetNode,randYear)
            if strKey not in dictEdges.keys():
                dictEdges[strKey]=1
                strLine='{}\t{}\t{}\t{}'.format(lstRelations[0],sourceNode,randTargetNode,randYear)
                lstEdgesAdd.append(strLine)
    f1=open(fpEdgeList,'w')
    f1.write('\n'.join(lstEdgesAdd)+'\n')
    f1.close()

    lenEdge = 100
    lstEdgesAdd = []
    for i in range(0, lenProgramElement):
        sourceNode = lstPE[i]
        randEdgeNum = random.randint(1, lenEdge)
        dictEdges = {}
        for j in range(0, randEdgeNum):
            indexTarget = random.randint(0, lenProgramElement-1)
            indexYear = random.randint(0, len(lstYears)-1)
            randTargetNode = lstPE[indexTarget]
            randYear = lstYears[indexYear]
            strKey = '{} {}'.format(randTargetNode, randYear)
            if strKey not in dictEdges.keys():
                dictEdges[strKey] = 1
                strLine = '{}\t{}\t{}\t{}'.format(lstRelations[3], sourceNode, randTargetNode, randYear)
                lstEdgesAdd.append(strLine)
    f1 = open(fpEdgeList, 'a')
    f1.write('\n'.join(lstEdgesAdd) + '\n')
    f1.close()


    lenEdge = 10
    lstEdgesAdd = []
    for i in range(0, lenNLRoot):
        sourceNode = lstNLR[i]
        randEdgeNum = random.randint(1, lenEdge)
        dictEdges = {}
        for j in range(0, randEdgeNum):
            indexTarget = random.randint(0, lenNLElement-1)
            indexYear = random.randint(0, len(lstYears)-1)
            randTargetNode = lstNLE[indexTarget]
            randYear = lstYears[indexYear]
            strKey = '{} {}'.format(randTargetNode, randYear)
            if strKey not in dictEdges.keys():
                dictEdges[strKey] = 1
                strLine = '{}\t{}\t{}\t{}'.format(lstRelations[1], sourceNode, randTargetNode, randYear)
                lstEdgesAdd.append(strLine)
    f1 = open(fpEdgeList, 'a')
    f1.write('\n'.join(lstEdgesAdd) + '\n')
    f1.close()

    lenEdge = 100
    lstEdgesAdd = []
    for i in range(0, lenNLElement):
        sourceNode = lstNLE[i]
        randEdgeNum = random.randint(1, lenEdge)
        dictEdges = {}
        for j in range(0, randEdgeNum):
            indexTarget = random.randint(0, lenNLElement-1)
            indexYear = random.randint(0, len(lstYears)-1)
            randTargetNode = lstNLE[indexTarget]
            randYear = lstYears[indexYear]
            strKey = '{} {}'.format(randTargetNode, randYear)
            if strKey not in dictEdges.keys():
                dictEdges[strKey] = 1
                strLine = '{}\t{}\t{}\t{}'.format(lstRelations[4], sourceNode, randTargetNode, randYear)
                lstEdgesAdd.append(strLine)
    f1 = open(fpEdgeList, 'a')
    f1.write('\n'.join(lstEdgesAdd) + '\n')
    f1.close()

    lenEdge = 2
    lstEdgesAdd = []
    for i in range(0, lenProgramRoot):
        sourceNode = lstPR[i]
        randEdgeNum = random.randint(1, lenEdge)
        dictEdges = {}
        for j in range(0, randEdgeNum):
            indexTarget = random.randint(0, lenNLRoot-1)
            indexYear = random.randint(0, len(lstYears)-1)
            randTargetNode = lstNLR[indexTarget]
            randYear = lstYears[indexYear]
            strKey = '{} {}'.format(randTargetNode, randYear)
            if strKey not in dictEdges.keys():
                dictEdges[strKey] = 1
                strLine = '{}\t{}\t{}\t{}'.format(lstRelations[2], sourceNode, randTargetNode, randYear)
                lstEdgesAdd.append(strLine)
    f1 = open(fpEdgeList, 'a')
    f1.write('\n'.join(lstEdgesAdd) + '\n')
    f1.close()

    lstLabels = []
    for i in range(0, lenProgramRoot):
        sourceNode = lstPR[i]
        indexRanPE = random.randint(0, 1)
        ranPE=lstPE[indexRanPE]
        lstLabels.append(ranPE)

    f1=open(fpLabelList,'w')
    f1.write('\n'.join(map(str,lstLabels)))
    f1.close()
    return lstPR,lstPE,lstNLR,lstNLE,lstLabels






import argparse

parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--output_dir', type=str, default='dataset_random/OGB_MAG_spoc.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()

fopRandomDataset='dataset_random/'
lstPR,lstPE,lstNLR,lstNLE,lstLabels=generateRandomDataset(fopRandomDataset)
print('finish random ds generation')
fpProgramRoot = fopRandomDataset + 'ProgramRoot.txt'
fpProgramElement = fopRandomDataset + 'ProgramElement.txt'
fpNLRoot = fopRandomDataset + 'NLRoot.txt'
fpNLElement = fopRandomDataset + 'NLElement.txt'
fpEdgeList = fopRandomDataset + 'edgeLists.txt'
fpLabelList = fopRandomDataset + 'labelLists.txt'


dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
evaluator = Evaluator(name='ogbn-mag')
edge_index_dict = data.edge_index_dict
graph = Graph()
edg   = graph.edge_list
# print('datta object {}'.format(data))
f1=open(fpProgramRoot,'r')
arrPRs=f1.read().split('\n')
f1.close()
lstYear1=[]
for item in arrPRs:
    arrTabs=item.split('\t')
    if len(arrTabs)>=2:
        lstYear1.append(int(arrTabs[1]))
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

train_paper = np.asarray(lstPR[:80000])
valid_paper = np.asarray(lstPR[80000:90000])
test_paper  = np.asarray(lstPR[90000:])


graph.y = y
graph.train_paper = train_paper
graph.valid_paper = valid_paper
graph.test_paper  = test_paper
graph.years       = years
# print('test paper {}'.format(test_paper))
# print('yshape {}\nsampple y{}'.format(y.shape,y[0]))
# input('test ')

graph.train_mask = np.zeros(len(graph.node_feature['ProgramRoot']), dtype=bool)
graph.train_mask[graph.train_paper] = True

graph.valid_mask = np.zeros(len(graph.node_feature['ProgramRoot']), dtype=bool)
graph.valid_mask[graph.valid_paper] = True

graph.test_mask = np.zeros(len(graph.node_feature['ProgramRoot']),  dtype=bool)
graph.test_mask[graph.test_paper] = True

# print('test mask {}'.format(graph.test_mask))
# input(' abc ')

dill.dump(graph, open(args.output_dir, 'wb'))
print('finish')