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
import ast

lstRelations = ['isProgramOf', 'isNLRootOf', 'isProgramOfNLRoot', 'isPLFatherOf', 'isNLFatherOf']
# lstYears = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
dictSourceTargetType = {}
dictSourceTargetType['isProgramOf'] = ('ProgramRoot', 'ProgramElement')
dictSourceTargetType['isNLRootOf'] = ('NLRoot', 'NLElement')
dictSourceTargetType['isProgramOfNLRoot'] = ('ProgramRoot', 'NLRoot')
dictSourceTargetType['isPLFatherOf'] = ('ProgramElement', 'ProgramElement')
dictSourceTargetType['isNLFatherOf'] = ('NLElement', 'NLElement')

fopRootData='../../dataPapers/textInSPOC/mixCode_v2/'
fopStep2=fopRootData+'step2/'
fopStep3=fopRootData+'step3_treesitter/'
fopStep5=fopRootData+'step5/'
fopStep6=fopRootData+'step6/'
createDirIfNotExist(fopStep6)
fopProgramTrain=fopStep2+'train/'
fopProgramTestP=fopStep2+'testP/'
fopProgramTestW=fopStep2+'testW/'
fopStep3Train=fopStep3+'train/'
fopStep3TestP=fopStep3+'testP/'
fopStep3TestW=fopStep3+'testW/'

fpStep5Train=fopStep5+'train'+'_label.txt'
fpStep5TestP=fopStep5+'testP'+'_label.txt'
fpStep5TestW=fopStep5+'testW'+'_label.txt'

fpStep6TextTrain=fopStep6+'train.text.txt'
fpStep6ASTTrain=fopStep6+'train.ast.txt'
fpStep6LabelTrain=fopStep6+'train.label.txt'
fpStep6TextTestP=fopStep6+'testP.text.txt'
fpStep6ASTTestP=fopStep6+'testP.ast.txt'
fpStep6LabelTestP=fopStep6+'testP.label.txt'
fpStep6TextTestW=fopStep6+'testW.text.txt'
fpStep6ASTTestW=fopStep6+'testW.ast.txt'
fpStep6LabelTestW=fopStep6+'testW.label.txt'
fpStep6OutEmbedded=fopStep6+'spoc.emb.txt'
fpStep6OutModel=fopStep6+'spoc.d2v.model'

parser = argparse.ArgumentParser(description='Preprocess OAG (CS/Med/All) Data')

offsetComment=3

def lookUpJSonObject(dictJson,dictFatherId,lstAppearId,indexComment,offsetComment):
    strId=dictJson['id']
    if 'type' in dictJson.keys():
        startLine=dictJson['startLine']
        endLine = dictJson['endLine']

        if startLine>=(indexComment-offsetComment) and endLine<=(indexComment+offsetComment):
            # print('id {}'.format(strId))
            strType=dictJson['type'].strip()
            lstAppearId.append(strId)

    if 'children' in dictJson.keys():
        lstChildren=dictJson['children']
        for child in lstChildren:
            strChildId=lookUpJSonObject(child,dictFatherId,lstAppearId,indexComment,offsetComment)
            dictFatherId[strChildId]=strId
    return strId

def lookUpJSonObjectStep2(dictJson, lstAddToGraph,dictIdsAddToWholeGraph,time,graph,idOfProgram,dictEntities,lstEdges):
    strId=dictJson['id']
    itemNode={}
    if 'type' in dictJson.keys():

        if strId in lstAddToGraph:
            # print('go here')
            strType=dictJson['type'].strip()
            if strType not in dictIdsAddToWholeGraph.keys():
                if 'isRootNode' in dictJson.keys():
                    itemNode={'id': strType, 'type': 'program', 'attr': dictJson['statementType']}
                    strIdToHGT='ProgramRoot_'+str(idOfProgram)
                    itemNode['hgtId']=strIdToHGT
                    if strIdToHGT not in dictEntities['ProgramRoot'].keys():
                        newIntId=len(dictEntities['ProgramRoot'].keys())
                        dictEntities['ProgramRoot']=newIntId
                else:
                    itemNode = {'id': strType, 'type': 'ast', 'attr': 'ast'}
                    strIdToHGT =  strType
                    itemNode['hgtId'] = strIdToHGT
                    if strIdToHGT not in dictEntities['ProgramElement'].keys():
                        newIntId=len(dictEntities['ProgramElement'].keys())
                        dictEntities['ProgramElement']=newIntId
                # print('strType {}'.format(strType))
                dictIdsAddToWholeGraph[strType] = itemNode
            else:
                itemNode=dictIdsAddToWholeGraph[strType]


    if 'children' in dictJson.keys():
        lstChildren=dictJson['children']
        for child in lstChildren:
            childNode=lookUpJSonObjectStep2(child,lstAddToGraph,dictIdsAddToWholeGraph,time,graph,idOfProgram,dictEntities,lstEdges)
            # print('child {}'.format(childNode))
            if str(childNode)!='{}':
                graph.add_edge(itemNode,childNode,time=time, relation_type='ast_edge')
                if itemNode['hgtId'].startswith('ProgramRoot'):
                    tup=('isProgramOf',itemNode['hgtId'],childNode['hgtId'],idOfProgram)
                    lstEdges.append(tup)
                else:
                    tup = ('isPLFatherOfPL', itemNode['hgtId'], childNode['hgtId'], idOfProgram)
                    lstEdges.append(tup)

    if 'nlGraph' in dictJson.keys():
        dictNL=dictJson['nlGraph']
        dictNL['label']='nlGraph'
        dictNL['isNLRootNode']=True
        nlNode=addNLNodeToGraph(dictNL,lstAddToGraph,dictIdsAddToWholeGraph,time,graph,idOfProgram,dictEntities,lstEdges)
        graph.add_edge(itemNode, nlNode, time=time, relation_type='ast_nl_edge')
        tup = ('isPLFatherOfNL', itemNode['hgtId'], nlNode['hgtId'], idOfProgram)
        lstEdges.append(tup)

    return itemNode

def addNLNodeToGraph(dictNL, lstAddToGraph, dictIdsAddToWholeGraph,time, graph,idOfProgram,dictEntities,lstEdges):
    strLabel = dictNL['label'].strip().replace('\t',' ').strip()
    itemNode = {'id': strLabel, 'type': 'nl_nonterminal', 'attr': 'nl'}
    dictIdsAddToWholeGraph[strLabel] = itemNode
    if 'isNLRootNode' in dictNL.keys():
        strIdToHGT = 'NLRoot_' + str(idOfProgram)
        itemNode['hgtId'] = strIdToHGT
        if strIdToHGT not in dictEntities['NLRoot'].keys():
            newIntId = len(dictEntities['NLRoot'].keys())
            dictEntities['NLRoot'] = newIntId
    else:
        strIdToHGT = strLabel
        itemNode['hgtId'] = strIdToHGT
        if strIdToHGT not in dictEntities['NLElement'].keys():
            newIntId = len(dictEntities['NLElement'].keys())
            dictEntities['NLElement'] = newIntId

    lstChildren = dictNL['children']
    for i in range(0, len(lstChildren)):
        childNode = addNLNodeToGraph(lstChildren[i],  lstAddToGraph, dictIdsAddToWholeGraph,time, graph)
        graph.add_edge(itemNode, childNode, relation_type='nl_pos_edge',time=time)
        if itemNode['hgtId'].startswith('NLRoot'):
            tup = ('isNLRootOf', itemNode['hgtId'], childNode['hgtId'], idOfProgram)
            lstEdges.append(tup)
        else:
            tup = ('isNLFatherOfNL', itemNode['hgtId'], childNode['hgtId'], idOfProgram)
            lstEdges.append(tup)

    # if 'dependencies' in dictNL.keys():
    #     lstDeps = dictNL['dependencies']
    #     for i in range(0, len(lstDeps)):
    #         tup = lstDeps[i]
    #         nodeSource =  {'id': tup[3], 'type': 'nl_terminal', 'attr': 'nl'}
    #         dictIdsAddToWholeGraph[nodeSource['id']]=nodeSource
    #         nodeTarget = {'id': tup[4], 'type': 'nl_terminal', 'attr': 'nl'}
    #         dictIdsAddToWholeGraph[nodeTarget['id']] = nodeTarget
    #         graph.add_edge(nodeSource, nodeTarget, relation_type='nl_dep_edge_{}'.format(tup[2]),time=time)
    return itemNode

def traverseHGTGraph(lstFpStep6Text,lstFpStep6Label,lstFopProgram,lstFopStep3,time,graph,fopOutput):
# load graph by id

    indexProgram = 0
    dictEntities={}
    dictEntities['ProgramRoot']={}
    dictEntities['ProgramElement']={}
    dictEntities['NLRoot'] = {}
    dictEntities['NLElement'] = {}

    # traverse train data
    indexBatch=0
    f1=open(lstFpStep6Text[indexBatch],'r')
    arrText=f1.read().strip().split('\n')
    f1.close()

    f1 = open(lstFpStep6Label[indexBatch], 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()

    # dictIdsAddToWholeGraph={}
    for i in range(0,len(arrText)):
        arrItem=arrText[i].split('\t')

        strLabel=arrLabel[i].split('\t')[1]
        if len(arrItem)>=2:
            key=arrItem[0]
            fpItemCode=lstFopProgram[indexBatch]+key+'.cpp'
            f1=open(fpItemCode,'r')
            arrItemCode=f1.read().strip().split('\n')
            f1.close()

            indexComment=-1
            for j in range(0,len(arrItemCode)):
                itemStrip=arrItemCode[j].strip()
                if itemStrip.startswith('//'):
                    indexComment=j
                    break

            fpItemAST=fopStep3+key+'_ast.txt'
            f1=open(fpItemAST,'r')
            strJson=f1.read().strip().split('\n')[1]
            f1.close()

            dictJson=ast.literal_eval(strJson)
            dictJson['type']=key
            dictJson['isRootNode']=1
            dictJson['statementType']=strLabel
            time=time+1

            lstItemAST = []
            lstAppearId = []
            dictFatherId = {}
            # print('indexCmt {}'.format(indexComment))
            idOfProgram=
            lookUpJSonObject(dictJson, dictFatherId, lstAppearId, indexComment, offsetComment,)
            # print('lstAppearId {}'.format(lstAppearId))
            lstAddToGraph = []
            for id in lstAppearId:
                strIdItem = id
                lstAncestor = []
                while strIdItem in dictFatherId.keys():
                    strIdItem = dictFatherId[strIdItem]
                    lstAncestor.insert(0, strIdItem)
                for idPar in lstAncestor:
                    if idPar not in set(lstAddToGraph):
                        lstAddToGraph.append(idPar)
                lstAddToGraph.append(id)
            lstItemAST = []
            # print('lstAddToGraph {}'.format(lstAddToGraph))
            lookUpJSonObjectStep2(dictJson, lstAddToGraph,dictIdsAddToWholeGraph, time,graph)
            print('end {}/{}'.format(i,len(arrText)))
            # if i == 200:
            #     if 'train' in fopStep3:
            #         break
            # elif i==100:
            #     if not 'train' in fopStep3:
            #         break
    return time






import argparse

parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--output_dir', type=str, default='dataset_random/OGB_MAG_spoc_v3.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()

fopRandomDataset='dataset_spoc/'
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