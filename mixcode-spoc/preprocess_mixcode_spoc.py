import glob
import traceback

from pyHGT_SPoC.data import *
from pyHGT_SPoC.utils import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from UtilFunctions_RTX3090 import createDirIfNotExist
import argparse

strSplitCharacterForNodeEdge=' ABAZ '
parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--input_mixgraph_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step5_totalGraph/3/',
                    help='The address of input spoc dataset.')
parser.add_argument('--input_embedding_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/embeddingModels/d2v/',
                    help='The address of pretrained embedding model.')
parser.add_argument('--output_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step6_hgt_problem2/3/mixcode_spoc.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()

fopInputMixGraph=args.input_mixgraph_dir
fopInputEmbeddingModel=args.input_embedding_dir
fpOutputGraph=args.output_dir
fopOutputGraph=os.path.dirname(fpOutputGraph)
createDirIfNotExist(fopOutputGraph)

problemId=1
fopRoot='/home/hungphd/media/dataPapersExternal/mixCodeRaw/'
fpDictLiterals=fopRoot+'step2_dictLiterals_all.txt'
fpNodeProgramRoot=fopInputMixGraph+'nodes_ProgramRoot.txt'
fpNodeASTNode = fopInputMixGraph + 'nodes_ASTNode.txt'
fpNodeNLRoot = fopInputMixGraph + 'nodes_NLRoot.txt'
fpNodeNLNode = fopInputMixGraph + 'nodes_NLNode.txt'

lstFpEdgesList=glob.glob(fopInputMixGraph+'edges_*.txt')

f1=open(fpDictLiterals,'r')
arrLits=f1.read().strip().split('\n')
f1.close()
dictLiteralsToValues={}
dictValuesToLiterals={}
for item in arrLits:
    arrTabs=item.split('\t')
    if len(arrTabs)>=2:
        strContent='\t'.join(arrTabs[1:])
        dictLiteralsToValues[arrTabs[0]]=strContent
        dictValuesToLiterals[strContent]=arrTabs[0]

dictProgramRoots={}
dictASTNodes={}
dictNLRoots={}
dictNLNodes={}

f1=open(fpNodeProgramRoot,'r')
arrPRs=f1.read().strip().split('\n')
f1.close()
dictProgramRoots={}
dictRangeTrainTest={}
prevTrainTest=''
dictLabelsTextToInt={}
lstIdxLabels=[]
for i in range(0,len(arrPRs)):
    item=arrPRs[i]
    arrItemContent=item.split(strSplitCharacterForNodeEdge)
    if len(arrItemContent)>=3:
        id=len(dictProgramRoots.keys())
        strTrainTest=arrItemContent[2].split('\t')[0]
        strLabel=arrItemContent[1].split('\t')[problemId]
        if strLabel not in dictLabelsTextToInt.keys():
            dictLabelsTextToInt[strLabel]=len(dictLabelsTextToInt.keys())+1
        idxLabel=dictLabelsTextToInt[strLabel]
        lstIdxLabels.append(idxLabel)
        tup = [id, arrItemContent[0],idxLabel]
        dictProgramRoots[arrItemContent[0]]=tup

        if strTrainTest not in dictRangeTrainTest.keys():
            dictRangeTrainTest[strTrainTest]=[i,-1]

        if prevTrainTest!='' and prevTrainTest!=strTrainTest:
            dictRangeTrainTest[prevTrainTest][1]=i-1
        if (i+1)==len(arrPRs):
            dictRangeTrainTest[strTrainTest][1]=i-1
        prevTrainTest=strTrainTest

f1=open(fpNodeNLRoot,'r')
arrNLRs=f1.read().strip().split('\n')
f1.close()
dictNLRoots={}
lstYears=[]
for i in range(0,len(arrNLRs)):
    item=arrNLRs[i]
    arrItemContent=item.split(strSplitCharacterForNodeEdge)
    if len(arrItemContent)>=3:
        id=len(dictNLRoots.keys())
        strTrainTest=arrItemContent[2].split('\t')[0]
        strLabel=arrItemContent[1].split('\t')[problemId]
        # if strLabel not in dictLabelsTextToInt.keys():
        #     dictLabelsTextToInt[strLabel]=len(dictLabelsTextToInt.keys())+1
        idxLabel=dictLabelsTextToInt[strLabel]
        lstIdxLabels.append(idxLabel)
        tup = [id, arrItemContent[0],idxLabel]
        dictNLRoots[arrItemContent[0]]=tup
        lstYears.append(2020)
years=np.asarray(lstYears)

f1=open(fpNodeASTNode,'r')
arrASTNodes=f1.read().strip().split('\n')
f1.close()
dictASTNodes={}
for i in range(0,len(arrASTNodes)):
    item=arrASTNodes[i]
    id=len(dictASTNodes.keys())
    dictASTNodes[item]=[id]

f1=open(fpNodeNLNode,'r')
arrNLNodes=f1.read().strip().split('\n')
f1.close()
dictNLNodes={}
for i in range(0,len(arrNLNodes)):
    item=arrNLNodes[i]
    id=len(dictNLNodes.keys())
    dictNLNodes[item]=[id]

dictAllNodes={}
dictAllNodes['ProgramRoot']=dictProgramRoots
dictAllNodes['NLRoot']=dictNLRoots
dictAllNodes['ASTNode']=dictASTNodes
dictAllNodes['NLNode']=dictNLNodes

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
evaluator = Evaluator(name='ogbn-mag')
edge_index_dict = data.edge_index_dict
graph = Graph()
edg   = graph.edge_list
# years = data.node_year_dict['paper'].t().numpy()[0]

graph = Graph()
edg   = graph.edge_list
# years = data.node_year_dict['paper'].t().numpy()[0]
# for key in edge_index_dict:
#     print(key)
#     edges = edge_index_dict[key]
#     s_type, r_type, t_type = key[0], key[1], key[2]
#     elist = edg[t_type][s_type][r_type]
#     rlist = edg[s_type][t_type]['rev_' + r_type]
#     for s_id, t_id in edges.t().tolist():
#         year = None
#         if s_type == 'paper':
#             year = years[s_id]
#         elif t_type == 'paper':
#             year = years[t_id]
#         elist[t_id][s_id] = year
#         rlist[s_id][t_id] = year

for i in range(0,len(lstFpEdgesList)):
    fpEdge=lstFpEdgesList[i]
    f1=open(fpEdge,'r')
    arrEdges=f1.read().strip().split('\n')
    f1.close()
    print('begin edge {}'.format(fpEdge))
    nameFileEdge=os.path.basename(fpEdge)
    strTypeForEdge=nameFileEdge.replace('edges_','').replace('.txt','')
    arrSourceTarget=strTypeForEdge.split(' - ')
    strSourceType=arrSourceTarget[0]
    strTargetType=arrSourceTarget[1]
    for line in arrEdges:
        arrTabsItem=line.split(strSplitCharacterForNodeEdge)
        if len(arrTabsItem)>=3:
            try:
                s_type = strSourceType
                t_type = strTargetType
                r_type = strTypeForEdge
                elist = edg[t_type][s_type][r_type]
                rlist = edg[s_type][t_type]['rev_' + r_type]
                strTextSource = arrTabsItem[0]
                strTextTarget = arrTabsItem[1]
                if strTextSource in dictValuesToLiterals.keys():
                    strTextSource = dictValuesToLiterals[strTextSource]
                if strTextTarget in dictValuesToLiterals.keys():

                    strNewTextTarget = dictValuesToLiterals[strTextTarget]
                    # print('text target {} AAAA {}'.format(strTextTarget,strNewTextTarget))
                    strTextTarget=strNewTextTarget

                if(strTextSource=='' or strTextTarget==''):
                    continue

                if strTextSource != 'translation_unit':
                    s_id = dictAllNodes[s_type][strTextSource][0]
                else:
                    strRootKey = arrTabsItem[2].split('\t')[1]
                    s_id = dictAllNodes[s_type][strRootKey][0]
                # s_id=dictAllNodes[s_type][arrTabsItem[0]][0]
                # print('t_type {} BBB {} AAA {}'.format(t_type,arrTabsItem[1],strTextTarget))
                t_id = dictAllNodes[t_type][strTextTarget][0]
                # year=int(arrTabsItem[3])
                year = 2020
                elist[t_id][s_id] = year
                rlist[s_id][t_id] = year
            except:
                traceback.print_exc()
                quit()

    print('end edge {}'.format(fpEdge))
edg = {}
deg={}
# deg = {key : np.zeros(data.num_nodes[key]) for key in data.num_nodes}
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
#             print(k1, k2, k3, len(edg[k1][k2][k3]))
deg['ProgramRoot']= np.zeros(len(dictProgramRoots.keys()))
deg['NLRoot']= np.zeros(len(dictNLRoots.keys()))
deg['ASTNode']= np.zeros(len(dictASTNodes.keys()))
deg['NLNode']= np.zeros(len(dictNLNodes.keys()))
print('deg {}\n{}\n{}'.format(deg.keys(),type(deg),len(deg['ASTNode'])))

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

fopTokenEmbed=fopInputEmbeddingModel+'token_emb/'
fopParagraphEmbed=fopInputEmbeddingModel+'paragraph_emb/'
lstFpTokenEmbed=sorted(glob.glob(fopTokenEmbed+'*.txt'))
lstFpParagraphEmbed=sorted(glob.glob(fopParagraphEmbed+'*.txt'))
dictVectorProgramRoot={}
dictVectorNLRoot={}
lengthOfVector=0
for fpParaItem in lstFpParagraphEmbed:
    f1=open(fpParaItem,'r')
    arrItemVectors=f1.read().strip().split('\n')
    f1.close()
    for i in range(0,len(arrItemVectors),2):
        arrPRInfo=arrItemVectors[i].split('\t')
        arrNLRInfo=arrItemVectors[i+1].split('\t')
        if(len(arrPRInfo)>=4 and len(arrNLRInfo)>=4):
            strId=arrNLRInfo[1]+'-'+arrNLRInfo[2]
            # print('idididd {} {}'.format(i,strId))
            dictVectorProgramRoot['ProgramRoot_'+strId]=[float(item) for item in arrPRInfo[3].split()]
            dictVectorNLRoot['NLRoot_'+strId]=[float(item) for item in arrNLRInfo[3].split()]
            if lengthOfVector==0:
                lengthOfVector=len(dictVectorProgramRoot['ProgramRoot_'+strId])
lstVectorPRs=[]
for i in range(0,len(dictProgramRoots.keys())):
    key=list(dictProgramRoots.keys())[i]
    # val=dictProgramRoots[key]
    lstVectorPRs.append(dictVectorProgramRoot[key])

lstVectorNLRs=[]
for i in range(0,len(dictNLRoots.keys())):
    key=list(dictNLRoots.keys())[i]
    # val=dictNLRoots[key]
    lstVectorNLRs.append(dictVectorNLRoot[key])

dictVectorTokens={}
for fpTokenItem in lstFpTokenEmbed:
    f1=open(fpTokenItem,'r')
    arrItemVectors=f1.read().strip().split('\n')
    f1.close()
    for i in range(0,len(arrItemVectors)):
        arrTabs=arrItemVectors[i].split('\t')
        if len(arrTabs)>=2:
            dictVectorTokens[arrTabs[0]]=[float(item) for item in arrTabs[1].split()]

lstVectorASTNode=[]
for i in range(0,len(dictASTNodes.keys())):
    strKey=list(dictASTNodes.keys())[i]
    if strKey in dictVectorTokens.keys():
        lstVectorASTNode.append(dictVectorTokens[strKey])
    else:
        lstVectorASTNode.append(np.zeros(lengthOfVector).tolist())

lstVectorNLNode=[]
for i in range(0,len(dictNLNodes.keys())):
    strKey=list(dictNLNodes.keys())[i]
    if strKey in dictVectorTokens.keys():
        lstVectorNLNode.append(dictVectorTokens[strKey])
    else:
        lstVectorNLNode.append(np.zeros(lengthOfVector).tolist())

npArrayPRs=np.array(lstVectorPRs)
npArrayNLRs=np.array(lstVectorNLRs)
npArrayASTNodes=np.array(lstVectorASTNode)
npArrayNLNodes=np.array(lstVectorNLNode)

graph.node_feature['ProgramRoot'] = np.concatenate((npArrayPRs, np.log10(deg['ProgramRoot'].reshape(-1, 1))), axis=-1)
graph.node_feature['NLRoot'] =np.concatenate((npArrayNLRs, np.log10(deg['NLRoot'].reshape(-1, 1))), axis=-1)
graph.node_feature['ASTNode'] = np.concatenate((npArrayASTNodes, np.log10(deg['ASTNode'].reshape(-1, 1))), axis=-1)
graph.node_feature['NLNode'] =np.concatenate((npArrayNLNodes, np.log10(deg['NLNode'].reshape(-1, 1))), axis=-1)



# cv = data.x_dict['paper'].numpy()
# graph.node_feature['paper'] = np.concatenate((cv, np.log10(deg['paper'].reshape(-1, 1))), axis=-1)
# for _type in data.num_nodes:
#     print(_type)
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
# cv = graph.node_feature['author'][:, :-1]
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



# y = data.y_dict['paper'].t().numpy()[0]
# split_idx = dataset.get_idx_split()
# train_paper = split_idx['train']['paper'].numpy()
# valid_paper = split_idx['valid']['paper'].numpy()
# test_paper  = split_idx['test']['paper'].numpy()

y=np.asarray(lstIdxLabels)
# print('type {}'.format(type(y)))
split_idx = dataset.get_idx_split()

# train_paper = split_idx['train']['paper'].numpy()
# valid_paper = split_idx['valid']['paper'].numpy()
# test_paper  = split_idx['test']['paper'].numpy()
# print('type train_paper {} {}'.format(valid_paper[10],type(train_paper)))

lstNLR=range(0,len(dictNLRoots.keys()))
trainEndIndex=dictRangeTrainTest['train'][1]+1
validStartIndex=dictRangeTrainTest['testP'][0]
validEndIndex=dictRangeTrainTest['testP'][1]+1
testStartIndex=dictRangeTrainTest['testW'][0]
testEndIndex=dictRangeTrainTest['testW'][1]+1

# trainEndIndex=int(len(lstNLR)*0.8)
# validStartIndex=int(len(lstNLR)*0.8)
# validEndIndex=int(len(lstNLR)*0.9)
# testStartIndex=int(len(lstNLR)*0.9)
# testEndIndex=len(lstNLR)
train_paper = np.asarray(lstNLR[:trainEndIndex])
valid_paper = np.asarray(lstNLR[validStartIndex:validEndIndex])
test_paper  = np.asarray(lstNLR[testStartIndex:testEndIndex])

graph.y = y
graph.train_paper = train_paper
graph.valid_paper = valid_paper
graph.test_paper  = test_paper
graph.years       = years

graph.train_mask = np.zeros(len(graph.node_feature['NLRoot']), dtype=bool)
graph.train_mask[graph.train_paper] = True

graph.valid_mask = np.zeros(len(graph.node_feature['NLRoot']), dtype=bool)
graph.valid_mask[graph.valid_paper] = True

graph.test_mask = np.zeros(len(graph.node_feature['NLRoot']),  dtype=bool)
graph.test_mask[graph.test_paper] = True

dill.dump(graph, open(args.output_dir, 'wb'))
