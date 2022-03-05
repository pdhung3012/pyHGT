import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear, HGTConv
from UtilFunctions_RTX3090 import createDirIfNotExist
import argparse
import numpy as np
import glob
from sentence_transformers import SentenceTransformer
import os,traceback

from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
import sys
import time
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score


# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
# dataset = DBLP(path)
# data = dataset[0]
# print(data)
# print(type(data))
# print(type(data.edge_index_dict.keys()))
# for key in data.edge_index_dict.keys():
#     print(type(key))
#     print(data.edge_index_dict[key][2][15])
# print(data.edge_index_dict.keys())
# # We initialize conference node features with a single feature.
# data['conference'].x = torch.ones(data['conference'].num_nodes, 1)

def loadHGTGraph(fopInputMixGraph ,fopInputEmbeddingModel,fopStep3V2,fopOutputGraph,fpDictLiterals,fnLabelFile):
    # fopInputMixGraph = args.input_mixgraph_dir.replace('BBBBB', strContext)
    # fopInputEmbeddingModel = args.input_embedding_dir
    fpOutputGraph =fopOutputGraph+'graph.pk'
    # fopOutputGraph = os.path.dirname(fpOutputGraph)


    # fpDictLiterals = fopRoot + 'step2_dictLiterals_all.txt'
    fpNodeProgramRoot = fopInputMixGraph + 'nodes_ProgramRoot.txt'
    fpNodeASTNode = fopInputMixGraph + 'nodes_ASTNode.txt'
    fpNodeNLRoot = fopInputMixGraph + 'nodes_NLRoot.txt'
    fpNodeNLNode = fopInputMixGraph + 'nodes_NLNode.txt'
    lstFpEdgesList = glob.glob(fopInputMixGraph + 'edges_*.txt')

    f1 = open(fpDictLiterals, 'r')
    arrLits = f1.read().strip().split('\n')
    f1.close()
    dictLiteralsToValues = {}
    dictValuesToLiterals = {}
    for item in arrLits:
        arrTabs = item.split('\t')
        if len(arrTabs) >= 2:
            strContent = '\t'.join(arrTabs[1:])
            dictLiteralsToValues[arrTabs[0]] = strContent
            dictValuesToLiterals[strContent] = arrTabs[0]

    # load set of error files
    fpErrorLogs=fopStep3V2+'log_error.txt'
    f1=open(fpErrorLogs,'r')
    arrErrors=f1.read().strip().split('\n')
    f1.close()
    setErrorLocations=set()
    for error in arrErrors:
        setErrorLocations.add(error)
    fpTrainValidTestIndex=fopStep3V2+'trainValidTest.index.txt'
    f1=open(fpTrainValidTestIndex,'r')
    arrTVTs=f1.read().strip().split('\n')
    f1.close()
    f1=open(fopStep3V2+fnLabelFile,'r')
    arrLabels=f1.read().strip().split('\n')
    f1.close()
    dictTrainTestIndexes = {}
    dictLabelsFromLocations={}
    for i in range(0,len(arrTVTs)) :
        item=arrTVTs[i]
        arrItemTabs=item.split('\t')
        arrLblTabs=arrLabels[i].split('\t')
        strKeyCorrect='1__'+arrItemTabs[2]+'__'+arrItemTabs[1]+'__'+arrItemTabs[3]
        strKeyIncorrect = '0__' + arrItemTabs[2] + '__' + arrItemTabs[1] + '__' + arrItemTabs[3]
        # print(strKey)
        dictTrainTestIndexes[strKeyCorrect]=arrItemTabs[0]
        dictLabelsFromLocations[strKeyCorrect]='1'
        dictTrainTestIndexes[strKeyIncorrect]=arrItemTabs[0]
        dictLabelsFromLocations[strKeyIncorrect]='0'

    dictProgramRoots = {}
    dictASTNodes = {}
    dictNLRoots = {}
    dictNLNodes = {}

    f1 = open(fpNodeProgramRoot, 'r')
    arrPRs = f1.read().strip().split('\n')
    f1.close()
    # print('len PR {}'.format(len(arrPRs)))
    dictProgramRoots = {}
    # dictRangeTrainTest = {}
    # prevTrainTest = ''
    dictLabelsTextToInt = {}
    # lstIdxLabels=[]
    dictCountValueInLabel = {}
    for i in range(0, len(arrPRs)):
        item = arrPRs[i]
        # print(item)
        arrItemContent = item.split(strSplitCharacterForNodeEdge)
        if len(arrItemContent) >= 2:
            # print(arrItemContent[0])
            arrSplitbyHy=arrItemContent[0].split('__')
            strRealKeyFile=arrSplitbyHy[1]+'__'+arrSplitbyHy[2]+'__'+arrSplitbyHy[3]
            if strRealKeyFile in setErrorLocations:
                continue
            id = len(dictProgramRoots.keys())
            strProgramId = arrItemContent[0]
            strLabel = dictLabelsFromLocations[strProgramId]
            if strLabel not in dictLabelsTextToInt.keys():
                dictLabelsTextToInt[strLabel] = len(dictLabelsTextToInt.keys())
                dictCountValueInLabel[strLabel] = 0
            dictCountValueInLabel[strLabel] = dictCountValueInLabel[strLabel] + 1
            idxLabel = dictLabelsTextToInt[strLabel]
            # lstIdxLabels.append(idxLabel)
            tup = [id, arrItemContent[0], idxLabel]
            dictProgramRoots['ProgramRoot_'+arrItemContent[0]] = tup
    # print(dictProgramRoots.keys())
    # f1 = open(fpNodeNLRoot, 'r')
    # arrNLRs = f1.read().strip().split('\n')
    # f1.close()
    dictNLRoots = {}
    lstYears = []
    lstIdxLabels = []
    for i in range(0, len(arrPRs)):
        item = arrPRs[i]
        arrItemContent = item.split(strSplitCharacterForNodeEdge)
        if len(arrItemContent) >= 2:
            arrSplitbyHy = arrItemContent[0].split('__')
            strRealKeyFile = arrSplitbyHy[1] + '__' + arrSplitbyHy[2] + '__' + arrSplitbyHy[3]
            if strRealKeyFile in setErrorLocations:
                continue
            id = len(dictNLRoots.keys())
            strProgramId = arrItemContent[0]
            strLabel = dictLabelsFromLocations[strProgramId]
            # if strLabel not in dictLabelsTextToInt.keys():
            #     dictLabelsTextToInt[strLabel]=len(dictLabelsTextToInt.keys())+1
            idxLabel = dictLabelsTextToInt[strLabel]
            lstIdxLabels.append(idxLabel)
            tup = [id, arrItemContent[0], idxLabel]
            dictNLRoots['NLRoot_'+arrItemContent[0]] = tup
            lstYears.append(2020)
    years = np.asarray(lstYears)

    f1 = open(fpNodeASTNode, 'r')
    arrASTNodes = f1.read().strip().split('\n')
    f1.close()
    dictASTNodes = {}
    for i in range(0, len(arrASTNodes)):
        item = arrASTNodes[i]
        id = len(dictASTNodes.keys())
        dictASTNodes[item] = [id]

    f1 = open(fpNodeNLNode, 'r')
    arrNLNodes = f1.read().strip().split('\n')
    f1.close()
    dictNLNodes = {}
    for i in range(0, len(arrNLNodes)):
        item = arrNLNodes[i]
        # print(item+' here')
        id = len(dictNLNodes.keys())
        dictNLNodes[item] = [id]

    dictAllNodes = {}
    dictAllNodes['ProgramRoot'] = dictProgramRoots
    dictAllNodes['NLRoot'] = dictNLRoots
    dictAllNodes['ASTNode'] = dictASTNodes
    dictAllNodes['NLNode'] = dictNLNodes

    # fopTokenEmbed = fopInputEmbeddingModel + 'token_emb/'
    # fopParagraphEmbed = fopInputEmbeddingModel + 'paragraph_emb/'
    # lstFpTokenEmbed = sorted(glob.glob(fopTokenEmbed + '*.txt'))
    # lstFpParagraphEmbed = sorted(glob.glob(fopParagraphEmbed + '*.txt'))
    fpVectorNLRoot=fopInputEmbeddingModel+'embInconsistent.txt'
    fpVectorProgramRoot = fopInputEmbeddingModel + 'embInconsistent.txt'
    fpVectorNode = fopInputEmbeddingModel + 'Node.vectorForEmb.txt'
    dictVectorProgramRoot = {}
    dictVectorNLRoot = {}
    lengthOfVector = 0

    f1=open(fpVectorProgramRoot,'r')
    arrVectorPRs=f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpVectorNLRoot, 'r')
    arrVectorNLRs = f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpVectorNode, 'r')
    arrVectorNodes = f1.read().strip().split('\n')
    f1.close()

    for i in range(0, len(arrVectorPRs)):
        arrTabPR=arrVectorPRs[i].split('\t')
        arrTabNLR = arrVectorNLRs[i].split('\t')
        strProgramDetailId=arrTabPR[4]+'__'+arrTabPR[2]+'__'+arrTabPR[1]+'__'+arrTabPR[3]
        strProgramId=arrTabPR[2]+'__'+arrTabPR[1]+'__'+arrTabPR[3]
        # print(strProgramDetailId)
        if (strProgramId not in setErrorLocations):
            # print('idididd {} {}'.format(i,strId))
            dictVectorProgramRoot['ProgramRoot_' + strProgramDetailId] = [float(item) for item in arrTabPR[5].split()]
            dictVectorNLRoot['NLRoot_' + strProgramDetailId] = [float(item) for item in arrTabNLR[5].split()]
            if lengthOfVector == 0:
                lengthOfVector = len(dictVectorProgramRoot['ProgramRoot_' + strProgramDetailId])

    lstVectorPRs = []
    for i in range(0, len(dictProgramRoots.keys())):
        key = list(dictProgramRoots.keys())[i]
        # val=dictProgramRoots[key]
        # print(key)
        lstVectorPRs.append(dictVectorProgramRoot[key])

    lstVectorNLRs = []
    for i in range(0, len(dictNLRoots.keys())):
        key = list(dictNLRoots.keys())[i]
        # val=dictNLRoots[key]
        lstVectorNLRs.append(dictVectorNLRoot[key])

    dictVectorTokens = {}
    f1 = open(fpVectorNode, 'r')
    arrItemVectors = f1.read().strip().split('\n')
    f1.close()
    for i in range(0, len(arrItemVectors)):
        arrTabs = arrItemVectors[i].split('\t')
        if len(arrTabs) >= 2:
            dictVectorTokens[arrTabs[0]] = [float(item) for item in arrTabs[1].split()]

    lstVectorASTNode = []
    for i in range(0, len(dictASTNodes.keys())):
        strKey = list(dictASTNodes.keys())[i]
        if strKey in dictVectorTokens.keys():
            lstVectorASTNode.append(dictVectorTokens[strKey])
        else:
            lstVectorASTNode.append(np.zeros(lengthOfVector).tolist())

    lstVectorNLNode = []
    for i in range(0, len(dictNLNodes.keys())):
        strKey = list(dictNLNodes.keys())[i]
        if strKey in dictVectorTokens.keys():
            lstVectorNLNode.append(dictVectorTokens[strKey])
        else:
            lstVectorNLNode.append(np.zeros(lengthOfVector).tolist())

    npArrayPRs = np.array(lstVectorPRs).astype(np.float32)
    npArrayNLRs = np.array(lstVectorNLRs).astype(np.float32)
    npArrayASTNodes = np.array(lstVectorASTNode).astype(np.float32)
    npArrayNLNodes = np.array(lstVectorNLNode).astype(np.float32)

    data = HeteroData()
    data['ProgramRoot'].x = torch.tensor(npArrayPRs)
    data['NLRoot'].x = torch.tensor(npArrayNLRs)
    data['ASTNode'].x = torch.tensor(npArrayASTNodes)
    data['NLNode'].x = torch.tensor(npArrayNLNodes)

    dict_edge_index = {}

    for i in range(0, len(lstFpEdgesList)):
        fpEdge = lstFpEdgesList[i]
        f1 = open(fpEdge, 'r')
        arrEdges = f1.read().strip().split('\n')
        f1.close()
        # print('begin edge {}'.format(fpEdge))
        nameFileEdge = os.path.basename(fpEdge)
        strTypeForEdge = nameFileEdge.replace('edges_', '').replace('.txt', '')
        arrSourceTarget = strTypeForEdge.split(' - ')
        strSourceType = arrSourceTarget[0]
        strTargetType = arrSourceTarget[1]
        tupEdgeLabel = (strSourceType, 'to', strTargetType)
        arrayLoop = [[], []]
        arrayLoopReverse = [[], []]

        for line in arrEdges:
            arrTabsItem = line.split(strSplitCharacterForNodeEdge)
            if len(arrTabsItem) >= 3:
                try:
                    s_type = strSourceType
                    t_type = strTargetType
                    r_type = strTypeForEdge
                    # elist = edg[t_type][s_type][r_type]
                    # rlist = edg[s_type][t_type]['rev_' + r_type]
                    strTextSource = arrTabsItem[0]
                    strTextTarget = arrTabsItem[1]
                    if arrTabsItem[2] in setErrorLocations:
                        continue
                    if s_type=='ProgramRoot':
                        strTextSource ='ProgramRoot_'+arrTabsItem[2]
                    elif s_type=='NLRoot':
                        strTextSource = 'NLRoot_' + arrTabsItem[2]
                    if t_type=='ProgramRoot':
                        strTextTarget ='ProgramRoot_'+arrTabsItem[2]
                    elif t_type=='NLRoot':
                        strTextTarget = 'NLRoot_' + arrTabsItem[2]

                    if strTextSource in dictValuesToLiterals.keys():
                        strTextSource = dictValuesToLiterals[strTextSource]
                    if strTextTarget in dictValuesToLiterals.keys():
                        strNewTextTarget = dictValuesToLiterals[strTextTarget]
                        # print('text target {} AAAA {}'.format(strTextTarget,strNewTextTarget))
                        strTextTarget = strNewTextTarget

                    if (strTextSource == '' or strTextTarget == ''):
                        continue
                    if strTextSource not in dictAllNodes[s_type].keys() or strTextTarget not in dictAllNodes[t_type].keys():
                        continue

                    if strTextSource != 'translation_unit':
                        s_id = dictAllNodes[s_type][strTextSource][0]
                    else:
                        strRootKey = arrTabsItem[2].split('\t')[1]
                        s_id = dictAllNodes[s_type][strRootKey][0]
                    # s_id=dictAllNodes[s_type][arrTabsItem[0]][0]
                    # print('t_type {} BBB {} AAA {}'.format(t_type,arrTabsItem[1],strTextTarget))
                    # print(t_type)
                    t_id = dictAllNodes[t_type][strTextTarget][0]
                    # year=int(arrTabsItem[3])
                    # year = 2020
                    # elist[t_id][s_id] = year
                    # rlist[s_id][t_id] = year
                    arrayLoop[0].append(s_id)
                    arrayLoop[1].append(t_id)
                    arrayLoopReverse[0].append(t_id)
                    arrayLoopReverse[1].append(s_id)
                except:
                    traceback.print_exc()
                    quit()
        data[tupEdgeLabel[0], tupEdgeLabel[1], tupEdgeLabel[2]].edge_index = torch.tensor(arrayLoop)
        # data[tupEdgeLabel[2], tupEdgeLabel[1], tupEdgeLabel[0]].edge_index = torch.tensor(arrayLoopReverse)
        # dict_edge_index[tupEdgeLabel]=torch.tensor(arrayLoop)

        # print('end edge {}'.format(fpEdge))

    # lstTrainMask=range(0,len(dictNLRoots.keys()))
    # lstValidMask=range(0,len(dictNLRoots.keys()))
    # lstTestMask=range(0,len(dictNLRoots.keys()))

    # print(dictRangeTrainTest)
    # trainEndIndex=dictRangeTrainTest['train'][1]+1
    # validStartIndex=dictRangeTrainTest['testP'][0]
    # validEndIndex=dictRangeTrainTest['testP'][1]+1
    # testStartIndex=dictRangeTrainTest['testW'][0]
    # testEndIndex=dictRangeTrainTest['testW'][1]+1

    lstTrainMask = []
    lstValidMask = []
    lstTestMask = []
    # print(dictTrainTestIndexes)
    # input('bbbb ')
    for i in range(0, len(dictNLRoots.keys())):
        strProgramId=list(dictNLRoots.keys())[i].replace('NLRoot_','')
        if dictTrainTestIndexes[strProgramId]=='train':
            lstTrainMask.append(True)
            lstValidMask.append(False)
            lstTestMask.append(False)
        elif dictTrainTestIndexes[strProgramId]=='valid':
            lstTrainMask.append(False)
            lstValidMask.append(True)
            lstTestMask.append(False)
        elif dictTrainTestIndexes[strProgramId]=='test':
            lstTrainMask.append(False)
            lstValidMask.append(False)
            lstTestMask.append(True)

    # data = ToUndirected()(data)
    train_mask = torch.tensor(lstTrainMask)
    val_mask = torch.tensor(lstValidMask)
    test_mask = torch.tensor(lstTestMask)
    # print(val_mask)
    data['NLRoot'].train_mask = train_mask
    data['NLRoot'].val_mask = val_mask
    data['NLRoot'].test_mask = test_mask
    data['NLRoot'].y = torch.tensor(lstIdxLabels)
    # print(data)
    # input('aaa ')
    return data,dictCountValueInLabel


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['NLRoot'])


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    # print(data.x_dict)
    # print(type(data.x_dict))
    # input('aaa ')
    # print(data.edge_index_dict)
    # print(type(data.edge_index_dict))
    # input('bbb ')
    mask = data['NLRoot'].train_mask
    loss = F.cross_entropy(out[mask], data['NLRoot'].y[mask])
    # print('loss {}'.format(loss))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    resultData=None
    resultPred=None
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['NLRoot'][split]
        # print('mask {} {}'.format(split,len(pred[mask])))
        # print('element {}'.format(pred[mask][0]))
        acc = (pred[mask] == data['NLRoot'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
        if split == 'test_mask':
            resultData = data['NLRoot'].y[mask]
            resultPred = pred[mask]
            accs.append(resultData)
            accs.append(resultPred)
    return accs


strSplitCharacterForNodeEdge='_ABAZ_'
# parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')
#
# '''
#     Dataset arguments
# '''
# parser.add_argument('--input_mixgraph_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step5_origin/BBBBB/',
#                     help='The address of input spoc dataset.')
# parser.add_argument('--input_embedding_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/embeddingModels/d2v/',
#                     help='The address of pretrained embedding model.')
# parser.add_argument('--output_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step6_hgt_problem_origin/AAAAA/BBBBB/mixcode_spoc.pk',
#                     help='The address to output the preprocessed graph.')
#
# args = parser.parse_args()

fopRoot='/home/hungphd/media/dataPapersExternal/mixCodeRaw/'

fopStep5HGT=fopRoot+'step5_Inconsistent/'
fopStep3V2=fopRoot+'step3_v2/'
fopEmbedModel=fopRoot+'embeddingModels/'
fopResult=fopRoot+'step7_Inconsistent/hgt_origin/'

createDirIfNotExist(fopResult)

# fpOutputGraph=args.output_dir
# fopOutputGraph=os.path.dirname(fpOutputGraph)
# createDirIfNotExist(fopOutputGraph)

lstProblemIds=['label.p1.overlap.txt']
# lstContexts=['1','3','5','all']
lstContexts=[0]
lstEmbeddingModel=['fasttext-cbow']
lstPOS=['stanford']

fpDictLiterals=fopRoot+'step2_dictLiterals_all.txt'

for problem in lstProblemIds:
    nameProblem=problem.replace('.txt','').replace('label.','')
    fopItemProblem=fopResult+nameProblem+'/'
    createDirIfNotExist(fopItemProblem)
    fpResultOverall = fopItemProblem+  'problem1_overall.txt'
    fpResultToExcel = fopItemProblem + 'problem1_toexcel.txt'
    dictTotalResults = {}
    f1=open(fpResultToExcel,'w')
    f1.write('Embedding\tContext (LOC)\tPOS\tPrecision\tRecall\tF1-Score\tAccuracy\tTrain Time\tTest Time'+'\n')
    f1.close()


    for embed in lstEmbeddingModel:
        for context in lstContexts:
            for pos in lstPOS:
                nameConfig='embed_{}_context_{}_pos_{}'.format(embed,context,pos)
                fopInputMixGraph=fopStep5HGT+'context_{}_pos_{}'.format(context,pos)+'/'
                fopInputEmbeddingModel=fopEmbedModel+embed+'/'
                fopItemOutputGraph=fopItemProblem+nameConfig+'/'
                createDirIfNotExist(fopItemOutputGraph)
                fpLogItem = fopItemOutputGraph + 'log.txt'
                # sys.stdout = open(fpLogItem, 'w')
                data,dictCountValueInLabel=loadHGTGraph(fopInputMixGraph ,fopInputEmbeddingModel,fopStep3V2,fopItemOutputGraph,fpDictLiterals,problem)

                model = HGT(hidden_channels=64, out_channels= len(dictCountValueInLabel.keys()), num_heads=2, num_layers=1)
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print('{} value {}'.format(name, param.data))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # print('devide {}'.format(device))
                data, model = data.to(device), model.to(device)

                print(data)
                # input('aaaa')
                with torch.no_grad():  # Initialize lazy modules.
                    out = model(data.x_dict, data.edge_index_dict)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

                start_time = time.time()
                best_test_acc=test_acc=0
                bestTestData=None
                bestTestPredict=None
                for epoch in range(1, 501):
                    loss = train()
                    train_acc, val_acc, test_acc,resultData,resultPredict = test()
                    if best_test_acc<test_acc:
                        best_test_acc=test_acc
                        bestTestData = resultData
                        bestTestPredict = resultPredict
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                end_time = time.time()
                train_time = end_time - start_time
                start_time = time.time()
                test()
                end_time = time.time()
                test_time = end_time - start_time
                strKey = 'context_{}'.format(context)

                fpLogAccItem = fopItemOutputGraph+ 'logAccDet.txt'
                fpLogAccSumItem = fopItemOutputGraph + 'logAccSum.txt'
                listTestExpected=bestTestData.tolist()
                listTestPredicted = bestTestPredict.tolist()
                lstToStr=[]
                for q in range(0,len(listTestExpected)):
                    strItem='{}\t{}'.format(listTestExpected[q],listTestPredicted[q])
                    lstToStr.append(strItem)
                f1=open(fpLogAccItem,'w')
                f1.write('\n'.join(lstToStr))
                f1.close()
                strConfMatrixAndReport='{}\n{}'.format(str(classification_report(listTestExpected,listTestPredicted)),str(confusion_matrix(listTestExpected,listTestPredicted)))
                f1=open(fpLogAccSumItem,'w')
                f1.write(strConfMatrixAndReport)
                f1.close()

                acc = accuracy_score(listTestExpected, listTestPredicted)
                precision = precision_score(listTestExpected, listTestPredicted, average='weighted')
                recall = recall_score(listTestExpected, listTestPredicted, average='weighted')
                fscore = f1_score(listTestExpected, listTestPredicted, average='weighted')

                f1 = open(fpResultToExcel, 'a')
                f1.write(
                    '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(embed,context,pos,precision,recall,fscore,acc,train_time,test_time) + '\n')
                f1.close()

                sys.stdout.close()
                sys.stdout=sys.__stdout__

                dictTotalResults[strKey] = [best_test_acc, val_acc, train_acc, train_time, test_time]

    lstSorted = sorted([(dictTotalResults[x][0], x) for x in dictTotalResults.keys()],reverse=True)
    f1 = open(fpResultOverall, 'a')
    f1.write('Problem {}\n'.format(problem))
    for val, key in lstSorted:
        val = dictTotalResults[key]
        strValue = 'context {}: {}'.format(key, '\t'.join(map(str, val)))
        f1.write(strValue + '\n')
    f1.close()