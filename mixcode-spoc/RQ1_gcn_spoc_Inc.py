import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv
import numpy as np
from UtilFunctions_RTX3090 import createDirIfNotExist
import argparse
import os
import glob
import traceback
import sys
import time
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import numpy as np
from UtilFunctions_RTX3090 import createDirIfNotExist
import argparse
import os
import glob
import traceback
import sys
import time

class Net(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    resultData = None
    resultPred = None
    index=0
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
        index=index+1
        # print('{} {} {} {}'.format(mask,type(pred),type(data.y[mask]),pred))
        if index==3:
            resultData = data.y[mask]
            resultPred = pred
            accs.append(resultData.tolist())
            accs.append(resultPred.tolist())
    return accs


# begin load SPoC data
strSplitCharacterForNodeEdge='_ABAZ_'
parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--input_mixgraph_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step5_totalGraph_small/AAAAAA/',
                    help='The address of input spoc dataset.')
parser.add_argument('--input_embedding_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/embeddingModels/d2v/',
                    help='The address of pretrained embedding model.')
# parser.add_argument('--output_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step6_hgt_problem2/1/mixcode_spoc.pk',
#                     help='The address to output the preprocessed graph.')


args = parser.parse_args()

fopInputMixGraph=args.input_mixgraph_dir
fopInputEmbeddingModel=args.input_embedding_dir
fopRoot='/home/hungphd/media/dataPapersExternal/mixCodeRaw/'
# fopLog=fopRoot+'step7_gnn_homo/agnn/'
fopStep5HGT=fopRoot+'step5_Inconsistent/'
fopStep3V2=fopRoot+'step3_v2/'
fopInputEmbeddingModel=fopRoot+'embeddingModels/fasttext-cbow/'
fopResult=fopRoot+'step7_Inconsistent/gcn/'
fpResult=fopResult+'a_agnn_result.txt'

createDirIfNotExist(fopResult)

# fpOutputGraph=args.output_dir
# fopOutputGraph=os.path.dirname(fpOutputGraph)
# createDirIfNotExist(fopOutputGraph)

lstProblemIds=['label.p1.overlap.txt']
lstContexts=[0]


f1=open(fpResult,'w')
f1.write('')
f1.close()

for problemId in lstProblemIds:
    dictTotalResults = {}
    strContext=lstContexts[0]
    fopInputMixGraph = fopStep5HGT+'context_{}_pos_stanford/'.format(strContext)
    createDirIfNotExist(fopInputMixGraph)
    fpLogItem=fopResult+'log.txt'.format(problemId,strContext)
    # fnLabelFile='embed_fastext-cbow_context_1000_pos_stanford/'
    fpOutputGraph = fopResult + 'graph.pk'
    # sys.stdout = open(fpLogItem, 'w')

    fpDictLiterals = fopRoot + 'step2_dictLiterals_all.txt'
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
    fpErrorLogs = fopStep3V2 + 'log_error.txt'
    f1 = open(fpErrorLogs, 'r')
    arrErrors = f1.read().strip().split('\n')
    f1.close()
    setErrorLocations = set()
    for error in arrErrors:
        setErrorLocations.add(error)
    fpTrainValidTestIndex = fopStep3V2 + 'trainValidTest.index.txt'
    f1 = open(fpTrainValidTestIndex, 'r')
    arrTVTs = f1.read().strip().split('\n')
    f1.close()
    f1 = open(fopStep3V2 + problemId, 'r')
    arrLabels = f1.read().strip().split('\n')
    f1.close()
    dictTrainTestIndexes = {}
    dictLabelsFromLocations = {}
    for i in range(0, len(arrTVTs)):
        item = arrTVTs[i]
        arrItemTabs = item.split('\t')
        arrLblTabs = arrLabels[i].split('\t')
        strKeyCorrect = '1__' + arrItemTabs[2] + '__' + arrItemTabs[1] + '__' + arrItemTabs[3]
        strKeyIncorrect = '0__' + arrItemTabs[2] + '__' + arrItemTabs[1] + '__' + arrItemTabs[3]
        # print(strKey)
        dictTrainTestIndexes[strKeyCorrect] = arrItemTabs[0]
        dictLabelsFromLocations[strKeyCorrect] = '1'
        dictTrainTestIndexes[strKeyIncorrect] = arrItemTabs[0]
        dictLabelsFromLocations[strKeyIncorrect] = '0'

    dictProgramRoots = {}
    dictASTNodes = {}
    dictNLRoots = {}
    dictNLNodes = {}

    f1 = open(fpNodeProgramRoot, 'r')
    arrPRs = f1.read().strip().split('\n')
    f1.close()
    dictNLRoots = {}
    dictLabelsTextToInt = {}
    dictAllNodesAndIds = {}
    lstIdxLabels = []
    for i in range(0, len(arrPRs)):
        item = arrPRs[i]
        # print(item)
        arrItemContent = item.split(strSplitCharacterForNodeEdge)
        if len(arrItemContent) >= 2:
            arrSplitbyHy = arrItemContent[0].split('__')
            strRealKeyFile = arrSplitbyHy[1] + '__' + arrSplitbyHy[2] + '__' + arrSplitbyHy[3]
            if strRealKeyFile in setErrorLocations:
                continue
            id = len(dictNLRoots.keys())
            strProgramId = arrItemContent[0]
            strLabel = dictLabelsFromLocations[strProgramId]
            if strLabel not in dictLabelsTextToInt.keys():
                dictLabelsTextToInt[strLabel] = len(dictLabelsTextToInt.keys())
            idxLabel = dictLabelsTextToInt[strLabel]
            lstIdxLabels.append(idxLabel)
            tup = [id, arrItemContent[0], idxLabel]
            dictNLRoots['NLRoot_' + arrItemContent[0]] = tup
            strInfoAboutNodeHomo = 'NLRoot_' + arrItemContent[0]
            if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
                dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())

    dictProgramRoots = {}
    dictRangeTrainTest = {}
    prevTrainTest = ''
    for i in range(0, len(arrPRs)):
        item = arrPRs[i]
        arrItemContent = item.split(strSplitCharacterForNodeEdge)
        if len(arrItemContent) >= 2:
            arrSplitbyHy = arrItemContent[0].split('__')
            strRealKeyFile = arrSplitbyHy[1] + '__' + arrSplitbyHy[2] + '__' + arrSplitbyHy[3]
            if strRealKeyFile in setErrorLocations:
                continue
            id = len(dictProgramRoots.keys())
            strProgramId = arrItemContent[0]
            strLabel = dictLabelsFromLocations[strProgramId]
            # if strLabel not in dictLabelsTextToInt.keys():
            #     dictLabelsTextToInt[strLabel] = len(dictLabelsTextToInt.keys())
            idxLabel = dictLabelsTextToInt[strLabel]
            lstIdxLabels.append(idxLabel)
            tup = [id, arrItemContent[0], idxLabel]
            dictProgramRoots['ProgramRoot_' + arrItemContent[0]] = tup
            strInfoAboutNodeHomo = 'ProgramRoot_' + arrItemContent[0]
            if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
                dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())

    f1 = open(fpNodeASTNode, 'r')
    arrASTNodes = f1.read().strip().split('\n')
    f1.close()
    dictASTNodes = {}
    for i in range(0, len(arrASTNodes)):
        item = arrASTNodes[i]
        id = len(dictASTNodes.keys())
        dictASTNodes[item] = [id]
        strInfoAboutNodeHomo = 'ASTNode_' + item
        if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
            dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())
        lstIdxLabels.append(0)

    f1 = open(fpNodeNLNode, 'r')
    arrNLNodes = f1.read().strip().split('\n')
    f1.close()
    dictNLNodes = {}
    for i in range(0, len(arrNLNodes)):
        item = arrNLNodes[i]
        id = len(dictNLNodes.keys())
        dictNLNodes[item] = [id]
        strInfoAboutNodeHomo = 'NLNode_' + item
        if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
            dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())
        lstIdxLabels.append(0)
    all_labels = torch.tensor(lstIdxLabels)

    dictAllNodes = {}
    dictAllNodes['ProgramRoot'] = dictProgramRoots
    dictAllNodes['NLRoot'] = dictNLRoots
    dictAllNodes['ASTNode'] = dictASTNodes
    dictAllNodes['NLNode'] = dictNLNodes

    # fopTokenEmbed=fopInputEmbeddingModel+'token_emb/'
    # fopParagraphEmbed=fopInputEmbeddingModel+'paragraph_emb/'
    # lstFpTokenEmbed=sorted(glob.glob(fopTokenEmbed+'*.txt'))
    # lstFpParagraphEmbed=sorted(glob.glob(fopParagraphEmbed+'*.txt'))
    # dictVectorProgramRoot={}
    # dictVectorNLRoot={}
    fpVectorNLRoot = fopInputEmbeddingModel + 'embInconsistent.txt'
    fpVectorProgramRoot = fopInputEmbeddingModel + 'embInconsistent.txt'
    fpVectorNode = fopInputEmbeddingModel + 'Node.vectorForEmb.txt'
    dictVectorProgramRoot = {}
    dictVectorNLRoot = {}
    lengthOfVector = 0

    f1 = open(fpVectorProgramRoot, 'r')
    arrVectorPRs = f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpVectorNLRoot, 'r')
    arrVectorNLRs = f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpVectorNode, 'r')
    arrVectorNodes = f1.read().strip().split('\n')
    f1.close()

    dictAllVectors = {}
    lengthOfVector = 0
    for i in range(0, len(arrVectorPRs)):
        arrTabPR = arrVectorPRs[i].split('\t')
        arrTabNLR = arrVectorNLRs[i].split('\t')
        strProgramDetailId = arrTabPR[4] + '__' + arrTabPR[2] + '__' + arrTabPR[1] + '__' + arrTabPR[3]
        strProgramId = arrTabPR[2] + '__' + arrTabPR[1] + '__' + arrTabPR[3]
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
        dictAllVectors['ProgramRoot_' + key] = dictVectorProgramRoot[key]

    lstVectorNLRs = []
    for i in range(0, len(dictNLRoots.keys())):
        key = list(dictNLRoots.keys())[i]
        # val=dictNLRoots[key]
        lstVectorNLRs.append(dictVectorNLRoot[key])
        dictAllVectors['NLRoot_' + key] = dictVectorNLRoot[key]

    print('size {}'.format(len(dictAllVectors.keys())))
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
            dictAllVectors['ASTNode_' + strKey] = dictVectorTokens[strKey]
        else:
            lstVectorASTNode.append(np.zeros(lengthOfVector).tolist())
            dictAllVectors['ASTNode_' + strKey] = np.zeros(lengthOfVector).tolist()

    print('size {}'.format(len(dictAllVectors.keys())))

    lstVectorNLNode = []
    for i in range(0, len(dictNLNodes.keys())):
        strKey = list(dictNLNodes.keys())[i]
        if strKey in dictVectorTokens.keys():
            lstVectorNLNode.append(dictVectorTokens[strKey])
            dictAllVectors['NLNode_' + strKey] = dictVectorTokens[strKey]
        else:
            lstVectorNLNode.append(np.zeros(lengthOfVector).tolist())
            dictAllVectors['NLNode_' + strKey] = np.zeros(lengthOfVector).tolist()

    # npArrayPRs=np.array(lstVectorPRs).astype(np.float32)
    # npArrayNLRs=np.array(lstVectorNLRs).astype(np.float32)
    # npArrayASTNodes=np.array(lstVectorASTNode).astype(np.float32)
    # npArrayNLNodes=np.array(lstVectorNLNode).astype(np.float32)
    listAllVectors = list(dictAllVectors.values())
    npArrayAllVectors = np.array(listAllVectors).astype(np.float32)

    dict_edge_index = {}
    arrayLoop = [[], []]

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
                    if s_type == 'ProgramRoot':
                        strTextSource = arrTabsItem[2]
                    elif s_type == 'NLRoot':
                        strTextSource = arrTabsItem[2]
                    if t_type == 'ProgramRoot':
                        strTextTarget = arrTabsItem[2]
                    elif t_type == 'NLRoot':
                        strTextTarget = arrTabsItem[2]

                    if strTextSource in dictValuesToLiterals.keys():
                        strTextSource = dictValuesToLiterals[strTextSource]
                    if strTextTarget in dictValuesToLiterals.keys():
                        strNewTextTarget = dictValuesToLiterals[strTextTarget]
                        # print('text target {} AAAA {}'.format(strTextTarget,strNewTextTarget))
                        strTextTarget = strNewTextTarget

                    if (strTextSource == '' or strTextTarget == ''):
                        continue
                    if strTextSource not in dictAllNodes[s_type].keys() or strTextTarget not in dictAllNodes[
                        t_type].keys():
                        continue
                    # print('come to here')
                    if strTextSource != 'translation_unit':
                        s_id = dictAllNodesAndIds[s_type + '_' + strTextSource]
                    else:
                        strRootKey = arrTabsItem[2].split('\t')[1]
                        s_id = dictAllNodesAndIds[s_type + '_' + strRootKey]
                    # s_id=dictAllNodes[s_type][arrTabsItem[0]][0]
                    # print('t_type {} BBB {} AAA {}'.format(t_type,arrTabsItem[1],strTextTarget))
                    t_id = dictAllNodesAndIds[t_type + '_' + strTextTarget]
                    # year=int(arrTabsItem[3])
                    # year = 2020
                    # elist[t_id][s_id] = year
                    # rlist[s_id][t_id] = year
                    arrayLoop[0].append(s_id)
                    arrayLoop[1].append(t_id)
                except:
                    traceback.print_exc()
                    quit()
        # data[tupEdgeLabel[0], tupEdgeLabel[1], tupEdgeLabel[2]].edge_index = torch.tensor(arrayLoop)
        # dict_edge_index[tupEdgeLabel]=torch.tensor(arrayLoop)
        # print('end edge {}'.format(fpEdge))
    all_edge_index = torch.tensor(arrayLoop)

    lstTrainMask = []
    lstValidMask = []
    lstTestMask = []

    for i in range(0, len(dictAllNodesAndIds.keys())):
        strProgramId = list(dictAllNodesAndIds.keys())[i].replace('NLRoot_', '')
        if strProgramId not in dictTrainTestIndexes.keys():
            lstTrainMask.append(False)
            lstValidMask.append(False)
            lstTestMask.append(False)
        elif dictTrainTestIndexes[strProgramId] == 'train':
            lstTrainMask.append(True)
            lstValidMask.append(False)
            lstTestMask.append(False)
        elif dictTrainTestIndexes[strProgramId] == 'valid':
            lstTrainMask.append(False)
            lstValidMask.append(True)
            lstTestMask.append(False)
        elif dictTrainTestIndexes[strProgramId] == 'test':
            lstTrainMask.append(False)
            lstValidMask.append(False)
            lstTestMask.append(True)

    train_mask = torch.tensor(lstTrainMask)
    val_mask = torch.tensor(lstValidMask)
    test_mask = torch.tensor(lstTestMask)

    # end load SPoC data
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.x=torch.tensor(npArrayAllVectors)
    data.y=torch.tensor(all_labels)
    data.edge_index=all_edge_index
    data.train_mask=train_mask
    data.val_mask=val_mask
    data.test_mask=test_mask

    print(data)
    print('{}\t{}'.format(len(data.x),len(data.train_mask)))
    # input('press key to move forward')
    # sys.stdout=open(fpResult,'w')
    num_classes=len(dictLabelsTextToInt.keys())
    if args.use_gdc:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=128,
                                               dim=0), exact=True)
        data = gdc(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(num_features=lengthOfVector, num_classes=num_classes).to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.

    best_test_acc = test_acc = 0
    listTestExpected=[]
    listTestPredicted=[]
    start_time=time.time()
    for epoch in range(1, 5001):
        train()
        train_acc, val_acc, test_acc,listCanTestExpected, listCanTestPredicted = test()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            listTestExpected=listCanTestExpected
            listTestPredicted=listCanTestPredicted
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    end_time=time.time()
    train_time=end_time-start_time
    start_time=time.time()
    test()
    end_time=time.time()
    test_time=end_time-start_time
    strKey='context_{}'.format(strContext)
    dictTotalResults[strKey]=[best_test_acc,val_acc,train_acc,train_time,test_time]

    acc = accuracy_score(listTestExpected, listTestPredicted)
    precision = precision_score(listTestExpected, listTestPredicted, average='weighted')
    recall = recall_score(listTestExpected, listTestPredicted, average='weighted')
    fscore = f1_score(listTestExpected, listTestPredicted, average='weighted')

    fpResultToExcel=fopResult+'toexcel.txt'
    fpResultPredictExpect = fopResult + 'predict-expect.txt'
    f1 = open(fpResultToExcel, 'w')
    f1.write(
        '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('fasttext-cbow', 1000, 'stanford', precision, recall, fscore, acc, train_time,
                                                    test_time) + '\n')
    f1.close()
    lstStrTemp=[]
    for q in range(0,len(listTestExpected)):
        lstStrTemp.append('{}\t{}'.format(listTestExpected[q],listTestPredicted[q]))
    f1=open(fpResultPredictExpect,'w')
    f1.write('\n'.join(lstStrTemp))
    f1.close()
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__

    # lstSorted=sorted([(dictTotalResults[x][0], x) for x in dictTotalResults.keys()],reverse=True)
    # f1=open(fpResult,'a')
    # f1.write('Problem {}\n'.format(problemId))
    # for val,key in lstSorted:
    #     val=dictTotalResults[key]
    #     strValue='context {}: {}'.format(key,'\t'.join(map(str,val)))
    #     f1.write(strValue+'\n')
    # f1.close()








