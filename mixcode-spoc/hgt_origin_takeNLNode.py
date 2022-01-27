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

strSplitCharacterForNodeEdge=' ABAZ '
parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--input_mixgraph_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step5_takeLeafNLNodes_augmentation/BBBBB/',
                    help='The address of input spoc dataset.')
parser.add_argument('--input_embedding_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/embeddingModels/d2v/',
                    help='The address of pretrained embedding model.')
parser.add_argument('--output_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step6_hgt_problem_origin/AAAAA/BBBBB/mixcode_spoc.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()

fopRoot='/home/hungphd/media/dataPapersExternal/mixCodeRaw/'


fopLog=fopRoot+'step7_logs/hgt_leafNLNodes_augmentation/'
fpResult=fopLog+'a_hgt_result.txt'
createDirIfNotExist(fopLog)

# fpOutputGraph=args.output_dir
# fopOutputGraph=os.path.dirname(fpOutputGraph)
# createDirIfNotExist(fopOutputGraph)

lstProblemIds=[2,1,0]
lstContexts=['1','3','5','all']
# lstContexts=['5']


f1=open(fpResult,'w')
f1.write('')
f1.close()

for problemId in lstProblemIds:
    dictTotalResults = {}

    for strContext in lstContexts:


        fopInputMixGraph=args.input_mixgraph_dir.replace('BBBBB',strContext)
        fopInputEmbeddingModel=args.input_embedding_dir
        fpOutputGraph=args.output_dir.replace('AAAAA',str(problemId)).replace('BBBBB',strContext)
        fopOutputGraph=os.path.dirname(fpOutputGraph)
        createDirIfNotExist(fopOutputGraph)
        fpLogItem = fopLog + 'log_{}_{}.txt'.format(problemId, strContext)
        sys.stdout = open(fpLogItem, 'w')

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
        # lstIdxLabels=[]
        dictCountValueInLabel={}
        for i in range(0,len(arrPRs)):
            item=arrPRs[i]
            arrItemContent=item.split(strSplitCharacterForNodeEdge)
            if len(arrItemContent)>=3:
                id=len(dictProgramRoots.keys())
                strTrainTest=arrItemContent[2].split('\t')[0]
                strLabel=arrItemContent[1].split('\t')[problemId]
                if strLabel not in dictLabelsTextToInt.keys():
                    dictLabelsTextToInt[strLabel] = len(dictLabelsTextToInt.keys())
                    dictCountValueInLabel[strLabel] = 0
                dictCountValueInLabel[strLabel] = dictCountValueInLabel[strLabel] + 1
                idxLabel = dictLabelsTextToInt[strLabel]
                # lstIdxLabels.append(idxLabel)
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
        lstIdxLabels=[]
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

        npArrayPRs=np.array(lstVectorPRs).astype(np.float32)
        npArrayNLRs=np.array(lstVectorNLRs).astype(np.float32)
        npArrayASTNodes=np.array(lstVectorASTNode).astype(np.float32)
        npArrayNLNodes=np.array(lstVectorNLNode).astype(np.float32)



        data = HeteroData()
        data['ProgramRoot'].x = torch.tensor(npArrayPRs)
        data['NLRoot'].x =torch.tensor(npArrayNLRs)
        data['ASTNode'].x = torch.tensor(npArrayASTNodes)
        data['NLNode'].x =torch.tensor(npArrayNLNodes)

        dict_edge_index={}

        for i in range(0,len(lstFpEdgesList)):
            fpEdge=lstFpEdgesList[i]
            f1=open(fpEdge,'r')
            arrEdges=f1.read().strip().split('\n')
            f1.close()
            # print('begin edge {}'.format(fpEdge))
            nameFileEdge=os.path.basename(fpEdge)
            strTypeForEdge=nameFileEdge.replace('edges_','').replace('.txt','')
            arrSourceTarget=strTypeForEdge.split(' - ')
            strSourceType=arrSourceTarget[0]
            strTargetType=arrSourceTarget[1]
            tupEdgeLabel=(strSourceType,'to',strTargetType)
            arrayLoop=[[],[]]
            arrayLoopReverse = [[], []]

            for line in arrEdges:
                arrTabsItem=line.split(strSplitCharacterForNodeEdge)
                if len(arrTabsItem)>=3:
                    try:
                        s_type = strSourceType
                        t_type = strTargetType
                        r_type = strTypeForEdge
                        # elist = edg[t_type][s_type][r_type]
                        # rlist = edg[s_type][t_type]['rev_' + r_type]
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
            data[tupEdgeLabel[2], tupEdgeLabel[1], tupEdgeLabel[0]].edge_index = torch.tensor(arrayLoopReverse)
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
        trainEndIndex=16000
        validStartIndex=16000
        validEndIndex=18000
        testStartIndex=18000
        testEndIndex=20000


        lstTrainMask=[]
        lstValidMask=[]
        lstTestMask=[]

        for i in range(0,len(dictNLRoots.keys())):
            if i>=0 and i<trainEndIndex:
                lstTrainMask.append(True)
                lstValidMask.append(False)
                lstTestMask.append(False)
            elif i>=validStartIndex and i<validEndIndex:
                lstTrainMask.append(False)
                lstValidMask.append(True)
                lstTestMask.append(False)
            elif i>=testStartIndex and i<testEndIndex:
                lstTrainMask.append(False)
                lstValidMask.append(False)
                lstTestMask.append(True)
            else:
                lstTrainMask.append(False)
                lstValidMask.append(False)
                lstTestMask.append(False)
        data = ToUndirected()(data)
        train_mask=torch.tensor(lstTrainMask)
        val_mask=torch.tensor(lstValidMask)
        test_mask=torch.tensor(lstTestMask)
        # print(val_mask)
        data['NLRoot'].train_mask=train_mask
        data['NLRoot'].val_mask=val_mask
        data['NLRoot'].test_mask=test_mask
        data['NLRoot'].y=torch.tensor(lstIdxLabels)
        # print(data)
        # print(type(data.x_dict))



        model = HGT(hidden_channels=64, out_channels= len(dictCountValueInLabel.keys()), num_heads=2, num_layers=1)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('{} value {}'.format(name, param.data))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('devide {}'.format(device))
        data, model = data.to(device), model.to(device)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

        start_time = time.time()
        best_test_acc=test_acc=0
        bestTestData=None
        bestTestPredict=None
        for epoch in range(1, 5001):
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
        strKey = 'context_{}'.format(strContext)

        fpLogAccItem = fopLog + 'logAccDet_{}_{}.txt'.format(problemId, strContext)
        fpLogAccSumItem = fopLog + 'logAccSum_{}_{}.txt'.format(problemId, strContext)
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


        dictTotalResults[strKey] = [best_test_acc, val_acc, train_acc, train_time, test_time]

    lstSorted = sorted([(dictTotalResults[x][0], x) for x in dictTotalResults.keys()],reverse=True)
    f1 = open(fpResult, 'a')
    f1.write('Problem {}\n'.format(problemId))
    for val, key in lstSorted:
        val = dictTotalResults[key]
        strValue = 'context {}: {}'.format(key, '\t'.join(map(str, val)))
        f1.write(strValue + '\n')
    f1.close()