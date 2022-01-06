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

class Net(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.lin1 = torch.nn.Linear(num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, num_classes)
        # print('type {} {}'.format(type(dataset.num_features),type(dataset.num_classes)))
        # print('val {} {}'.format(dataset.num_features, dataset.num_classes))
        # print('type {} {}'.format(type(data.x), type(data.edge_index)))
        # print('val {} {}'.format(data.x.shape, data.edge_index.shape))
        # print('edge index {}'.format(data.edge_index))

    def forward(self):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


# begin load SPoC data
strSplitCharacterForNodeEdge=' ABAZ '
parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
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
fopLog=fopRoot+'step7_logs/agnn/'
fpResult=fopLog+'a_agnn_result.txt'
createDirIfNotExist(fopLog)

# fpOutputGraph=args.output_dir
# fopOutputGraph=os.path.dirname(fpOutputGraph)
# createDirIfNotExist(fopOutputGraph)

lstProblemIds=[0,1,2]
lstContexts=['1','3','5','all']


f1=open(fpResult,'w')
f1.write('')
f1.close()

for problemId in lstProblemIds:
    dictTotalResults = {}

    for strContext in lstContexts:
        fopInputMixGraph = args.input_mixgraph_dir.replace('AAAAAA',strContext)
        fpLogItem=fopLog+'log_{}_{}.txt'.format(problemId,strContext)
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

        f1=open(fpNodeNLRoot,'r')
        arrNLRs=f1.read().strip().split('\n')
        f1.close()
        dictNLRoots={}
        dictLabelsTextToInt={}
        dictAllNodesAndIds={}
        lstIdxLabels=[]
        for i in range(0,len(arrNLRs)):
            item=arrNLRs[i]
            arrItemContent=item.split(strSplitCharacterForNodeEdge)
            if len(arrItemContent)>=3:
                id=len(dictNLRoots.keys())
                strTrainTest=arrItemContent[2].split('\t')[0]
                strLabel=arrItemContent[1].split('\t')[problemId]
                if strLabel not in dictLabelsTextToInt.keys():
                    dictLabelsTextToInt[strLabel]=len(dictLabelsTextToInt.keys())
                idxLabel=dictLabelsTextToInt[strLabel]
                lstIdxLabels.append(idxLabel)
                tup = [id, arrItemContent[0],idxLabel]
                dictNLRoots[arrItemContent[0]]=tup
                strInfoAboutNodeHomo='NLRoot'+strSplitCharacterForNodeEdge+arrItemContent[0]
                if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
                    dictAllNodesAndIds[strInfoAboutNodeHomo]=len(dictAllNodesAndIds.keys())

        f1=open(fpNodeProgramRoot,'r')
        arrPRs=f1.read().strip().split('\n')
        f1.close()
        dictProgramRoots={}
        dictRangeTrainTest={}
        prevTrainTest=''
        for i in range(0,len(arrPRs)):
            item=arrPRs[i]
            arrItemContent=item.split(strSplitCharacterForNodeEdge)
            if len(arrItemContent)>=3:
                id=len(dictProgramRoots.keys())
                strTrainTest=arrItemContent[2].split('\t')[0]
                strLabel=arrItemContent[1].split('\t')[problemId]
                # if strLabel not in dictLabelsTextToInt.keys():
                #     dictLabelsTextToInt[strLabel] = len(dictLabelsTextToInt.keys())
                idxLabel = dictLabelsTextToInt[strLabel]
                lstIdxLabels.append(idxLabel)
                tup = [id, arrItemContent[0],idxLabel]
                dictProgramRoots[arrItemContent[0]]=tup
                strInfoAboutNodeHomo = 'ProgramRoot' + strSplitCharacterForNodeEdge + arrItemContent[0]
                if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
                    dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())

                if strTrainTest not in dictRangeTrainTest.keys():
                    dictRangeTrainTest[strTrainTest]=[i,-1]
                if prevTrainTest!='' and prevTrainTest!=strTrainTest:
                    dictRangeTrainTest[prevTrainTest][1]=i-1
                if (i+1)==len(arrPRs):
                    dictRangeTrainTest[strTrainTest][1]=i-1
                prevTrainTest=strTrainTest



        f1=open(fpNodeASTNode,'r')
        arrASTNodes=f1.read().strip().split('\n')
        f1.close()
        dictASTNodes={}
        for i in range(0,len(arrASTNodes)):
            item=arrASTNodes[i]
            id=len(dictASTNodes.keys())
            dictASTNodes[item]=[id]
            strInfoAboutNodeHomo = 'ASTNode' + strSplitCharacterForNodeEdge + item
            if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
                dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())
            lstIdxLabels.append(0)



        f1=open(fpNodeNLNode,'r')
        arrNLNodes=f1.read().strip().split('\n')
        f1.close()
        dictNLNodes={}
        for i in range(0,len(arrNLNodes)):
            item=arrNLNodes[i]
            id=len(dictNLNodes.keys())
            dictNLNodes[item]=[id]
            strInfoAboutNodeHomo = 'NLNode' + strSplitCharacterForNodeEdge + item
            if strInfoAboutNodeHomo not in dictAllNodesAndIds.keys():
                dictAllNodesAndIds[strInfoAboutNodeHomo] = len(dictAllNodesAndIds.keys())
            lstIdxLabels.append(0)
        all_labels=torch.tensor(lstIdxLabels)
        # dictAllNodes={}
        # dictAllNodes['ProgramRoot']=dictProgramRoots
        # dictAllNodes['NLRoot']=dictNLRoots
        # dictAllNodes['ASTNode']=dictASTNodes
        # dictAllNodes['NLNode']=dictNLNodes

        fopTokenEmbed=fopInputEmbeddingModel+'token_emb/'
        fopParagraphEmbed=fopInputEmbeddingModel+'paragraph_emb/'
        lstFpTokenEmbed=sorted(glob.glob(fopTokenEmbed+'*.txt'))
        lstFpParagraphEmbed=sorted(glob.glob(fopParagraphEmbed+'*.txt'))
        dictVectorProgramRoot={}
        dictVectorNLRoot={}
        dictAllVectors={}
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
                    # dictAllVectors['ProgramRoot'+strSplitCharacterForNodeEdge+strId]=[float(item) for item in arrPRInfo[3].split()]
                    # dictAllVectors['NLRoot' + strSplitCharacterForNodeEdge + strId]=[float(item) for item in arrNLRInfo[3].split()]
                    if lengthOfVector==0:
                        lengthOfVector=len(dictVectorProgramRoot['ProgramRoot_'+strId])

        lstVectorNLRs=[]
        for i in range(0,len(dictNLRoots.keys())):
            key=list(dictNLRoots.keys())[i]
            # val=dictNLRoots[key]
            lstVectorNLRs.append(dictVectorNLRoot[key])
            dictAllVectors['NLRoot'+strSplitCharacterForNodeEdge+key]=dictVectorNLRoot[key]

        lstVectorPRs=[]
        for i in range(0,len(dictProgramRoots.keys())):
            key=list(dictProgramRoots.keys())[i]
            # val=dictProgramRoots[key]
            lstVectorPRs.append(dictVectorProgramRoot[key])
            dictAllVectors['ProgramRoot'+strSplitCharacterForNodeEdge+key] = dictVectorProgramRoot[key]

        print('size {}'.format(len(dictAllVectors.keys())))
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
                dictAllVectors['ASTNode'+strSplitCharacterForNodeEdge+strKey]=dictVectorTokens[strKey]
            else:
                lstVectorASTNode.append(np.zeros(lengthOfVector).tolist())
                dictAllVectors['ASTNode' + strSplitCharacterForNodeEdge + strKey]=np.zeros(lengthOfVector).tolist()

        print('size {}'.format(len(dictAllVectors.keys())))

        lstVectorNLNode=[]
        for i in range(0,len(dictNLNodes.keys())):
            strKey=list(dictNLNodes.keys())[i]
            if strKey in dictVectorTokens.keys():
                lstVectorNLNode.append(dictVectorTokens[strKey])
                dictAllVectors['NLNode' + strSplitCharacterForNodeEdge + strKey] = dictVectorTokens[strKey]
            else:
                lstVectorNLNode.append(np.zeros(lengthOfVector).tolist())
                dictAllVectors['NLNode' + strSplitCharacterForNodeEdge + strKey] = np.zeros(lengthOfVector).tolist()

        # npArrayPRs=np.array(lstVectorPRs).astype(np.float32)
        # npArrayNLRs=np.array(lstVectorNLRs).astype(np.float32)
        # npArrayASTNodes=np.array(lstVectorASTNode).astype(np.float32)
        # npArrayNLNodes=np.array(lstVectorNLNode).astype(np.float32)
        listAllVectors=list(dictAllVectors.values())
        npArrayAllVectors=np.array(listAllVectors).astype(np.float32)

        dict_edge_index={}
        arrayLoop=[[],[]]

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
                            s_id = dictAllNodesAndIds[s_type+strSplitCharacterForNodeEdge+strTextSource]
                        else:
                            strRootKey = arrTabsItem[2].split('\t')[1]
                            s_id = dictAllNodesAndIds[s_type+strSplitCharacterForNodeEdge+strRootKey]
                        # s_id=dictAllNodes[s_type][arrTabsItem[0]][0]
                        # print('t_type {} BBB {} AAA {}'.format(t_type,arrTabsItem[1],strTextTarget))
                        t_id = dictAllNodesAndIds[t_type+strSplitCharacterForNodeEdge+strTextTarget]
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
        all_edge_index=torch.tensor(arrayLoop)

        trainEndIndex=16000
        validStartIndex=16000
        validEndIndex=18000
        testStartIndex=18000
        testEndIndex=20000


        lstTrainMask=[]
        lstValidMask=[]
        lstTestMask=[]

        for i in range(0,len(dictAllNodesAndIds.keys())):
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
        train_mask=torch.tensor(lstTrainMask)
        val_mask=torch.tensor(lstValidMask)
        test_mask=torch.tensor(lstTestMask)


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
        num_classes=len(dictLabelsTextToInt.keys())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, data = Net(num_features=lengthOfVector,num_classes=num_classes).to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        best_test_acc = test_acc = 0
        start_time=time.time()
        for epoch in range(1, 500):
            train()
            train_acc, val_acc, test_acc = test()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
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


    lstSorted=sorted([(dictTotalResults[x][0], x) for x in dictTotalResults.keys()],reverse=True)
    f1=open(fpResult,'a')
    f1.write('Problem {}\n'.format(problemId))
    for val,key in lstSorted:
        val=dictTotalResults[key]
        strValue='context {}: {}'.format(key,'\t'.join(map(str,val)))
        f1.write(strValue+'\n')
    f1.close()








