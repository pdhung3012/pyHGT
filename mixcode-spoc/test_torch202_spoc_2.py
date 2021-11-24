import argparse
import os.path as osp
from tqdm import tqdm

import torch
from torch.nn import ReLU
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero
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


strSplitCharacterForNodeEdge=' ABAZ '

parser = argparse.ArgumentParser()
parser.add_argument('--use_hgt_loader', action='store_true')
parser.add_argument('--input_mixgraph_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step5_totalGraph_small/1/',
                    help='The address of input spoc dataset.')
parser.add_argument('--input_embedding_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/embeddingModels/d2v/',
                    help='The address of pretrained embedding model.')
parser.add_argument('--output_dir', type=str, default='/home/hungphd/media/dataPapersExternal/mixCodeRaw/step6_hgt_problem2/1/mixcode_spoc.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()

fopInputMixGraph=args.input_mixgraph_dir
fopInputEmbeddingModel=args.input_embedding_dir
fpOutputGraph=args.output_dir
fopOutputGraph=os.path.dirname(fpOutputGraph)
createDirIfNotExist(fopOutputGraph)

problemId=0
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
    print('begin edge {}'.format(fpEdge))
    nameFileEdge=os.path.basename(fpEdge)
    strTypeForEdge=nameFileEdge.replace('edges_','').replace('.txt','')
    arrSourceTarget=strTypeForEdge.split(' - ')
    strSourceType=arrSourceTarget[0]
    strTargetType=arrSourceTarget[1]
    tupEdgeLabel=(strSourceType,'to',strTargetType)
    arrayLoop=[[],[]]

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
            except:
                traceback.print_exc()
                quit()
    data[tupEdgeLabel[0], tupEdgeLabel[1], tupEdgeLabel[2]].edge_index = torch.tensor(arrayLoop)
    # dict_edge_index[tupEdgeLabel]=torch.tensor(arrayLoop)

    print('end edge {}'.format(fpEdge))

# lstTrainMask=range(0,len(dictNLRoots.keys()))
# lstValidMask=range(0,len(dictNLRoots.keys()))
# lstTestMask=range(0,len(dictNLRoots.keys()))

print(dictRangeTrainTest)
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
data['NLRoot'].train_mask=train_mask
data['NLRoot'].val_mask=val_mask
data['NLRoot'].test_mask=test_mask
data['NLRoot'].y=torch.tensor(lstIdxLabels)

print(data)


# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB_2')
# transform = T.ToUndirected(merge=True)
# dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)
# data = dataset[0]

train_input_nodes = ('NLRoot', data['NLRoot'].train_mask)
val_input_nodes = ('NLRoot', data['NLRoot'].val_mask)
kwargs = {'batch_size': 100, 'num_workers': 1, 'persistent_workers': True}

if not args.use_hgt_loader:
    train_loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[10] * 2,
                            input_nodes=val_input_nodes, **kwargs)
else:
    train_loader = HGTLoader(data, num_samples=[1000] * 4, shuffle=True,
                             input_nodes=train_input_nodes, **kwargs)
    val_loader = HGTLoader(data, num_samples=[1000] * 4,
                           input_nodes=val_input_nodes, **kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Sequential('x, edge_index', [
    (SAGEConv((-1, -1), 50), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (SAGEConv((-1, -1), 50), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (Linear(-1, len(dictCountValueInLabel.keys())), 'x -> x'),
])
print('metadata {}'.format(data.metadata()))
print('node_dict {} {}'.format(type(data.x_dict),data.x_dict.keys()))
model = to_hetero(model, data.metadata(), aggr='sum').to(device)


def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device)
    print('len {} {}'.format(len(batch.x_dict),len(batch.edge_index_dict)))
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['NLRoot'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['NLRoot'][:batch_size]
        loss = F.cross_entropy(out, batch['NLRoot'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


# @torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch['NLRoot'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['NLRoot'][:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['NLRoot'].y[:batch_size]).sum())

    return total_correct / total_examples


init_params()  # Initialize parameters.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 21):
    loss = train()
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
