import argparse

from transformers import *

from pyHGT.data import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from UtilFunctions import createDirIfNotExist
from tree_sitter import Language, Parser
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../../')))
import ast

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

def lookUpJSonObjectStep2(dictJson, lstAddToGraph,dictIdsAddToWholeGraph,time,graph):
    strId=dictJson['id']
    itemNode={}
    if 'type' in dictJson.keys():

        if strId in lstAddToGraph:
            # print('go here')
            strType=dictJson['type'].strip()
            if strType not in dictIdsAddToWholeGraph.keys():
                if 'isRootNode' in dictJson.keys():
                    itemNode={'id': strType, 'type': 'program', 'attr': dictJson['statementType']}
                else:
                    itemNode = {'id': strType, 'type': 'ast', 'attr': 'ast'}
                # print('strType {}'.format(strType))
                dictIdsAddToWholeGraph[strType] = itemNode
            else:
                itemNode=dictIdsAddToWholeGraph[strType]


    if 'children' in dictJson.keys():
        lstChildren=dictJson['children']
        for child in lstChildren:
            childNode=lookUpJSonObjectStep2(child,lstAddToGraph,dictIdsAddToWholeGraph,time,graph)
            # print('child {}'.format(childNode))
            if str(childNode)!='{}':
                graph.add_edge(itemNode,childNode,time=time, relation_type='ast_edge')
    if 'nlGraph' in dictJson.keys():
        dictNL=dictJson['nlGraph']
        dictNL['label']='nlGraph'
        nlNode=addNLNodeToGraph(dictNL,lstAddToGraph,dictIdsAddToWholeGraph,time,graph)
        graph.add_edge(itemNode, nlNode,time=time, relation_type='ast_nl_edge')
    return itemNode

def addNLNodeToGraph(dictNL, lstAddToGraph, dictIdsAddToWholeGraph,time, graph):
    strLabel = dictNL['label'].strip()
    itemNode = {'id': strLabel, 'type': 'nl_nonterminal', 'attr': 'nl'}
    dictIdsAddToWholeGraph[strLabel] = itemNode

    lstChildren = dictNL['children']
    for i in range(0, len(lstChildren)):
        childNode = addNLNodeToGraph(lstChildren[i],  lstAddToGraph, dictIdsAddToWholeGraph,time, graph)
        graph.add_edge(itemNode, childNode, relation_type='nl_pos_edge',time=time)


    if 'dependencies' in dictNL.keys():
        lstDeps = dictNL['dependencies']
        for i in range(0, len(lstDeps)):
            tup = lstDeps[i]
            nodeSource =  {'id': tup[3], 'type': 'nl_terminal', 'attr': 'nl'}
            dictIdsAddToWholeGraph[nodeSource['id']]=nodeSource
            nodeTarget = {'id': tup[4], 'type': 'nl_terminal', 'attr': 'nl'}
            dictIdsAddToWholeGraph[nodeTarget['id']] = nodeTarget
            graph.add_edge(nodeSource, nodeTarget, relation_type='nl_dep_edge_{}'.format(tup[2]),time=time)
    return itemNode


def createHGTGraph(fpStep6Text,fpStep6Label,fopProgram,fopStep3,dictIdsAddToWholeGraph,time,graph):
# load graph by id
    f1=open(fpStep6Text,'r')
    arrText=f1.read().strip().split('\n')
    f1.close()

    f1 = open(fpStep6Label, 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()


    # dictIdsAddToWholeGraph={}
    for i in range(0,len(arrText)):
        arrItem=arrText[i].split('\t')

        strLabel=arrLabel[i].split('\t')[1]
        if len(arrItem)>=2:
            key=arrItem[0]
            fpItemCode=fopProgram+key+'.cpp'
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
            lookUpJSonObject(dictJson, dictFatherId, lstAppearId, indexComment, offsetComment)
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
            if i == 200:
                if 'train' in fopStep3:
                    break
            elif i==100:
                if not 'train' in fopStep3:
                    break
    return time

time=0
graph = Graph()
dictIdsAddToWholeGraph={}

time=createHGTGraph(fpStep6TextTrain,fpStep6LabelTrain,fopProgramTrain,fopStep3Train,dictIdsAddToWholeGraph,time,graph)
trainOffset=time
time=createHGTGraph(fpStep6TextTestP,fpStep6LabelTestP,fopProgramTestP,fopStep3TestP,dictIdsAddToWholeGraph,time,graph)
testPOffset=time
time=createHGTGraph(fpStep6TextTestW,fpStep6LabelTestW,fopProgramTestW,fopStep3TestW,dictIdsAddToWholeGraph,time,graph)
testWOffset=time

f1=open(fpStep6OutEmbedded,'r')
arrEmbeddeds=f1.read().strip().split('\n')
f1.close()
for i in range(0,len(arrEmbeddeds)):
    arrItem=arrEmbeddeds[i].split('\t')
    key=arrItem[0]
    val=arrItem[1]
    if key in dictIdsAddToWholeGraph.keys():
        nodeItem=dictIdsAddToWholeGraph[key]
        nodeItem['emb']=np.array(val.split(' '))

print('Calculating embeddings for non-Paper nodes...')
df = pd.DataFrame(graph.node_backward['program'])
print(df)
graph.node_feature = {'program': df}

print('train {} testP {} testW {}'.format(trainOffset,testPOffset,testWOffset))
print('Writting graph in file:')
dill.dump(graph, open(fopStep6 + '/graph_spoc.pk' , 'wb'))
print('Done.')



