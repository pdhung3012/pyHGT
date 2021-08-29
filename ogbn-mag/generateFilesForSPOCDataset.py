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
    createDirIfNotExist(fopOutput)
    fpProgramRootRaw=fopOutput+'ProgramRoot_raw.txt'
    fpProgramElementRaw = fopOutput + 'ProgramElement_raw.txt'
    fpNLRootRaw = fopOutput + 'NLRoot_raw.txt'
    fpNLElementRaw = fopOutput + 'NLElement_raw.txt'
    fpEdgeListRaw = fopOutput + 'edgeLists_raw.txt'
    fpLabelRaw = fopOutput + 'labels_raw.txt'
    fpLogRaw = fopOutput + 'log_raw.txt'

    indexProgram = 0
    dictEntities={}
    dictEntities['ProgramRoot']={}
    dictEntities['ProgramElement']={}
    dictEntities['NLRoot'] = {}
    dictEntities['NLElement'] = {}
    lstTotalLabel=[]

    f1=open(fpEdgeListRaw,'w')
    f1.write('')
    f1.close()

    # traverse train data
    indexBatch=0
    f1=open(lstFpStep6Text[indexBatch],'r')
    arrText=f1.read().strip().split('\n')
    f1.close()

    f1 = open(lstFpStep6Label[indexBatch], 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()

    dictIdsAddToWholeGraph={}
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
            lstEdges=[]
            indexProgram=indexProgram+1
            lookUpJSonObjectStep2(dictJson, lstAddToGraph,dictIdsAddToWholeGraph, time,graph,indexProgram,dictEntities,lstEdges)
            lstTotalLabel.append(arrLabel[i])
            if len(lstEdges)>0:
                lstStrEdge=[]
                for edge in lstEdges:
                    if len(edge)>=5:
                        strEdge='{}\t{}\t{}\t{}\t{}'.format(edge[0],edge[1],edge[2],edge[3],edge[4])
                        lstStrEdge.append(strEdge)
                f1=open(fpEdgeListRaw,'a')
                f1.write('\n'.format(lstStrEdge)+'\n')
                f1.close()
            if i==100:
                break
            print('end train {}/{}'.format(i,len(arrText)))
            # if i == 200:
            #     if 'train' in fopStep3:
            #         break
            # elif i==100:
            #     if not 'train' in fopStep3:
            #         break
    indexBeginTestP=indexProgram

    # traverse testP data
    indexBatch=1
    f1=open(lstFpStep6Text[indexBatch],'r')
    arrText=f1.read().strip().split('\n')
    f1.close()
    f1 = open(lstFpStep6Label[indexBatch], 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()
    dictIdsAddToWholeGraph={}
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
            lstEdges=[]
            # dictIdsAddToWholeGraph={}
            indexProgram=indexProgram+1
            lookUpJSonObjectStep2(dictJson, lstAddToGraph,dictIdsAddToWholeGraph, time,graph,indexProgram,dictEntities,lstEdges)
            lstTotalLabel.append(arrLabel[i])
            if len(lstEdges)>0:
                lstStrEdge=[]
                for edge in lstEdges:
                    if len(edge)>=5:
                        strEdge='{}\t{}\t{}\t{}\t{}'.format(edge[0],edge[1],edge[2],edge[3],edge[4])
                        lstStrEdge.append(strEdge)
                f1=open(fpEdgeListRaw,'a')
                f1.write('\n'.format(lstStrEdge)+'\n')
                f1.close()
            if i==100:
                break
            print('end testP {}/{}'.format(i,len(arrText)))

    indexBeginTestW = indexProgram
    # traverse testW data
    indexBatch=2
    f1=open(lstFpStep6Text[indexBatch],'r')
    arrText=f1.read().strip().split('\n')
    f1.close()
    f1 = open(lstFpStep6Label[indexBatch], 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()
    dictIdsAddToWholeGraph={}
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
            lstEdges=[]
            indexProgram=indexProgram+1
            lookUpJSonObjectStep2(dictJson, lstAddToGraph,dictIdsAddToWholeGraph, time,graph,indexProgram,dictEntities,lstEdges)
            lstTotalLabel.append(arrLabel[i])
            if len(lstEdges)>0:
                lstStrEdge=[]
                for edge in lstEdges:
                    if len(edge)>=5:
                        strEdge='{}\t{}\t{}\t{}\t{}'.format(edge[0],edge[1],edge[2],edge[3],edge[4])
                        lstStrEdge.append(strEdge)
                f1=open(fpEdgeListRaw,'a')
                f1.write('\n'.format(lstStrEdge)+'\n')
                f1.close()
            if i==100:
                break
            print('end testW {}/{}'.format(i,len(arrText)))

    # write obj to file
    f1=open(fpProgramRootRaw,'w')
    f1.write('\n'.format(dictEntities['ProgramRoot'].keys()))
    f1.close()
    f1=open(fpProgramElementRaw,'w')
    f1.write('\n'.format(dictEntities['ProgramElement'].keys()))
    f1.close()
    f1=open(fpNLRootRaw,'w')
    f1.write('\n'.format(dictEntities['NLRoot'].keys()))
    f1.close()
    f1=open(fpNLElementRaw,'w')
    f1.write('\n'.format(dictEntities['NLElement'].keys()))
    f1.close()
    f1 = open(fpLabelRaw, 'w')
    f1.write('\n'.format(lstTotalLabel))
    f1.close()
    f1 = open(fpLogRaw, 'w')
    f1.write('IndexBeginTestP\t{}\nIndexBeginTestW\t{}'.format(indexBeginTestW,indexBeginTestW))
    f1.close()

    return time

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

fopOutput='dataset_spoc/'
offsetComment=3
dictIdsAddToWholeGraph={}

lstFpStep6Text=[fpStep6TextTrain,fpStep6TextTestP,fpStep6TextTestW]
lstFpStep6Label=[fpStep6LabelTrain,fpStep6LabelTestP,fpStep6LabelTestW]
lstFpStep6AST=[fpStep6ASTTrain,fpStep6ASTTestP,fpStep6ASTTestW]
lstFopStep3=[fopStep3Train,fopStep3TestP,fopStep3TestW]
time=0
graph = Graph()

traverseHGTGraph(lstFpStep6Text,lstFpStep6Label,lstFpStep6AST,lstFopStep3,dictIdsAddToWholeGraph,time,graph,fopOutput)




print('finish')