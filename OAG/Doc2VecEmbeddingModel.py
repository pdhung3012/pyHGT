import glob
import sys, os
import operator
import traceback

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from UtilFunctions import createDirIfNotExist
from tree_sitter import Language, Parser
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../../')))
import ast

nDim=100
offsetComment=3
distanceHeader=33
strEndLine=' ENDLINE '
strTABCHAR=' TABCHAR '
strPREFIXSTART=' PREFIXSTART '
strPREFIXEND=' PREFIXEND '
strPOSTFIXSTART=' POSTFIXSTART '
strPOSTFIXEND=' POSTFIXEND '

def lookUpJSonObject(dictJson,lstTextAST):
    if 'type' in dictJson.keys():
        strType=dictJson['type'].replace('\t',strTABCHAR).replace('\n',strEndLine).strip()
        lstTextAST.append(strType)

    if 'children' in dictJson.keys():
        lstChildren=dictJson['children']
        for child in lstChildren:
            lookUpJSonObject(child,lstTextAST)



def getEmbeddingForCodeAndWords(fpOriginLabel,fopCode,fopAST,fpOutputText,fpOutputAST,fpOutputLabel):
    try:
        dictProgram={}
        f1=open(fpOriginLabel,'r')
        arrLabels=f1.read().strip().split('\n')
        f1.close()
        for item in arrLabels:
            arrItem=item.split('\t')
            if(len(arrItem)>=4):
                dictProgram[arrItem[0]]=arrItem[2]
        lstTextD2v=[]
        lstASTD2v = []
        lstLabelD2v = []
        indexGen=0
        lenGen=len(dictProgram.keys())
        for key in dictProgram.keys():
            try:
                fpKeyCodeCpp = fopCode + key + '.cpp'
                fpAST=fopAST+key+'_ast.txt'
                f1 = open(fpKeyCodeCpp, 'r')
                arrCodeItem = f1.read().strip().split('\n')
                f1.close()
                indexComment = -1
                for i in range(0, len(arrCodeItem)):
                    item = arrCodeItem[i].strip()
                    if item.startswith('//'):
                        indexComment = i
                        break

                if indexComment >= 0:
                    lstPrefixSentence = []
                    for j in range(indexComment - offsetComment, indexComment):
                        if j <= distanceHeader:
                            continue
                        strItem = arrCodeItem[j].strip()
                        if strItem != '':
                            lstPrefixSentence.append(strItem)

                    lstPostfixSentence = []
                    for j in range(indexComment + 1, indexComment + offsetComment + 1):
                        if j <= distanceHeader or j>=len(arrCodeItem):
                            continue
                        strItem = arrCodeItem[j].strip()
                        if strItem != '':
                            lstPostfixSentence.append(strItem)

                    strComment = arrCodeItem[indexComment].strip().replace('//', '').replace('\t', strTABCHAR).strip()
                    strPrefixComment = strEndLine.join(lstPrefixSentence).replace('\t', strTABCHAR).strip()
                    strPostfixComment = strEndLine.join(lstPostfixSentence).replace('\t', strTABCHAR).strip()
                    strTextualComment = '{} {} {} {} {} {} {}'.format(strComment, strPREFIXSTART, strPrefixComment,
                                                                      strPREFIXEND, strPOSTFIXSTART, strPostfixComment,
                                                                      strPostfixComment)
                    f1=open(fpAST,'r')
                    strJson=f1.read().strip().split('\n')[1]
                    f1.close()
                    jsonOfCorrectVersion = ast.literal_eval(strJson)
                    lstItemAST=[]
                    lookUpJSonObject(jsonOfCorrectVersion, lstItemAST)
                    strItemAST=' '.join(lstItemAST)
                    lstTextD2v.append('{}\t{}'.format(key, strTextualComment))
                    lstLabelD2v.append('{}\t{}'.format(key, dictProgram[key]))
                    lstASTD2v.append('{}\t{}'.format(key, strItemAST))
                    indexGen=indexGen+1
                    print('index gen {}/{}'.format(indexGen,lenGen))
            except:
                traceback.print_exc()

        f1=open(fpOutputText,'w')
        f1.write('\n'.join(lstTextD2v))
        f1.close()
        f1 = open(fpOutputLabel, 'w')
        f1.write('\n'.join(lstLabelD2v))
        f1.close()
        f1 = open(fpOutputAST, 'w')
        f1.write('\n'.join(lstASTD2v))
        f1.close()
    except:
        traceback.print_exc()

def generateD2VEmbedding(fpTrainText,fpTestPText,fpTestWText,fpTrainAST,fpTestPAST,fpTestWAST,fpOutEmbed,fpOutModel):
    try:
        # doc2vec
        X_Train=[]
        key_Train=[]
        ast_Train = []
        X_TestP=[]
        key_TestP = []
        ast_TestP = []
        X_TestW=[]
        key_TestW = []
        ast_TestW = []
        lstAllTextAST=[]

        f1=open(fpTrainText,'r')
        arrItems=f1.read().strip().split('\n')
        f1.close()
        f1 = open(fpTrainAST, 'r')
        arrASTItems = f1.read().strip().split('\n')
        f1.close()

        for i  in range(0,len(arrItems)):
            item=arrItems[i]
            arrTabs=item.split('\t')
            key_Train.append(arrTabs[0])
            X_Train.append(arrTabs[1])
            itemAST=arrASTItems[i]
            arrASTTabs=itemAST.split('\n')
            ast_Train.append(arrASTTabs[1])



        f1 = open(fpTestPText, 'r')
        arrItems = f1.read().strip().split('\n')
        f1.close()
        f1 = open(fpTestPAST, 'r')
        arrASTItems = f1.read().strip().split('\n')
        f1.close()

        for i in range(0, len(arrItems)):
            item = arrItems[i]
            arrTabs = item.split('\t')
            key_TestP.append(arrTabs[0])
            X_TestP.append(arrTabs[1])
            itemAST = arrASTItems[i]
            arrASTTabs = itemAST.split('\n')
            ast_TestP.append(arrASTTabs[1])

            lstAllTextAST.append(arrTabs[1])
            lstAllTextAST.append(arrASTTabs[1])

        f1 = open(fpTestWText, 'r')
        arrItems = f1.read().strip().split('\n')
        f1.close()
        f1 = open(fpTestWAST, 'r')
        arrASTItems = f1.read().strip().split('\n')
        f1.close()

        for i in range(0, len(arrItems)):
            item = arrItems[i]
            arrTabs = item.split('\t')
            key_TestW.append(arrTabs[0])
            X_TestW.append(arrTabs[1])

            itemAST = arrASTItems[i]
            arrASTTabs = itemAST.split('\n')
            ast_TestW.append(arrASTTabs[1])

            lstAllTextAST.append(arrTabs[1])
            lstAllTextAST.append(arrASTTabs[1])


        tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(lstAllTextAST)]
        max_epochs = 5
        vec_size = nDim
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=0)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            # print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
            print('End epoch{}'.format(epoch))
        model.save(fpOutModel)

        d2v_all = []
        dictWords = {}
        # lstAllText=[]
        for i in range(0,len(X_Train)):
            # lstAllText.append(X_Train[i])
            x_data = word_tokenize(X_Train[i])
            v1 = model.infer_vector(x_data)
            d2v_all.append('{}\t{}'.format(key_Train[i],' '.join(map(str,v1))))

        for i in range(0, len(X_TestP)):
            # lstAllText.append(X_TestP[i])
            x_data = word_tokenize(X_TestP[i])
            v1 = model.infer_vector(x_data)
            d2v_all.append('{}\t{}'.format(key_TestP[i], ' '.join(map(str, v1))))

        for i in range(0, len(X_TestW)):
            # lstAllText.append(X_TestW[i])
            x_data = word_tokenize(X_TestW[i])
            v1 = model.infer_vector(x_data)
            d2v_all.append('{}\t{}'.format(key_TestW[i], ' '.join(map(str, v1))))

        for item in lstAllTextAST:
            arrWords=word_tokenize(item)
            for j in range(0,len(arrWords)):
                strJ = arrWords[j]
                if strJ not in dictWords.keys():
                    dictWords[strJ] = 1
                    v1 = model.infer_vector([strJ])
                    d2v_all.append('{}\t{}'.format( strJ, ' '.join(map(str, v1))))
        f1=open(fpOutEmbed,'w')
        f1.write('\n'.join(d2v_all))
        f1.close()

        # X_train_vect = pd.concat([X_train[['lengthSplit','seqTerms','pastVerbs','SWcount','AgreementNounsError','TotalTime']].reset_index(drop=True),
        #            pd.DataFrame(d2v_train)], axis=1)
        # X_test_vect = pd.concat([X_test[['lengthSplit','seqTerms','pastVerbs', 'SWcount','AgreementNounsError','TotalTime']].reset_index(drop=True),
        #            pd.DataFrame(d2v_test)], axis=1)

    except:
        traceback.print_exc()

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

getEmbeddingForCodeAndWords(fpStep5Train,fopProgramTrain,fopStep3Train,fpStep6TextTrain,fpStep6ASTTrain,fpStep6LabelTrain)
getEmbeddingForCodeAndWords(fpStep5TestP,fopProgramTestP,fopStep3TestP,fpStep6TextTestP,fpStep6ASTTestP,fpStep6LabelTestP)
getEmbeddingForCodeAndWords(fpStep5TestW,fopProgramTestW,fopStep3TestW,fpStep6TextTestW,fpStep6ASTTestW,fpStep6LabelTestW)
generateD2VEmbedding(fpStep6TextTrain,fpStep6TextTestP,fpStep6TextTestW,fpStep6ASTTrain,fpStep6ASTTestP,fpStep6ASTTestW,fpStep6OutEmbedded,fpStep6OutModel)





