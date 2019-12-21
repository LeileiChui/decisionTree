from math import log
import operator
import csv


def calcShannonEnt(dataSet):  # 计算数据的香农熵(Shannon Entropy)
    numEntries = len(dataSet)  # 数据条数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 统计有多少个类以及每个类的数量
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算单个类的熵值
        shannonEnt -= prob * log(prob, 2)  # 累加每个类的熵值
    return shannonEnt


def readData(data_path):  # 读取数据
    dataSet = []
    labels = ""
    for row in csv.reader(open(data_path,encoding="UTF-8")):
        if not labels:
            labels = row[1:-1]
        else:
            for i in range(int(row[0])):
                dataSet.append(row[1:])
    return dataSet, labels


def splitDataSet(dataSet, axis, value):  # 按某个特征分类后的数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain > bestInfoGain):  # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain = infoGain
            bestFeature = i
    if (bestFeature == -1 and numFeatures == 1):
        return 0
    return bestFeature


def majorityCnt(classList):  # 如果分类结束后还不纯，则少数服从多数
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 结果类别
    # 如果当前分类只有一种情况，则结束建树
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优属性
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 分类结果以字典形式保存
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def testHelper(decisionTree, labels, testItem):
    firstLable = list(decisionTree.keys())[0]
    firstDict = decisionTree[firstLable]
    LableIndex = labels.index(firstLable)
    result = ""
    for key in firstDict.keys():
        if testItem[LableIndex] == key:
            if (isinstance(firstDict[key], dict)):
                result = testHelper(firstDict[key], labels, testItem)
            else:
                result = firstDict[key]
    return result


def doTest(decisionTree):
    testDataPath = input("请输入测试数据集路径：")
    testDataSet = [row for row in csv.reader(open(testDataPath,encoding="UTF-8"))]
    labels = testDataSet.pop(0)
    for testItem in testDataSet:
        testItem.append(testHelper(decisionTree, labels, testItem))
    return testDataSet


if __name__ == '__main__':
    data_path = input('请输入训练数据集路径：')
    dataSet, labels = readData(data_path)  # 读取数据
    decisionTree = createTree(dataSet, labels)
    print("决策树json格式输出：")
    # 替换单引号为双引号
    print(decisionTree.__str__().replace("'", "\""))  # 输出决策树模型结果
    testResult = doTest(decisionTree)
    for item in testResult:
        if not item[-1]:
            item[-1] = "训练数据不足，无法预测"
        print(item)
