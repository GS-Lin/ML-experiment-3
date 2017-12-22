import pickle
import numpy as np
from PIL import Image
from feature import NPDFeature
import random
from sklearn import tree
from sklearn.metrics import classification_report
from ensemble import AdaBoostClassifier

def getFeature(image):
    NPD = NPDFeature(image)
    return NPD

def createImageList():
    ImageList = []
    for i in range(500):
        imageNumber = str(i)
        imageNumber = imageNumber.zfill(3)
        imageName = 'datasets\\original\\face\\' + 'face_' + imageNumber + '.jpg'
        ImageList.append(imageName)
    for i in range(500):
        imageNumber = str(i)
        imageNumber = imageNumber.zfill(3)
        imageName = 'datasets\\original\\nonface\\' + 'nonface_' + imageNumber + '.jpg'
        ImageList.append(imageName)
    return ImageList

def loadImages(imageName):
    im = Image.open(imageName).resize((24,24))
    im = im.convert('L')
    im = np.array(im)
    return getFeature(im)

def saveData(filename,data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def loadData(filename):
    with open(filename, "rb") as data:
        return pickle.load(data)

def split(Data,label,size):
    numOfData = len(Data)
    numOfVaildation = size*numOfData
    n = 0
    validationList = []
    train_x = []
    train_y = []
    validation_x = []
    validation_y = []
    while n<numOfVaildation:
        i = random.randint(0,numOfData)
        if i not in validationList:
            validationList.append(i)
            n = n+1
    for i in range(numOfData):
        if i not in validationList:
            train_x.append(Data[i])
            train_y.append(label[i])
        else:
            validation_x.append(Data[i])
            validation_y.append(label[i])
    return train_x,train_y,validation_x,validation_y

def validate_result(y,target):
    numOfCorrect = 0
    for i in range(y.shape[0]):
        if y[i]==target[i]:
            numOfCorrect = numOfCorrect+1
    return numOfCorrect/y.shape[0]

num_classifier = 5

if __name__ == "__main__":
    '''
    #数据集加标签，并划分训练集，验证集
    Data = loadData('data')
    label = np.ones(1000)
    label[500:] = -1
    train_x,train_y,validation_x,validation_y = split(Data,label,0.2)
    saveData("train",train_x)
    saveData("label",train_y)
    saveData("validation",validation_x)
    saveData("target",validation_y)
    '''
    '''
    预处理数据，得到NPD特征
    imageList = createImageList()
    Data = []
    for image in imageList:
        NPD = loadImages(image)
        Data.append(NPD.extract())
    saveData("data",Data)
    '''
    train = loadData("train")
    train_x = np.array(train)
    label = loadData("label")
    train_y = np.array(label)
    validation = loadData("validation")
    test_x = np.array(validation)
    target = loadData("target")
    test_y = np.array(target)
    weakClassifier = tree.DecisionTreeClassifier(max_depth=3)
    cls = AdaBoostClassifier(weakClassifier,num_classifier)
    cls = cls.fit(train_x,train_y)
    result_adaboost = cls.predict(test_x,0)
    print ('adaboost result: ',result_adaboost)
    print ('accuracy: ',validate_result(result_adaboost,test_y))
    target_names = {'nonface','face'}
    output = open('report.txt','w')
    output.write(classification_report(test_y,result_adaboost,target_names=target_names))


