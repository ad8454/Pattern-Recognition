import numpy
import sys
import platform
import xml.etree.ElementTree as ET
import cv2
import os
import math
import scipy.ndimage

class view:
    __slots__ = 'folderName','filePath','HTMLFile','fileDiscriptor','resize','resize_other'

    def __init__(self):
        self.filePath = {}
        self.folderName=""
        self.resize = 50


    def getFolderName(self,fileName,file):
        '''
        To get the folder path of the file
        '''

        os1 = platform.platform()

        if '\\' in fileName:
            osType = "\\"
        if '/' in  fileName:
            osType = "/"

        flag=False
        # To check if the file is present in the current folder, if not then go up one folder and check again
        #until the root has reached or the file is found

        fileName1 = fileName.split(osType)
        fileName1 = fileName1[-1]
        fileName=fileName.strip(fileName1)
        path=""
        while (not flag) and len(fileName)>3:
            for root, dirs, files in os.walk(fileName):
                for name in files:
                    if name == file :
                        flag=True
                        path= os.path.abspath(root)
            fileName1 = fileName.split("/")
            fileName1 = fileName1[-1]
            fileName = fileName.strip(fileName1)
        return path+osType

    def getVal(self,tree):
        strokList=[]
        strokList1 = []
        root = tree.getroot()
        for trace in root.iter("{http://www.w3.org/2003/InkML}trace"):
            temp=[]
            points = trace.text
            points = points.strip(" ")
            points = points.split(",")
            #print(trace.text)
            for point in points:
                point = point.strip()
                #print(point, end=":")
                point = point.split(" ")
                temp.append([float(point[0]), float(point[1])])
                strokList1.append([float(point[0]), float(point[1])])
            strokList.append(temp)
            #print(strokList)
        return strokList,strokList1

    def getFileName(self, Class, line):
        if Class == "junk" or Class=="Junk":
            pathName = "junk"
        else:
            pathName = "iso"
        inkml = pathName + line.split("_")[-1] + ".inkml"
        return inkml

    def createDict(self,Class,line):
        if Class == "junk" or Class=="Junk":
            pathName = "junk"
        else:
            pathName = "iso"
        #path = pathName + line.split("_")[-1] + ".inkml"
        if Class in self.filePath:
            self.filePath[Class] += [line]
        else:
            self.filePath[Class] = [line]


    def centerTheImage(self,img,deltaX,deltaY):
        factor = max(deltaX,deltaY)
        if factor != 0:
            val2 = (int)((deltaX * self.resize // factor))
            val=(int)(deltaY*self.resize//factor)
        else:
            val2=val=0
        #print(val, " ", val2)
        if self.resize > val:
            #print(val)
            #img=cv2.copyMakeBorder(img, (int)(deltaY*self.resize//(deltaX* 2)), 0,0,0, cv2.BORDER_CONSTANT, 0)
            M = numpy.float32([[1, 0, 0], [0, 1, (self.resize-val)//2]])
            img = cv2.warpAffine(img, M, (self.resize, self.resize))

        if self.resize > val2:
            #print(val)
            #img=cv2.copyMakeBorder(img,0,0,(int)((deltaX*self.resize//(deltaY*2))),0,cv2.BORDER_CONSTANT,0)
            M = numpy.float32([[1, 0, (self.resize-val2)//2], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (self.resize, self.resize))


        return img

    def normalizedImage(self,widthMax,widthMin,heightMax,heightMin,strockInfo):
        normalized = []
        if (widthMax - widthMin) > (heightMax - heightMin):
            div = widthMax - widthMin
        else:
            div = heightMax - heightMin
        for strock in strockInfo:
            first = True
            strockList = []
            for item1 in strock:
                if widthMax - widthMin != 0:
                    item1[0] = (item1[0] - widthMin) * (self.resize / div)
                else:
                    item1[0] = (item1[0] - widthMin) * (self.resize / 0.0001)
                if heightMax - heightMin != 0:
                    item1[1] = (item1[1] - heightMin) * (self.resize / div)
                else:
                    item1[1] = (item1[1] - heightMin) * (self.resize / 0.0001)
                if first:
                    first = False

                strockList.append([item1[0], item1[1]])
            normalized.append(strockList)
        return normalized


    def createImage(self,normalized):
        img = numpy.zeros((self.resize, self.resize, 3), numpy.uint8)
        for eachStrock in normalized:
            pts = numpy.asarray(eachStrock, numpy.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (255, 255, 255))
        #cv2.imshow('window', img)
        #cv2.waitKey(0)
        return img

    def calculateTheDelta(self,strockInfo1):
        widthMax = numpy.max(strockInfo1[:, 0])
        widthMin = numpy.min(strockInfo1[:, 0])
        heightMax = numpy.max(strockInfo1[:, 1])
        heightMin = numpy.min(strockInfo1[:, 1])
        return widthMax,widthMin,heightMax,heightMin

    def openFile(self,fileName):
        with open(fileName) as fileDiscriptor:
            for line in fileDiscriptor:
                line = line.strip()
                line = line.split(",")
                Class = line[1]
                self.createDict(Class,line[0])


    def preprocessing(self,tree):
        strockList, strockList1 = self.getVal(tree)
        strockInfo = numpy.asarray(strockList)
        strockInfo1 = numpy.asarray(strockList1)
        widthMax, widthMin, heightMax, heightMin = self.calculateTheDelta(strockInfo1)
        normalized = self.normalizedImage(widthMax, widthMin, heightMax, heightMin, strockInfo)
        img = self.createImage(normalized)
        img = self.centerTheImage(img, widthMax - widthMin, heightMax - heightMin)
        # print(img.shape)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        if  heightMax - heightMin == 0:
            aspect=0
        else:
            aspect = (widthMax - widthMin)/ (heightMax - heightMin)
        return img,strockInfo,aspect

    def OnlineFeature(self,strock):
        #print(strock)
        first=strock[0][0]
        #print(first)
        last = strock[len(strock)-1][len(strock[len(strock)-1])-1]
        ans=(first[0]-last[0])**2+(first[1]-last[1])**2
        return math.sqrt(ans)




    def start(self,fileName,featureFunctionsOffline,featureFunctionsOnline, isTest=False, limit=99999):

        self.openFile(fileName)
        len1 = 0
        for i in self.filePath.keys():
            # print(i," ",len(self.filePath[i]))
            len1 += len(self.filePath[i])
        # limit = len1/len(self.filePath.keys())
        firstPath = True

        numberToClass={}
        featureMatrix=[]
        #count = 0
        #featureFile = open("feature"+fileName[-5:], 'w', newline='')
        symSet=list(self.filePath.keys())
        symSet.sort()
        opFileList = []
        for symb in symSet:
            print('working on: ',symb)

            #if len(self.filePath[symb]) < limit:
            #    continue
            size = min(int(limit), len(self.filePath[symb]))
            values = self.filePath[symb][:size]
            if isTest:
                values =self.filePath[symb]
            for item in values:
                inkml = self.getFileName(symb, item)
                featureVector = numpy.array([])
                if firstPath:
                    self.folderName = self.getFolderName(fileName, inkml)
                    firstPath = False
                try:
                    tree = ET.parse(self.folderName + inkml)
                except:
                    self.folderName = self.getFolderName(fileName, inkml)
                    tree = ET.parse(self.folderName + inkml)


                img,strockList,aspectR=self.preprocessing(tree)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.GaussianBlur(img, (3, 3), 1)
                for function in featureFunctionsOffline:
                    feature=function(img)
                    featureVector=numpy.append(featureVector,feature)
                #featureVector = numpy.append(featureVector, aspectR)
                for function in featureFunctionsOnline:
                    feature=function(strockList)
                    featureVector=numpy.append(featureVector,feature)
                featureVector = numpy.append(featureVector,symb)
                featureMatrix.append(featureVector)# = numpy.append(featureMatrix, featureVector)
                opFileList.append(item)
                #featureFile.write(','.join(str(i) for i in featureVector))
                #featureFile.write('\n')
            #numberToClass[count] = symb
            #count += 1
        #featureFile.close()

        if isTest:
            return numpy.asarray(featureMatrix), opFileList
        return numpy.asarray(featureMatrix)

    def XaxisProjection(self,img):
        projection=[]
        for iter in range(self.resize):
            projection.append(numpy.sum(img[:,iter]))
        return numpy.asarray(projection)

    def YaxisProjection(self,img):
        projection=[]
        for iter in range(0,self.resize):
            projection.append(numpy.sum(img[iter]))
        return numpy.asarray(projection)

    def DiagonalProjections(self, img):
        # create initial projection arrays initialzed to zeroes
        projectionTopLeft = [0]*(self.resize *2 - 1)
        projectionTopRight = [0] * (self.resize * 2 - 1)
        for i in range(0, self.resize):
            for j in range(0, self.resize):
                # add to TopLeft if index sums match
                projectionTopLeft[i+j] += img[i, j]
                # reverse the index of column and then add if index sums match
                idY = self.resize - j - 1
                projectionTopRight[i + idY] += img[i, j]

        # merge both diagonal features and return
        projectionTopLeft.extend(projectionTopRight)
        return numpy.asarray(projectionTopLeft)

    def zonning(self,img,numberOfbins=10):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = self.resize // numberOfbins
        prevY = 0
        zone=[]
        for iter in range(size, self.resize, size):
            prevX = 0
            #row=[]
            for jiter in range(size, self.resize, size):
                part = img[prevY:iter, prevX:jiter]
                zone.append(numpy.sum(part)/(size*size))
            #zonning.append(row)
        return numpy.asarray(zone)
