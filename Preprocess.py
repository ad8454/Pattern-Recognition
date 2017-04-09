import numpy
import sys
import platform
import xml.etree.ElementTree as ET
import cv2
import os
import math

class view:
    __slots__ = 'folderName','filePath','HTMLFile','fileDiscriptor','resize','resize_other'

    def __init__(self):
        self.filePath = {}
        self.folderName=""
        self.resize = 100

    def getFolderName(self,fileName,file):
        '''
        To get the folder apth of the file
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


    def createDict(self,Class,line):
        if Class == "junk" or Class=="Junk":
            pathName = "junk"
        else:
            pathName = "iso"
        path = pathName + line.split("_")[-1] + ".inkml"
        if Class.lower() in self.filePath:
            self.filePath[Class.lower()] += [path]
        else:
            self.filePath[Class.lower()] = [path]


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
        return img

    def start(self,fileName,featureFunctions,limit=999999):
        self.openFile(fileName)
        len1=0
        for i in self.filePath.keys():
            #print(i," ",len(self.filePath[i]))
            len1+=len(self.filePath[i])
        #limit = len1/len(self.filePath.keys())
        firstPath = True
        numberToClass={}
        featureMatrix=[]
        count = 0
        #featureFile = open("feature"+fileName[-5:], 'w', newline='')
        for symb in self.filePath.keys():
            print('working on: ', symb)
            #if len(self.filePath[symb]) < limit:
            #    continue
            size = min(int(limit), len(self.filePath[symb]))
            values = self.filePath[symb][:size]
            for item in self.filePath[symb]:
                featureVector = numpy.array([])
                if firstPath:
                    self.folderName = self.getFolderName(fileName,item)
                    firstPath=False
                try:
                    tree = ET.parse(self.folderName + item)
                except:
                    self.folderName = self.getFolderName(fileName, item)
                    tree = ET.parse(self.folderName + item)

                img=self.preprocessing(tree)
                for function in featureFunctions:
                    feature=function(img)
                    featureVector=numpy.append(featureVector,feature)
                featureVector = numpy.append(featureVector,count)
                featureMatrix.append(featureVector)
                #featureFile.write(','.join(str(i) for i in featureVector))
                #featureFile.write('\n')
            numberToClass[count] = symb
            count += 1
        #featureFile.close()
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


    def zonning(self,img,numberOfbins=10):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


    def Histogram(self,img,numberOfChunks=10):
        pass

    def hog(self, image):
        Gh = image.copy()
        Gd = image.copy()

        # convert to 1 channel binary image
        image = image[:, :, 0]

        cell_size = self.resize // 10

        X = numpy.asarray([-1, 0, 1])
        Y = X.transpose()

        # Calculate the intensity gradients of the image using the above operators
        for i in range(0, len(image), cell_size):
            for j in range(0, len(image[i]), cell_size):
                img_cell = image[i:i+cell_size, j:j+cell_size]
                sumx = 0
                sumy = 0
                for a in range(3):
                    for b in range(3):
                        valx = X[a][b]
                        valy = Y[a][b]
                        x = i + 1 - a
                        y = j + 1 - b
                        if x < 0 or x >= len(image) or y < 0 or y >= len(image[i]):
                            px = image[i][j]
                        else:
                            px = image[x][y]
                        sumx += (valx * px)
                        sumy += (valy * px)

                sumx //= 9
                sumy //= 9
                Gh[i][j] = (math.sqrt((sumx ** 2) + (sumy ** 2)))

                # Get absolute angle in degrees and assign to appropriate bin.
                angle = abs(math.atan2(sumy, sumx) * 180 / 3.141592)
                if angle < 22.5 or angle > 157.5:
                    Gd[i][j] = 0
                elif angle < 67.5:
                    Gd[i][j] = 45
                elif angle < 112.5:
                    Gd[i][j] = 90
                else:
                    Gd[i][j] = 135









