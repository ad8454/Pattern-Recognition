import numpy
import sys
import platform
import  xml.etree.ElementTree as ET
import  cv2
import os


class view:
    __slots__ = 'folderName','filePath','HTMLFile','fileDiscriptor','resize','resize_other'

    def __init__(self):
        self.filePath = {}
        self.folderName=""
        self.resize = 100  #Size of the each symbol on the html file





    def getFolderName(self,fileName,file):
        '''
        To get the folder apth of the file
        '''

        os1 = platform.platform()

        if "Windows" in os1:
            osType = "\\"
        if "Linux" in os1:
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
        val2 = (int)((deltaX * self.resize // factor))
        val=(int)(deltaY*self.resize//factor)
        print(val, " ", val2)
        if self.resize > val:
            print(val)
            #img=cv2.copyMakeBorder(img, (int)(deltaY*self.resize//(deltaX* 2)), 0,0,0, cv2.BORDER_CONSTANT, 0)
            M = numpy.float32([[1, 0, 0], [0, 1, (self.resize-val)//2]])
            img = cv2.warpAffine(img, M, (self.resize, self.resize))

        if self.resize > val2:
            print(val)
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
            cv2.polylines(img, [pts], False, (0, 255, 255))
        return img

    def start(self,fileName,limit=10):

        with open(fileName) as fileDiscriptor:
            for line in fileDiscriptor:
                line = line.strip()
                line = line.split(",")
                Class = line[1]
                self.createDict(Class,line[0])

        firstPath = True

        print(self.filePath)

        for symb in self.filePath.keys():
            count = 0
            size = min(int(limit), len(self.filePath[symb]))
            values = self.filePath[symb][:size]
            for item in values:
                if firstPath:
                    self.folderName = self.getFolderName(fileName,item)
                    firstPath=False
                tree = ET.parse(self.folderName + item)
                strockList,strockList1 = self.getVal(tree)
                strockInfo = numpy.asarray(strockList)
                strockInfo1 = numpy.asarray(strockList1)

                widthMax = numpy.max(strockInfo1[:,0])
                widthMin =  numpy.min(strockInfo1[:,0])
                heightMax = numpy.max(strockInfo1[:,1])
                heightMin = numpy.min(strockInfo1[:, 1])

                normalized=self.normalizedImage(widthMax,widthMin,heightMax,heightMin,strockInfo)
                print(normalized)

                img=self.createImge(normalized)
                img = self.centerTheImage(img, widthMax-widthMin, heightMax-heightMin)
                print(img.shape)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                count += 1


    def bins(self,img,numberOfbins):
        size = self.resize / numberOfbins
        prevY = 0
        bins=[]
        for iter in range(size, self.resize, size):
            prevX = 0
            row=[]
            for jiter in range(size, self.resize, size):
                part = img[prevY:iter, prevX:jiter]
                row.append(numpy.sum(part))
            bins.append(row)
        return numpy.asarray(bins)


    def Histogram(self,img,numberOfChunks):
        '''
        This program creates the histogram
        :param img:
        :param numberOfChunks:
        :return:
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bining = self.bins(img,numberOfChunks)
        hist={}

        for iter in bining:
            for jiter in iter:
                if jiter in hist:
                    hist[jiter]+=1
                else:
                    hist[jiter]=1















if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(len(sys.argv))
        print(" ERROR: wrong argument \n EXPEXTED: view-class.py <name of the file>  ")
        exit(0)
    fileName = sys.argv[1]
    aView = view()
    #print(sys.argv[2])
    if len(sys.argv) ==3:
        aView.start(fileName,sys.argv[2])
    else:
        aView.start(fileName)
