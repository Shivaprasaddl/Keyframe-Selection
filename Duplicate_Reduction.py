import os
import cv2
import pandas as pd

class Duplicates_Removal:
    def __init__(self,pt,th=0.95):
        self.th= th
        self.Hist=[]
        self.images=pt
        self.imnmb=[]
        self.imnm=[]
        self.work()
    def Compare_images(self,A,B):
        d1 = cv2.compareHist(A[0], B[0], cv2.HISTCMP_CORREL)
        d2 = cv2.compareHist(A[1], B[1], cv2.HISTCMP_CORREL)
        d3 = cv2.compareHist(A[2], B[2], cv2.HISTCMP_CORREL)
        return (d1+d2+d3)/3
        
    def Histogram(self,I):
        return [cv2.calcHist([I],[0],None,[256],[0,256]),cv2.calcHist([I],[1],None,[256],[0,256]),cv2.calcHist([I],[2],None,[256],[0,256])]
    
    def work(self):
        for n,i in enumerate(self.images):
            self.Hist.append(self.Histogram(i))
            self.imnmb.append(n)
        self.dict=[]
        for a1,c1 in zip(self.imnmb,self.Hist):
            l=[]
            l.append(a1)
            for a2,c2 in zip(self.imnmb,self.Hist):
                if a1!=a2:
                    rs=self.Compare_images(c1,c2)
                    if rs>=self.th:
                        l.append(a2)
            self.dict.append(set(l))
        
        self.SS=pd.concat([pd.DataFrame(self.imnmb,columns=['Image']),pd.DataFrame([None for i in self.imnmb],columns=['Label'])],axis=1)
        for t,u in enumerate(self.dict):
            for y in u:
                if self.SS.iloc[y,1]==None:
                    self.SS.iloc[y,1]=t
                else:
                    pass
        self.sel=[]
        for s in list(set(self.SS['Label'])):
            l=self.SS[self.SS['Label']==s]['Image'].tolist()
            M=[cv2.Laplacian(cv2.cvtColor(self.images[int(i)], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for i in l]
            self.sel.append(self.images[l[M.index(max(M))]])
        print('Taken {} Samples reduced to {} samples.!'.format(str(len(self.images)),str(len(self.sel))))
        return self.sel