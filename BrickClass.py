import numpy as np


class BrickClass:

        def __init__(self, id, shield):
            self.id = id
            self.b_PointList = np.array([30,0],[80,0],[80,20],[30,20],[30,0])
            linex = np.max(self.b_PointList,axis=0) - np.min(self.b_PointList, axis=0)
            liney = np.max(self.b_PointList,axis=1) - np.min(self.b_PointList, axis=1)
            self.midPoint = np.array([linex/2,liney/2])
            self.shield = int(shield)

        def getPos(self):
            return self.b_PointList

        def setPos(self,newPos):
            for i in range(0,len(self.b_PointList)-1):
                self.b_PointList[i][0] = self.b_PointList[i][0] + newPos[0]
                self.b_PointList[i][1] = self.b_PointList[i][1] + newPos[1]
            self.midPoint[0] += newPos[0]
            self.midPoint[1] += newPos[1]
            return self.b_PointList,self.midPoint

        def hitShield(self):

            self.shield -=1
            return self.shield
            


def newBrick(id, shield):
    brick = BrickClass(id, shield)
    return brick