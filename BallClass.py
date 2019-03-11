import numpy as np
import pygame
import pygame.freetype  # Import the freetype module.
from pygame.locals import *
import config as cfg 


class BallClass:

        def __init__(self, id):
            
            self.ballPosition = np.array([int(cfg.SCREENWIDTH/2-30),int(cfg.FIELDHEIGHT-70)])
            self.ballSpeed = int(4)
            self.ballAngle = float(35)
            self.ballRadius = int(8)
            self.ballRotation = float(0)
            self.id = id
            self.ballPosArray = []
            #self.getPos()
            
        
        def clearBallPosArray(self):
            self.ballPosArray = []
            return self.ballPosArray
            
        def addToBallPosArray(self,pos):
            self.ballPosArray.append(pos)
            if (len(self.ballPosArray) > 25):
                self.ballPosArray.pop(-len(self.ballPosArray))

        def getBallPosArray(self):
            return self.ballPosArray

        def getRotation(self):
            return self.ballRotation

        def setRotation(self, r):
            self.ballRotation = r

        def getSpeed(self):
            return self.ballSpeed

        def setSpeed(self, s):
            self.ballSpeed = s

        def getRadius(self):
            return self.ballRadius

        def getAngle(self):
            return self.ballAngle
        
        def setAngle(self, s):
            self.ballAngle = s

        def getPos(self):
            return self.ballPosition

        def setPos(self,newPos):
            self.ballPosition = newPos

            pass

def newBall(id):
    ball = BallClass(id)
    return ball