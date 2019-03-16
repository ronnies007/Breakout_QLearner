import random, sys, os
from pygame.locals import *
import config as cfg
import pygame
import pygame.freetype  # Import the freetype module.
import math
from collision import *
from collision import Vector as v
import BallClass as ball
import BrickClass as brick
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime, timedelta, time
import time

FPS = 800
WINDOWWIDTH = 360 # 500
WINDOWHEIGHT = 420 # 800
FIELDHEIGHT = 420 # 650

#             R    G    B
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
BRIGREEN  = (150, 255, 150)
DARKGREEN = (  20, 100,   20)
DARKGRAY  = ( 20,  10,  10)
BLUEISH = ( 0, 0, 255)
BGCOLOR = BLACK
HEADCOLOR = DARKGREEN # GREEN # BRIGREEN



LEFT = 'left'
MIDDLE = 'middle'
RIGHT = 'right'
BUTTON = 'button'
actionsArray = [LEFT, MIDDLE, RIGHT, BUTTON]

pygame.init()
FPSCLOCK = pygame.time.Clock()
DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
run = cfg.lastRun
aliveRunTime = time.time()
counter = int(0)


class gameState:

        def __init__(self):
                global FPSCLOCK, DISPLAYSURF, BASICFONT, run, actionsArray, WINDOWWIDTH, WINDOWHEIGHT, FIELDHEIGHT
                self.gameRun = 0
                self.SCREENWIDTH = WINDOWWIDTH
                self.FIELDWIDTH = self.SCREENWIDTH
                self.SCREENHEIGHT = WINDOWHEIGHT
                self.FIELDHEIGHT = FIELDHEIGHT
                self.screen = DISPLAYSURF  # pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
                pygame.mouse.set_pos(self.SCREENWIDTH/2,self.SCREENHEIGHT-10)
                pygame.mouse.set_visible(False)
                pygame.display.set_caption('***Breakout***')
                self.clock = pygame.time.Clock()
                self.posX = int(self.FIELDWIDTH/2+30)
                self.posY = int(self.FIELDHEIGHT-40-10)
                self.ball = ball.newBall(0) 
                self.ball.setPos([self.posX,self.posY])
                self.velX = int(0)
                self.velY = int(0)
                self.ballRadius = int(10)
                self.ballMoveVector = pygame.math.Vector2([self.velX,self.velY])
                self.done = False
                self.paddlePointList = [[0,5],[5,0],[55,0],[60,5],[60,15],[0,15],[0,5]]
                #self.brickPointList = [[30,5],[35,0],[75,0],[80,5],[80,15],[75,20],[35,20],[30,15],[30,5]]
                self.brickPointList = [[0,0],[30,0],[30,12],[0,12],[0,0]]
                self.borderPointList = [[1,self.FIELDHEIGHT-1],[1,1],[self.SCREENWIDTH-1,1],[self.SCREENWIDTH-1,self.FIELDHEIGHT-1]]
                #self.poly = [[200,200],[100,335],[200, 400],[400,250],[400,200], [200,200]]
                self.collCounter = 0
                self.xyPlayerList =[]
                self.paddleColor = (150,150,150)
                self.deleteThis = int(-1)
                self.hitPointList = []
                i = int(0)
                for x,y in self.paddlePointList:
                        self.xyPlayerList.append([x+int(self.FIELDWIDTH/2), y+self.FIELDHEIGHT-40])
                self.paddlePointList = self.xyPlayerList
                self.collLineList = []
                self.brickMuster = []
                self.xybrickList = []
                self.leng = 0
                self.cornerPixelFactor = 8 # int(2*self.ball.getRadius()/3)
                self.ballSpeed = math.sqrt( (self.velX*self.velX) + (self.velY*self.velY) )
                self.gameCount = int(0)
                self.score = int(0)
                self.polyCollissionList = []
                self.mouseVelX , self.mouseVelY = 0,0
                self.rewardSave = float(-0.1)
                self.reward = float(-0.1)
                self.framesAlive = int(0)
                self.scA = []
                self.gridx = []
                self.gridy = []
                self.direction = random.choice(actionsArray)
                self.pre_direction = self.direction
                self.totalscore = 0
                self.totalreward = []
                self.secondsAlive = []
                self.time_stop = datetime(2019,3,6,0,0,0)   
                self.polyCollissionList = [self.paddlePointList,self.borderPointList]
                self.ballAlive_frames = 0


        def frameStep(self, action):
            global run
            self.framesAlive += 1
            self.ballAlive_frames += 1
            #sf = pygame.Surface.convert(self.screen) 
            #self.scA.append(sf)
            image_data, reward, done = self.runGame(action)
            cfg.reward = self.reward
            return image_data, reward, done


        def runGame(self, action):
            global run
            pygame.event.pump()
            #--->         # action[0] LEFT    #action[1] MIDDLE    #action[2] RIGHT    #action[3] BUTTON
            self.pre_direction = self.direction    
            if (action[0] == 1):
                self.direction = LEFT
            elif (action[1] == 1):
                self.direction = MIDDLE
            elif (action[2] == 1):
                self.direction = RIGHT
            elif (action[3] == 1):
                self.direction = BUTTON
           
            self.reward = -0.1 # -= 0.0000532 * (len(self.polyCollissionList)-2)
            cfg.reward = self.reward
            done = False
            # print ("polycollList len:",len(self.polyCollissionList))
            if (len(self.polyCollissionList) < 3) and (self.ball.getPos()[1] > self.FIELDHEIGHT-100):
                self.initBricks()
    
            self.screen.fill((0,0,0))
            sys.stdout.flush()
            events = pygame.event.get()
            for event in events:
                    if event.type == pygame.MOUSEMOTION:
                            # move paddle
                            mouseX, mouseY = pygame.mouse.get_pos()          # 	— 	get the mouse cursor position
                            oldMouseVelX = self.mouseVelX
                            self.mouseVelX , self.mouseVelY = 0,0 #pygame.mouse.get_rel()   # 	— 	get the amount of mouse movement
                            if (oldMouseVelX == self.mouseVelX):
                                    self.mouseVelX = 0
                            if (len(self.polyCollissionList) > 0):
                                    if (self.polyCollissionList[0][0][0]+90 <= self.SCREENWIDTH-10) and (self.mouseVelX > 0):
                                            movedPaddle = []
                                            for p in self.polyCollissionList[0]:
                                                    x = p[0]
                                                    y = p[1]
                                                    x += self.mouseVelX
                                                    movedPaddle.append([x,y])
                                            self.polyCollissionList[0] = movedPaddle
                                    if  (self.polyCollissionList[0][0][0] >= 10) and (self.mouseVelX <  0):
                                            movedPaddle = []
                                            for p in self.polyCollissionList[0]:
                                                    x = p[0]
                                                    y = p[1]
                                                    x += self.mouseVelX
                                                    movedPaddle.append([x,y])
                                            self.polyCollissionList[0] = movedPaddle

                    if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:      
                                            sys.exit()               
                            if event.key == pygame.K_SPACE:
                                    done2=False
                                    while not done2:
                                            events = pygame.event.get()
                                            for event in events:
                                                    if event.type == pygame.KEYDOWN:
                                                            if event.key == pygame.K_SPACE:        
                                                                    done2=True

            
            # -------------------------------   BRICK getroffen ?? --> hier wird er geloescht !!  -------------------------------
            if not (self.deleteThis == -1):
                del self.polyCollissionList[self.deleteThis]        
                self.score += 1
                self.ballAlive_frames = 0
                self.totalscore += 1
                # print ("len(polyCollissionList):",len(self.polyCollissionList))
                self.deleteThis = -1
                self.reward = 3
                cfg.reward = self.reward
                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                cfg.aliveGameTime = time.time()
                return image_data, self.reward, done


            oldscore = self.score
            self.bounceBall()
            if not (self.ballMoveVector == 0,0):
                self.ballMoveVector = self.ballMoveVector.normalize()
                self.ballMoveVector = self.ballMoveVector*self.ball.getSpeed()

            self.velX, self.velY = self.ballMoveVector
            self.posX += self.velX
            self.posY += self.velY
            self.posX = math.floor(self.posX+0.5)
            self.posY = math.floor(self.posY+0.5)
            self.ballSpeed, self.ballAngle = self.ballMoveVector.as_polar()
            self.ball.setAngle(self.ballAngle)
            self.posX = int(self.posX)
            self.posY = int(self.posY)

            if (self.posY >= self.FIELDHEIGHT ):   # ---  ball CRASHED  ---
                self.reward = -10 # -54 / (len(self.polyCollissionList)-2)
                cfg.reward = self.reward
                done = True
                # print("WAS HERE CRASHED !!!")
                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                #time.sleep(1)
                #self.newBall()
                return image_data, self.reward, done
        
            else: 
                if (len(self.polyCollissionList) > 0):
                    if self.direction == LEFT and (self.polyCollissionList[0][0][0] >= 10):
                        movedPaddle = []
                        for p in self.polyCollissionList[0]:
                                x = p[0]
                                y = p[1]
                                x -= 3
                                movedPaddle.append([x,y])
                        self.polyCollissionList[0] = movedPaddle
                        #print("cfg.ballReleased:",cfg.ballReleased)
                        if (cfg.ballReleased == False): 
                            self.posX -= 3
                         
                    elif self.direction == MIDDLE:
                            pass

                    elif self.direction == RIGHT and (self.polyCollissionList[0][0][0]+90 <= self.SCREENWIDTH-10):
                        movedPaddle = []
                        for p in self.polyCollissionList[0]:
                                x = p[0]
                                y = p[1]
                                x += 3
                                movedPaddle.append([x,y])
                        self.polyCollissionList[0] = movedPaddle
                        if (cfg.ballReleased == False): 
                            self.posX += 3
                           
                    elif self.direction == BUTTON:  
                        #print ("BUTTON pressed !!!") 
                        if (cfg.ballReleased == False): 
                            cfg.ballReleased = True  
                            if (self.pre_direction == RIGHT):   self.velX = 1
                            elif (self.pre_direction == LEFT):   self.velX = -1
                            elif (self.pre_direction == MIDDLE):  
                                while (self.velX == 0):
                                        self.velX = random.randint(-1,1)
                            else:       
                                while (self.velX == 0):
                                        self.velX = random.randint(-1,1)

                            self.velY = -1
                            self.ballMoveVector = pygame.math.Vector2([self.velX,self.velY])    
                            self.velX, self.velY = self.ballMoveVector
                            if not ([int(self.velX),int(self.velY)] == 0,0):
                                self.ballMoveVector = self.ballMoveVector.normalize()
                            self.ballMoveVector = self.ballMoveVector*self.ball.getSpeed()         
                            #print ("BUTTON pressed !!!",self.ballMoveVector,self.ball.getSpeed())
                            #self.ballSpeed, self.ballAngle = self.ballMoveVector.as_polar()
                            #self.ball.setAngle(self.ballAngle)
                            #self.ball.setSpeed(self.ballSpeed)
                            #self.ball.setPos([self.posX,self.posY])# = [self.posX,self.posY]

                self.ball.setPos([self.posX,self.posY])# = [self.posX,self.posY]
                pygame.draw.circle(self.screen, (220, 220, 220), self.ball.getPos(), self.ball.getRadius(), 0)
                pygame.draw.polygon(self.screen,  (150, 150, 150), self.borderPointList, 5)       
                
                # -------------  draw paddle  -------------
                if (len(self.polyCollissionList) > 0):    
                    pygame.draw.polygon(self.screen, self.paddleColor, self.polyCollissionList[0], 0)
                # -------------  draw the other collision objects  -------------
                for o in range(0,len(self.polyCollissionList)):
                    if not (o == 0) and not (o == 1):
                        pygame.draw.polygon(self.screen, self.paddleColor, self.polyCollissionList[o], 0)
                # -------------  draw helping lines of collisions  -------------
                # for j in range(0,len(self.collLineList)):
                #        pygame.draw.polygon(self.screen, (225, 100, 100), self.collLineList[j], 2)        
                
                #  -------------  Leerlauf-Watchdog  -------------
                #print (self.ballAlive_frames)
                cfg.ballAlive_frames = self.ballAlive_frames
                if (self.ballAlive_frames > 3600) and (cfg.ballReleased == True):
                        fl = open("r://WATCHDOG-alert.txt","a")
                        time.sleep(.5)
                        fl.write("\r\n") 
                        fl.write("alert at: ")
                        fl.write(str(time.strftime("%d.%m.%Y %H:%M:%S")))
                        fl.write("   "+str(self.posX)+", "+str(self.posY))
                        fl.close()  
                        time.sleep(.5)        
                        print("WATCHDOG alert!! ballPos:", self.posX, self.posY)
                        self.initBricks()
                        self.newBall()
                        
                        # minute = now / 60
                        # seconds = now % 60        
                        #now = time.ctime(int(time.time()))
                #  -----------------------------------------------
                        
                if (self.leng >= self.ball.getRadius()):
                        self.leng = 0

                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                FPSCLOCK.tick(FPS) 
                pygame.display.update()
                self.scA = []
                if (done==True): 
                        cfg.aliveGameTime = time.time() - cfg.aliveGameTime
                        #cfg.framesPerLife = 0

                        pass

                #self.totalreward.append(self.reward)
                return image_data, self.reward, done


        def initBricks(self):
            #pygame.mouse.set_pos(self.SCREENWIDTH/2,580)
            if (len(self.polyCollissionList) > 0):
                    self.paddlePointList = self.polyCollissionList[0]
            self.collLineList = []
            self.brickMuster = []
            self.xybrickList = []
            self.leng = 0
            self.collCounter = 0
            self.xyPlayerList =[]
            self.paddleColor = (150,150,150)
            self.deleteThis = int(-1)
            self.hitPointList = []
            i = int(0)
            self.polyCollissionList = []
            vPlus = 0
            hPlus = 25
            columns = 9
            rows = 6
            count = int(0)
            for z in range(0,rows):
                    hPlus = 25
                    for n in range(0,columns):
                            self.xybrickList = []
                            #self.brick = brick.newBrick(count, 1)
                            for x,y in self.brickPointList:
                                    posCroud = [self.FIELDWIDTH/2-columns*20, self.FIELDHEIGHT/3-rows*15]
                                    self.xybrickList.append([x+hPlus+posCroud[0], y+vPlus+posCroud[1]])
                                    
                            hPlus += 35
                            count += 1
                            self.brickMuster.append(self.xybrickList)
                    vPlus += 20
            
            self.polyCollissionList = [self.paddlePointList,self.borderPointList]
            self.polyCollissionList.extend(self.brickMuster)
            self.collLineList = []
            cfg.rundenZeit = time.time()


        def bounceBall(self):
            factX = random.uniform(-1, 1)
            if (factX > 0): factY = 1 - factX
            if (factX < 0): factY = 1 + factX
            disc = -1
            polIndex = 0
            hitIndex = 0
            distanceList = np.array([])
            self.hitPointIndexList = []
            done = False
            for polygon in self.polyCollissionList:
                    for i in range(0,len(polygon)-1):
                            f = 1.2 #self.ballMoveVector.length()/self.ball.getRadius()
                            lineIndex = i
                            point1 = polygon[i]
                            point2 = polygon[i+1]
                            lineX = point2[0] - point1[0]
                            lineY = point2[1] - point1[1]
                            line = [lineX,lineY]
                            delta = 0.05
                            while (delta < f):
                                    Q = pygame.math.Vector2([self.posX+self.velX*delta,self.posY+self.velY*delta])               # Centre of circle
                                    r = self.ball.getRadius()                  # Radius of circle
                                    P1 = pygame.math.Vector2(point1)     # Start of line segment
                                    V = pygame.math.Vector2(line)  # Vector along line segment
                                    a = V.dot(V)
                                    b = 2 * V.dot(P1 - Q)
                                    c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r*r
                                    disc = (b*b) - (4 * a * c)
                                    if not (disc < 0):
                                            delta = f
                                    delta += 0.04

                            if not (disc < 0) and not (a==0):  
                                sqrt_disc = math.sqrt(disc)
                                t1 = (-b + sqrt_disc) / (2 * a)
                                t2 = (-b - sqrt_disc) / (2 * a)
                                t = max(0, min(1, - b / (2 * a)))
                                if (0 <= t1 < 1) or (0 <= t2 < 1):
                                    hitPoint1 = P1 + V*t1
                                    hitPoint2 = P1 + V*t2
                                    #self.paddleColor = (255,100,100)
                                    hitIndex = polIndex
                                    if not (hitPoint1==P1) and not (hitPoint2==P1):
                                            if (self.collCounter >= 0):
                                                    self.hitPointList.append([hitPoint1,hitPoint2])
                                                    self.hitPointIndexList.append(hitIndex)
                                            else:
                                                    #  print ("collision-Delete blocked")
                                                    pass
                            else:
                                    self.paddleColor = (150,150,150)
                    # -----------------------------------------------------   search for closest hitpoint   ------------------------------
                    countPairs = 0
                    if (1 >= len(self.hitPointList) > 0):     # nur ein Paar HitPoints
                            self.ball.addToBallPosArray(self.hitPointList[0][0])
                            self.ball.addToBallPosArray(self.hitPointList[0][1])
                            self.reflectBall(polygon,hitIndex,self.hitPointList[0][0],self.hitPointList[0][1],0)
                            self.collLineList.append([self.hitPointList[0][0],self.hitPointList[0][1]])
                            self.hitPointList = []
                            self.hitPointIndexList = []
                    elif (len(self.hitPointList) > 1):              #  mehr als ein Paar Hitpoints --- SEARCH BEGINN ---
                            #   print ("polIndex:",polIndex)
                            #   print ("len(self.hitPointList):", len(self.hitPointList))
                            #   print ("self.hitPointList:", self.hitPointList)
                            oldBallVector = pygame.math.Vector2(self.ballMoveVector)
                            oldBallPos = pygame.math.Vector2(self.ball.getPos())
                            oldBallPos = oldBallPos - oldBallVector*15

                            for p in range(0,len(self.hitPointList)):
                                    lineMid = self.hitPointList[p][0].lerp(self.hitPointList[p][1], 0.5)
                                    line = self.hitPointList[p][0] - self.hitPointList[p][1]
                                    if (line.length()>0):
                                            distance = lineMid.distance_to(oldBallPos) # /line.length()
                                    else:
                                            distance = lineMid.distance_to(oldBallPos)
                                #       print ("mid:",lineMid)
                                #      print ("dist:",distance)
                                    #print (distance1,distance2)
                                    distanceList = np.append(distanceList,(distance))
                            #   print ("distanceList",distanceList)
                            if (len(distanceList)>0):
                                    shortest = np.min(distanceList, axis=None)
                                    shortestIndex = np.argmin(distanceList, axis=None)
                                    #shortestIndex = self.hitPointIndexList[shortestIndex]
                                #     print ("shortest:", shortest)
                                #    print ("shortestIndex:", shortestIndex)
                                    #shortestIndex = self.hpIndexList[shortestIndex]
                                    self.leng = self.hitPointList[shortestIndex][0].distance_to(self.hitPointList[shortestIndex][1])
                                    if not (self.leng >= self.ball.getRadius()):
                                        self.reflectBall(polygon,polIndex,self.hitPointList[shortestIndex][0],self.hitPointList[shortestIndex][1], shortestIndex)
                                        randAngle = random.randint(-30,30)
                                        if (self.ballMoveVector[0] < 0) and (self.ballMoveVector[1] == 0):
                                                self.ballMoveVector.rotate(randAngle)
                                        if (self.ballMoveVector[0] > 0) and (self.ballMoveVector[1] == 0):
                                                self.ballMoveVector.rotate(randAngle)
                                        pass
                                    else:
                                            endex = 0
                                            for p in polygon:
                                                    testPoint = pygame.math.Vector2(p)
                                                    hp1 = pygame.math.Vector2(self.hitPointList[shortestIndex][0])
                                                    hp2 = pygame.math.Vector2(self.hitPointList[shortestIndex][1])
                                                    l1 = testPoint.distance_to(hp1)
                                                    l2 = testPoint.distance_to(hp2)
                                                    # hp3=pygame.math.Vector2([0,0])
                                                    #  hp4=pygame.math.Vector2([0,0])
                                                    if (l1<=self.cornerPixelFactor):
                                                    #          print ("-------------------- hp1 is eckpoint!",hp1, "index:",endex)
                                                            if (endex==len(polygon)-1):
                                                                    leftVec = testPoint - polygon[1]
                                                                    rightVec = testPoint - polygon[len(polygon)-2]
                                                            else:
                                                                    if (endex==0):
                                                                            leftVec = testPoint - polygon[1]
                                                                            rightVec = testPoint - polygon[len(polygon)-2]
                                                                    else:
                                                                            leftVec = testPoint - polygon[endex+1]
                                                                            rightVec = testPoint - polygon[endex-1]
                                                            leftVec = leftVec.normalize()
                                                            rightVec = rightVec.normalize()
                                                            midVec = leftVec + rightVec
                                                            midVec = midVec.normalize()
                                                #              print ("midVec",midVec)
                                                            randAngle = random.randint(-6,6)
                                                            #print ("randAngle",randAngle)
                                                            #self.ball.setRotation(randAngle)
                                                            midVec = midVec.rotate(90)
                                                            hp3 = testPoint + midVec*3
                                                            hp4 = testPoint - midVec*3
                                                #              print ("leftVec",leftVec)
                                                #              print ("rightVec",rightVec)
                                                    #         print ("midVec(r)",midVec)
                                                    #          print ("hp3",hp3)
                                                    #         print ("hp4",hp4)
                                                            self.collLineList.append([hp3,hp4])
                                                            #self.reflectBall( polygon, polIndex, hp3, hp4, shortestIndex )
                                                            midVec = midVec.rotate(-90+randAngle)
                                                            self.ballMoveVector = midVec*self.ball.getSpeed()
                                                            randAngle = random.randint(-30,30)
                                                            if (self.ballMoveVector[1] == 0):
                                                                    self.ballMoveVector.rotate(randAngle)
                                                            if (polIndex>1):
                                                                self.deleteThis = polIndex
                                                    #            print("deleted polyIndex:",self.deleteThis,"lineIndex.:", str(shortestIndex))
                                                            else:
                                                                    self.deleteThis = -1
                                                            #break
                                                    if (l2<=self.cornerPixelFactor):
                                                    #          print ("-------------------- hp2 is eckpoint!",hp2, "index:",endex)
                                                            if (endex==len(polygon)-1):
                                                                    leftVec = testPoint - polygon[1]
                                                                    rightVec = testPoint - polygon[len(polygon)-2]
                                                            else:
                                                                    if (endex==0):
                                                                            leftVec = testPoint - polygon[1]
                                                                            rightVec = testPoint - polygon[len(polygon)-2]
                                                                    else:
                                                                            leftVec =  testPoint - polygon[endex+1]
                                                                            rightVec = testPoint - polygon[endex-1]
                                                            leftVec = leftVec.normalize()
                                                            rightVec = rightVec.normalize()
                                                            midVec = leftVec + rightVec
                                                            midVec = midVec.normalize()
                                                    #          print ("midVec",midVec)
                                                            randAngle = random.randint(-6,6)
                                                            #print ("randAngle",randAngle)
                                                            #self.ball.setRotation(randAngle)
                                                            midVec = midVec.rotate(90)
                                                            hp3 = testPoint - midVec*3
                                                            hp4 = testPoint + midVec*3
                                                    #           print ("leftVec",leftVec)
                                                    #          print ("rightVec",rightVec)
                                                    #         print ("midVec(r)",midVec)
                                                    #         print ("hp3",hp3)
                                                    #         print ("hp4",hp4)
                                                            self.collLineList.append([hp3,hp4])
                                                            #self.reflectBall( polygon, polIndex, hp3, hp4, shortestIndex )
                                                            midVec = midVec.rotate(-90+randAngle)
                                                            self.ballMoveVector = midVec*self.ball.getSpeed()
                                                            randAngle = random.randint(-30,30)
                                                            if (self.ballMoveVector[1] == 0):
                                                                    self.ballMoveVector.rotate(randAngle)
                                                            if (polIndex>1):
                                                                self.deleteThis = polIndex
                                                        #         print("deleted polyIndex:",self.deleteThis,"lineIndex.:", str(shortestIndex))
                                                            else:
                                                                    self.deleteThis = -1
                                                            #break
                                                    endex +=1

                                    self.hitPointList = []
                                    self.hitPointIndexList = []
                                    distanceList = np.array([])
                                    shortestIndex = 0
                                    #break
                                    done = True
                    if (done == True): return True
                    polIndex +=1

        def newBall(self):
            #print("WAS HERE NEWBALL,self")
            self.ballAlive_frames = 0
            cfg.ballReleased = False
            cfg.aliveGameTime = time.time()
            self.gameRun += 1
            self.paddlePointList = [[0,5],[5,0],[55,0],[60,5],[60,15],[0,15],[0,5]]
            self.xyPlayerList =[]
            randomPaddleX = random.randint(5,self.FIELDWIDTH-65)
            for x,y in self.paddlePointList:
                    self.xyPlayerList.append([x+int(randomPaddleX), y+self.FIELDHEIGHT-40])
            self.paddlePointList = self.xyPlayerList

            self.posX = int(self.paddlePointList[1][0]+21)
            self.posY = int(self.paddlePointList[1][1]-self.ball.getRadius()-2)
            self.ball = ball.newBall(0) 
            self.ball.setPos([self.posX,self.posY])
            self.velX = int(0)
            self.velY = int(0)
            self.ballRadius = int(10)
            self.ballMoveVector = pygame.math.Vector2([self.velX,self.velY])
            self.done = False
            
            #self.ballSpeed = math.sqrt( (self.velX*self.velX) + (self.velY*self.velY) )
            self.score = int(0)
            #self.totalscore = int(0)
            self.rewardSave = float(-0.1)
            self.reward = float(-0.1)
            cfg.reward = self.reward
            self.framesAlive = int(0)
            self.scA = []
            self.gridx = []
            self.gridy = []
            #self.polyCollissionList = []
            if (len(self.polyCollissionList)>0):
                self.polyCollissionList[0] = self.paddlePointList
            else:
                self.polyCollissionList = [self.paddlePointList,self.borderPointList]


        def retScore(self):  # -----------  ball crashed
            global run
            run +=1
           # cfg.rundenZeit = time.time()
            #print("WAS HERE retScore")
            tmp1 = self.score
            tmp2 = run
            self.newBall()
            return tmp1, tmp2 

        def retScore2(self):  # -----------  normaler framestep waehrend dem spiel
            global run
            tmp1 = self.score
            tmp2 = run
            return tmp1, tmp2 

        def reflectBall(self,poly,ind,t1,t2,si):
                t1 = pygame.math.Vector2(t1)
                t2 = pygame.math.Vector2(t2)
                t3 = pygame.math.Vector2(t1-t2)

                if (t3.length() > 0) and (self.ballMoveVector.length() > 0):
                        t3 = t3.normalize()
                        oldLength = self.ball.getSpeed()
                        self.ballMoveVector = -self.ballMoveVector.reflect(-t3)
                        self.ballMoveVector = self.ballMoveVector.normalize()
                        self.ballMoveVector = self.ballMoveVector*self.ball.getSpeed()
                        randAngle = random.randint(-30,30)
                        if (self.ballMoveVector[1] == 0):
                            self.ballMoveVector.rotate(randAngle)
                if (ind>1):
                        #if len(poly) > 10:
                        self.deleteThis = ind
                   #     print("deleted polyIndex:",self.deleteThis,"lineIndex.:", str(si))
                   #     print ("len(getBallPosArray):",len(self.ball.getBallPosArray()))
                else:
                        self.deleteThis = -1

                points = self.ball.getBallPosArray()
                for point in points:
                        p_counter = 0
                        for p in points:
                                if (p==point):
                                        p_counter +=1
                        #print ("p_counter:",p_counter)
                        if (p_counter > 1):
                                randAngle = random.randint(-20,20)
                                self.ballMoveVector.rotate(randAngle)
                     #           print  ("rotated ----------------------------!!! um", randAngle, "°")
                                #self.ball.clearBallPosArray()

                if (ind == 0):
                        if (self.mouseVelX < 0):
                                self.ballMoveVector = self.ballMoveVector.rotate(int(self.mouseVelX*5))
                                print  ("drift --------!!! left", self.mouseVelX*5, "°")
                                self.mouseVelX = 0
                        if (self.mouseVelX > 0):
                                self.ballMoveVector = self.ballMoveVector.rotate(-int(self.mouseVelX*5))
                                print  ("drift --------!!! right", self.mouseVelX*5, "°")
                                self.mouseVelX = 0
                    
                if (self.ballMoveVector.length() > 0):
                     self.ballMoveVector = self.ballMoveVector.normalize()
                self.ballMoveVector = self.ballMoveVector*self.ball.getSpeed()



        def distance(self,a,b):
                return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


        def is_between(self,a,c,b):
                return self.distance(a,c) + self.distance(c,b) == self.distance(a,b)


        def text_objects(self, text, font):
                textSurface = font.render(text, True, (220,220,220))
                return textSurface, textSurface.get_rect()


        def message_display(self, text):
                te = np.array([])
                textArray = text.split("|")
                largeText = pygame.font.Font(None, 22)  #'freesansbold.ttf'
                zeilenSprung = 0
                count = 0
                column = 0
                for i in range(0,len(textArray)):
                        for wort in textArray[i]:
                                count += 1
                        TextSurf, TextRect = self.text_objects(textArray[i], largeText)
                        TextRect.midleft = ((3+column),(self.FIELDHEIGHT+zeilenSprung+20))
                        self.screen.blit(TextSurf, TextRect)
                        np.append(te,wort)
                
                        zeilenSprung += 15
                        if (668+zeilenSprung > self.FIELDHEIGHT-20):
                                try:
                                        column += np.max(te)*5
                                except ValueError:  #raised if `y` is empty.
                                        pass

                pygame.display.update()
                pygame.display.flip()



if __name__ == '__main__':
    pygame.init()
    g = gameState()
    while True:
        g.runGame([0,0,0,1])
        #time.sleep(2)
# g.message_display('You Crashed')
