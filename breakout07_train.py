# for snake(snaky)
import breakout07 as game
import cv2
# for tensor
import numpy as np
import threading
#from tensorboard import summary
#from tensorboard.plugins.custom_scalar import layout_pb2
import tensorflow as tf
import subprocess
import random
from collections import deque
import time
import pandas as pd
import sys
import os
import config as cfg
from pygame.locals import*
import pygame
from pynput import keyboard
import pickle
# import matplotlib as mp
# import matplotlib.pyplot as plt
# import CNNDiagrammCLASS as dc

#import tf_cnnvis
#tf.summary.FileWriterCache.clear()
# Based on NIPS 2013
eps = float(0)
class DQN:
    def __init__(self, DISCFT, FLAG, INIT_EPSILON, FIN_EPSILON, REPLAY_MEMORY, BATCH_SIZE, ACTIONS):
        # Initialize Variables
        self.run = int(0)  # game/round/run
        self.run_old = self.run
        self.epoch = int(0)    # frame
        self.episode = int(0)  # training runs     
        self.observe = int(25000) # vorlauf-frame ohne training
        self.step = 0 # self.observe #fixed observer value
        self.discft = 0.99 #DISCFT 0.9993
        cfg.discft = self.discft
        self.multiplier = float(30000/self.observe) #for saving (multiplies with self.observe)
        self.fpl = []
        self.myQV = []
        self.score = []
        self.obergrenze = int(28)
        self.flag = FLAG
        self.epsilon = INIT_EPSILON
        self.epsilon_main = 0.6 #INIT_EPSILON
        self.finep = FIN_EPSILON
        self.REPLAYMEM = 123000 # REPLAY_MEMORY
        self.batchsize = BATCH_SIZE
        self.actions = ACTIONS
        self.repmem = deque()
        self.reduce = 3000000
        self.reduce_main = 5000000  # self.reduce
        self.memSaid = 0
        self.memoryReplayCount = 250
        self.startReduce = False
        self.thisFrameWasTrained = False
        self.stepTime = ""
        self.time_now = time.time()
        self.time_stamp = time.time()
        self.actionRepeat = 4
        self.oldAction = []
        #self.minibatch_old = [deque()]
        

        if not os.path.exists(cfg.states_path + 'training_states.txt'):
            # WRITE A BRANDNEW Logfile
            print("'training_states.txt' nicht gefunden...")
            print("Eine nagelneue training_states.txt wird erstellt...")
            time.sleep(.5)
            fl = open(cfg.states_path + 'training_states.txt',"w")
            fl.write("stateTime|epsilon|qvalue|run|fpl|epoch|totalScore|repmem|rundenZeit")
            fl.write("\r\n") 
            fl.close()  
            time.sleep(.5)
            print ("--- starte training bei NULL mit epsilon = 1 ---")
        else:
            df = pd.read_csv(cfg.states_path + 'training_states.txt', sep="|", header=0, encoding="utf8", parse_dates=True)
            latest = int(len(df.index))
            if (int(latest) > 0):  
                #lastIndex = latest - 1
                lastEpisode = int(df['epoch'].max()) #int(df['epoch'].iloc[-1])
                lastEpsilon = float(df['epsilon'].iloc[-1])
                lastRun = int(df['run'].max()) #int(df['run'].iloc[-1])
                # --- User Input ---
                #msg = ' *** training-data (training_states.txt) found / continue training?'
                shall = True #shall = input("%s (y/N) " % msg).lower() == 'y'
                #print (shall)
                if (shall==True):
                    if (lastEpisode): self.episode = lastEpisode
                    if (lastEpsilon): self.epsilon = lastEpsilon 
                    if (lastRun): 
                        cfg.lastRun = lastRun  
                        self.lastRun = lastRun  

                    print ("--- starte training bei game Nr.", str(lastRun) ," ---")
                    time.sleep(2)
            else:   
                pass
        # ---------------------------- LADE REPLAYMEMORY from FILE ----------------------------
        if not os.path.exists(cfg.repmem_path + 'repmemSave.obj'):
            # WRITE A BRANDNEW Logfile
            print("'repmemSave.obj' nicht gefunden...")
            time.sleep(.5)
            print ("--- starte training mit NULL replaymemory ---")
            time.sleep(1)
        else:
            print ("- loading ReplayMemory from file...")
            filehandler = open(cfg.repmem_path + 'repmemSave.obj', 'rb') 
            obj = pickle.load(filehandler)
            filehandler.close()
            print ("- loaded - Success!! - finished with", len(obj)," entries (Eintraege).")
            time.sleep(1)
            print ("- gewünschte Groesse (ReplayMemory):", self.REPLAYMEM," entries (Eintraege).")
            time.sleep(5)
            self.repmem = obj
            tmpMEM = deque()
            if (len(self.repmem) > self.REPLAYMEM):
                for i in range(0,self.REPLAYMEM):
                    tmpMEM.append(self.repmem[i])
            else:
                for i in range(0,len(self.repmem)):
                    tmpMEM.append(self.repmem[i])
                
                self.repmem = tmpMEM
                if (int(latest) > 0):  
                    self.step = lastEpisode
                else:
                    self.step = 0
                self.observe = 0
                    
                if (self.step == 0):
                    self.observe = 0
                    self.step = 0
                        
            del tmpMEM
            del obj
        

            
            # self.observe = 100
            # self.step = 100
            time.sleep(3)

            pass


    # Init weight and bias
    #with tf.device('/gpu:0'):
        self.w1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.001))
        self.b1 = tf.Variable(tf.constant(0.01, shape = [32]))
    
        self.w2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.001))
        self.b2 = tf.Variable(tf.constant(0.01, shape = [64]))
    #with tf.device('/gpu:1'):
        self.w3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.001))
        self.b3 = tf.Variable(tf.constant(0.01, shape = [64]))

    #with tf.device('/gpu:0'):
        self.wfc = tf.Variable(tf.truncated_normal([2304, 6240], stddev = 0.001)) # 2304,2000
        self.bfc = tf.Variable(tf.constant(0.001, shape = [6240])) #2000
    # #with tf.device('/gpu:1'):
        # self.wfc1 = tf.Variable(tf.truncated_normal([1048, 982], stddev = 0.001)) #2000,1000
        #   self.bfc1 = tf.Variable(tf.constant(0.01, shape = [982])) #1000

    #with tf.device('/gpu:0'):
        self.wfc2 = tf.Variable(tf.truncated_normal([6240, 1024], stddev = 0.001)) #1000,972
        self.bfc2 = tf.Variable(tf.constant(0.001, shape = [1024])) #972
    #with tf.device('/gpu:1'):
        self.wfc3 = tf.Variable(tf.truncated_normal([1024, 512], stddev = 0.001)) #972,512
        self.bfc3 = tf.Variable(tf.constant(0.001, shape = [512])) #512
        #tf.summary.histogram("self.wfc2", self.wfc2)
        
    #with tf.device('/gpu:0'):
        self.wto = tf.Variable(tf.truncated_normal([512, self.actions], stddev = 0.001)) #512
        self.bto = tf.Variable(tf.constant(0.001, shape = [self.actions]))
        #tf.summary.histogram("self.bto", self.bto)

        self.initConvNet()
        self.initNN()
    
    def initConvNet(self):
        #with tf.device('/gpu:1'):
        # input layer
        self.input = tf.placeholder("float", [None, 84, 84, 4])
        # Convolutional Neural Network
        # zero-padding
        # 84 x 84 x 4
        # 8 x 8 x 4 with 32 Filters
        # Stride 4 -> Output 21 x 21 x 32 -> max_pool 11 x 11 x 32
        
        tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME", use_cudnn_on_gpu=True)
        conv1 = tf.nn.relu(tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME") + self.b1)
        pool = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        # 11 x 11 x 32
        # 4 x 4 x 32 with 64 Filters
        # Stride 2 -> Output 6 x 6 x 64
        conv2 = tf.nn.relu(tf.nn.conv2d(pool, self.w2, strides = [1, 2, 2, 1], padding = "SAME") + self.b2)  
        # 6 x 6 x 64
        # 3 x 3 x 64 with 64 Filters
        # Stride 1 -> Output 6 x 6 x 64
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.w3, strides = [1, 1, 1, 1], padding = "SAME") + self.b3)
        # 6 x 6 x 64 = 2304
        conv3_to_reshaped = tf.reshape(conv3, [-1, 2304])
        # Matrix (1, 2304) * (2304, 512)
        #fullyconnected = tf.nn.relu(tf.matmul(conv3_to_reshaped, self.wfc) + self.bfc)
        # Matrix (1, 2304) * (2304, 512)

    
        # Matrix (1, 2304) * (2304, 512)
        fullyconnected = tf.nn.relu(tf.matmul(conv3_to_reshaped, self.wfc) + self.bfc)
        # Matrix (1, 2304) * (2304, 2304)
        dropped = tf.nn.dropout(fullyconnected,keep_prob=0.7)
    
        #fullyconnected1 = tf.nn.relu(tf.matmul(fullyconnected, self.wfc1) + self.bfc1)     
        # Matrix (1, 2304) * (2304, 512)
        
        fullyconnected2 = tf.nn.relu(tf.matmul(dropped, self.wfc2) + self.bfc2)  
        
        #dropped2 = tf.nn.dropout(fullyconnected2,keep_prob=0.9)
        
        fullyconnected3 = tf.nn.relu(tf.matmul(fullyconnected2, self.wfc3) + self.bfc3)
        
            # output(Q) layer
            # Matrix (1, 512) * (512, ACTIONS) -> (1, ACTIONS)
        #with tf.device('/gpu:0'):
        #dropped3 = tf.nn.dropout(fullyconnected3,keep_prob=0.9)
        #dense = tf.matmul(dropped,W)+b
        self.output = tf.matmul(fullyconnected3, self.wto) + self.bto
        

    def initNN(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
        conf = tf.ConfigProto()
        #conf.gpu_options.per_process_gpu_memory_fraction = 1
        
        #with tf.device('/gpu:0'):
        self.a = tf.placeholder("float", [None, self.actions])
        self.y = tf.placeholder("float", [None]) 
        out_action = tf.reduce_sum(tf.multiply(self.output, self.a), axis = 1)       
        # Minimize error using cross entropy  
        self.cost = tf.reduce_mean(tf.square(self.y - out_action))
        # Gradient Descent   
        self.optimize = tf.train.AdamOptimizer(0.00001).minimize(self.cost) #(learning_rate=0.01, beta1=0.93, beta2=0.999, epsilon=1).minimize(self.cost)
        #tf.train.MonitoredTrainingSession()

        self.saver = tf.train.Saver(max_to_keep=2) #max_to_keep=3, keep_checkpoint_every_n_hours=2    
        self.session = tf.InteractiveSession(config=conf)
        self.session.run(tf.global_variables_initializer())
        # self.writer.add_graph(self.session.graph)
        # Write logs at every iteration
        checkpoint = tf.train.get_checkpoint_state(str(cfg.model_path))
        # For fresh start, comment below 2 lines
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            

    def train(self):
        # DQN
        minibatch = random.sample(self.repmem, self.batchsize)
        s_batch = [data[0] for data in minibatch]
        a_batch = [data[1] for data in minibatch]
        r_batch = [data[2] for data in minibatch]
        s_t1_batch = [data[3] for data in minibatch]
      #  this_index = [data[6] for data in minibatch]
        # this_qv = [data[7] for data in minibatch]
       # this_replayCount = [data[5] for data in minibatch]
        y_batch = []

        #with tf.device('/gpu:0'):
        Q_batch = self.output.eval(feed_dict={self.input : s_t1_batch})

        for i in range(0,self.batchsize):
            done = minibatch[i][4]

       
            if done:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + cfg.discft * np.max(Q_batch[i]))
        # self.repmem.append((   self.s_t, action, reward, tmp, done, int(0), int(len(self.repmem)-1), np.max(self.qv)   ))

        #with tf.device('/gpu:1'):
        self.optimize.run(feed_dict={self.y : y_batch, self.a : a_batch, self.input : s_batch})

        # ------SAVE Training Model -------
        if (self.epoch % 15000 == 0): # (self.step*self.multiplier) == 0):

            self.saver.save(self.session, cfg.model_path + 'breakout.ckpt', global_step = self.epoch)
            print("Training Model saving..complete")
            # os.system("start C:\\Windows\\WinSxS\\amd64_microsoft-windows-shell-sounds_31bf3856ad364e35_6.3.9600.16384_none_07d0dc3d89809d9b\\Ring10.wav /b")
            # pygame.mixer.music.load("C:\\Users\\chefo\\Documents\\LiClipse Workspace\\deep_learner\\qlearner\\Snake-Reinforcement-Deep-Q-Learning-master\\Ring10v.wav")
            # pygame.mixer.music.play()
            cfg.lastSave = time.strftime("%d.%m.%Y %H:%M:%S")
       


        self.episode += 1
        

    def addReplay(self, s_t1, action, reward, done):
        global eps
         # action[0] LEFT    #action[1] MIDDLE    #action[2] RIGHT    #action[3] BUTTON
        tmp = np.append(self.s_t[:,:,1:], s_t1, axis = 2)
             #np.append(self.s t[:,:,1:], s_t1, axis = 2)
        self.repmem.append([self.s_t, action, reward, tmp, done])

        if len(self.repmem) > self.REPLAYMEM:
            self.repmem.popleft()
            # print (len(self.memSaid))
        self.thisFrameWasTrained = False

        if (self.epoch >= -1): #self.step):
            if (len(self.repmem) > self.observe) and (self.epsilon > self.finep):
                self.train()
                self.thisFrameWasTrained = True


        self.s_t = tmp
        self.epoch += 1
        self.run = game.run   

        if (cfg.saveRepmemAndExit == True):
            #object_pi = str(self.repmem)
            #epo = str(self.epoch)
            file_pi = open(cfg.repmem_path + 'repmemSave.obj', 'wb') 
            pickle.dump(self.repmem, file_pi)
            file_pi.close()
            sys.exit()

        return self.epoch, np.max(self.qv)
        
    def getAction(self):
        global eps, a
        self.time_now = time.time()
        # self.time_stamp = time.time()

        # action[0] LEFT    #action[1] MIDDLE    #action[2] RIGHT    #action[3] BUTTON
        #with tf.device('/gpu:0'):
        Q_val = self.output.eval(feed_dict={self.input : [self.s_t]})[0]

        self.qv = Q_val
        self.myQV.append(np.max(self.qv))
        cfg.qValue = float(np.mean(self.myQV))

        if  (len(self.repmem) > 5000): # (cfg.qValue > 0) and 
            self.startReduce = True
        else:
            self.startReduce = False #False
          

        self.fpl.append(cfg.framesPerLife)
        # self.score.append(cfg.totalScore)
        action = np.zeros(self.actions)
        if (self.actionRepeat == 4):
            # action array
            
            idx = 0
            qv = round(cfg.qValue,7)
            self.actionRepeat -= 1
            # epsilon greedily
            if random.random() <= self.epsilon:
                idx = random.randrange(0,self.actions,1)
                print (" --   random:",str(cfg.action),"   --lm_saved: " + cfg.lastSave, "frwTrained: "+ str(self.thisFrameWasTrained)," qv:",qv, " rw:",cfg.reward," colTimer:", str(cfg.ballAlive_frames),"    ", end="\r")
                print("*", end="\r")
                self.oldAction = action
                action[idx] = 1

            else:
                print (" -- computed:",str(cfg.action),"   --lm_saved: " + cfg.lastSave, "frwTrained: "+ str(self.thisFrameWasTrained)," qv:",qv, " rw:",cfg.reward," colTimer:", str(cfg.ballAlive_frames),"    ", end="\r")
                print("*", end="\r")  
                idx = np.argmax(Q_val)
                self.oldAction = action
                action[idx] = 1
        else:
            action = self.oldAction
            self.actionRepeat -= 1
            if (self.actionRepeat == 0): self.actionRepeat = 4

        # change episilon
        if self.epsilon > self.finep and self.epoch > self.step: #self.finep
            if (self.startReduce == True):
                self.epsilon -= (1 - self.finep) / self.reduce #self.step*5

        elif (self.epsilon < self.finep):
            self.epsilon = self.epsilon_main
            self.reduce = self.reduce_main
            pass
            
                # ------------------------------------------------------- #
        if (self.thisFrameWasTrained == True): #  and not (self.run_old == self.run):  # 
                
                self.time_stamp = self.time_now  
                # ---------------------------------------  save trainingstates  --------------------------------------------------
                fl = open(cfg.states_path +'training_states.txt',"a") #states.txt' + time.strftime("%d.%m.%Y_%H_%M_%S" + '
                # time  |  epsilon  |  Q-value  |  run  |  fpl  |  epoch  |  totalscore
                fl.write(time.strftime("%d.%m.%Y %H:%M:%S"))
                fl.write("|")
                fl.write(str(self.epsilon))
                fl.write("|")
                sqv = str(np.mean(self.myQV))
                fl.write(sqv) #np.max(self.qv)
                fl.write("|")
                fl.write(str(int(self.run) + int(cfg.lastRun)))
                fl.write("|")
                fl.write(str(int(np.mean(self.fpl))))
                fl.write("|")
                fl.write(str(self.episode))
                fl.write("|")
                fl.write(str(cfg.totalScore)) #np.mean(self.score)))
                fl.write("|")
                fl.write(str(len(self.repmem)/10000))
                fl.write("|")
                if (cfg.rundeDone == True):
                    fl.write(str(cfg.rundenFZeit))
                    cfg.rundenFZeit = 0
                    cfg.rundeDone = False
                else:
                    fl.write("0")
                fl.write("\r\n") 
                fl.close()  

                #self.observe += self.step   
                self.myQV = []
                self.fpl = []
                self.score = []
                self.run_old = self.run
               # print ("training_states.txt updated.",end="\n")

        eps = self.epsilon
        cfg.qv = np.max(self.qv)
        return action

    def initState(self, state):
        self.s_t = np.stack([state, state, state, state], axis=2)

class agent:
    global epsilon, thread1, thread2

    def __init__(self):
        self.ts_old = int(0)
        self.rew = int(0)
        self.stepTime = float(0)


    def screen_handle(self, screen, a): 
        procs_screen = cv2.cvtColor(cv2.resize(screen, (84, 84)), cv2.COLOR_BGR2GRAY)
        #if (ag.episode % 10 == 0):
         #  procs_screen = cv2.flip( procs_screen, 0 )
          # a[2],a[3] = a[3],a[2]
        dummy, bin_screen = cv2.threshold(procs_screen, 1, 255, cv2.THRESH_BINARY)
        bin_screen = np.reshape(bin_screen, (84, 84, 1))
        

        return bin_screen,a


    def run(self):
        global ag
        # initialize
        # discount factor 0.99
        #       DISCFT, FLAG,  INIT_EPSILON,    FIN_EPSILON, REPLAY_MEMORY,  BATCH_SIZE,   ACTIONS
        ag = DQN(   .956,   0,       1,           0.01,     50000,       32,         4)
                        # 2.0004979999997333e-06
        g = game.gameState()

        a_0 = np.array([0, 1, 0, 0])
        s_0, r_0, d = g.frameStep(a_0)
        # if (ag.episode % 2 == 0):
        #   s_0 = cv2.flip( s_0, 0 )
        s_0 = cv2.cvtColor(cv2.resize(s_0, (84, 84)), cv2.COLOR_BGR2GRAY)
        _, s_0 = cv2.threshold(s_0, 0, 255, cv2.THRESH_BINARY)
        ag.initState(s_0)
        # screenArray = []
        qv_old = 0
        while True:
           # with keyboard.Listener(on_press=self.on_press) as listener:
            now_time = time.time()
            a = ag.getAction()
            cfg.action = a
            s_t1, r, done = g.frameStep(a) # get infos (state) from game
            if (cfg.done == False) and (done == True):
                cfg.done = done

            cfg.reward = r
            s_t1, a = self.screen_handle(s_t1,a) 
            ts, qv = ag.addReplay(s_t1, a, r, done)
            qv_old = qv
            cfg.qValue = qv
            fpl = ts - int(self.ts_old)
            # ag.fpl.append(fpl)
            # screenArray.append(s_0)
            cfg.framesPerLife = fpl    
            # print ("qv:", qv)
            sc = 0
            self.stepTime = round(time.time() - now_time, 5)
            ag.stepTime = str(self.stepTime)
            cfg.stepTime = self.stepTime
            if done == True:
                
                sc, ep = g.retScore()
                cfg.totalScore = sc
                self.ts_old = ts
                self.stepTime = round((time.time() - now_time)%60, 5)
                cfg.thisRundenZeit = round((time.time()-cfg.rundenZeit),2)
                print("discft:", round(cfg.discft,4), "repmem:", len(ag.repmem), "run:", ag.run,"episode:",ag.episode,"score:", sc,"rew:", r,"frame:", ts,"qv:", round(qv,7),"fpl:",fpl,"epsilon:",round(ag.epsilon,6), "stepTime:", self.stepTime)
                print(end="\r")
                ag.fpl = []
                ag.score = []

                
            else:
                self.stepTime = round((time.time() - now_time)%60, 5)
                sc, ep = g.retScore2()
                cfg.totalScore = sc
                if (sc > 0):          
                   # print("discft:", round(cfg.discft,4), "repmem:", len(ag.repmem), "run:", ag.run,"episode:",ag.episode,"score:", sc,"rew:", r,"frame:", ts,"qv:", round(qv,3),"fpl:",fpl,"epsilon:",round(ag.epsilon,6), "stepTime:", self.stepTime)
                   # print(end="\r")
                   pass
                    
    

def Main():
    a = agent()
    a.run()
    

if __name__ == '__main__':
    Main()
  
