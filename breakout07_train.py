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
        self.observe = int(10000) # vorlauf-frame ohne training
        self.step = self.observe #fixed observer value
        self.discft = 0.934 #DISCFT 0.9993
        cfg.discft = self.discft
        self.multiplier = float(5000/self.observe) #for saving (multiplies with self.observe)
        self.fpl = []
        self.myQV = []
        self.score = []
        self.obergrenze = int(28)
        self.flag = FLAG
        self.epsilon = INIT_EPSILON
        self.epsilon_main = 0.5 #INIT_EPSILON
        self.finep = FIN_EPSILON
        self.REPLAYMEM = REPLAY_MEMORY
        self.batchsize = BATCH_SIZE
        self.actions = ACTIONS
        self.repmem = deque()
        self.reduce = 1000000
        self.reduce_main = 1000000  # self.reduce
        self.logs_path = '\\\\127.0.0.1\\temp\\'
        self.model_path = '\\\\127.0.0.1\\temp\\'
        self.states_path = '\\\\127.0.0.1\\temp\\'
        self.memSaid = 0
        self.memoryReplayCount = 250
        self.startReduce = False
        self.thisFrameWasTrained = False
        self.stepTime = ""
        #self.minibatch_old = [deque()]
        if not os.path.exists(self.states_path + 'repmem.txt'):
            # WRITE A BRANDNEW Logfile
            print("'repmem.txt' nicht gefunden...")

            time.sleep(.5)
            print ("--- starte training mit NULL replaymemory ---")
            time.sleep(1)
        else:
            pass

        if not os.path.exists(self.states_path + 'training_states.txt'):
            # WRITE A BRANDNEW Logfile
            print("'training_states.txt' nicht gefunden...")
            print("Eine nagelneue training_states.txt wird erstellt...")
            time.sleep(.5)
            fl = open(self.states_path + 'training_states.txt',"w")
            fl.write("stateTime|epsilon|qvalue|run|fpl|epoch|totalScore|repmem")
            fl.write("\r\n") 
            fl.close()  
            time.sleep(.5)
            print ("--- starte training bei NULL ---")
        else:
            df = pd.read_csv(self.states_path + 'training_states.txt', sep="|", header=0, encoding="utf8", parse_dates=True)
            latest = int(len(df.index))
            if (int(latest) > 0):  
                #lastIndex = latest - 1
                lastEpisode = int(df['epoch'].max()) #int(df['epoch'].iloc[-1])
                lastEpsilon = float(df['epsilon'].iloc[-1])
                lastRun = int(df['run'].max()) #int(df['run'].iloc[-1])
                # --- User Input ---
                msg = ' *** training-data (training_states.txt) found / continue training?'
                shall = True #shall = input("%s (y/N) " % msg).lower() == 'y'
                print (shall)
                if (shall==True):
                    if (lastEpisode): self.episode = lastEpisode
                    if (lastEpsilon): self.epsilon = lastEpsilon 
                    if (lastRun): 
                        cfg.lastRun = lastRun  
                        self.lastRun = lastRun  

                    print ("--- starte training bei game Nr.", str(lastRun) ," ---")
            else:   
                pass
 
        


        # Init weight and bias
        with tf.device('/gpu:1'):
            self.w1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.01))
            self.b1 = tf.Variable(tf.constant(0.01, shape = [32]))

            self.w2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            self.b2 = tf.Variable(tf.constant(0.01, shape = [64]))

        with tf.device('/gpu:0'):
            self.w3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01))
            self.b3 = tf.Variable(tf.constant(0.01, shape = [64]))

           # self.wfc = tf.Variable(tf.truncated_normal([2304, 2304], stddev = 0.01)) # 2304,2000
          #  self.bfc = tf.Variable(tf.constant(0.01, shape = [2304])) #2000

            self.wfc1 = tf.Variable(tf.truncated_normal([2304, 1028], stddev = 0.01)) #2000,1000
            self.bfc1 = tf.Variable(tf.constant(0.01, shape = [1028])) #1000

            self.wfc2 = tf.Variable(tf.truncated_normal([1028, 1000], stddev = 0.01)) #1000,972
            self.bfc2 = tf.Variable(tf.constant(0.01, shape = [1000])) #972

        with tf.device('/gpu:1'):
            self.wfc3 = tf.Variable(tf.truncated_normal([1000, 512], stddev = 0.01)) #972,512
            self.bfc3 = tf.Variable(tf.constant(0.01, shape = [512])) #512
            #tf.summary.histogram("self.wfc2", self.wfc2)

            self.wto = tf.Variable(tf.truncated_normal([512, self.actions], stddev = 0.01)) #512
            self.bto = tf.Variable(tf.constant(0.01, shape = [self.actions]))
            #tf.summary.histogram("self.bto", self.bto)

        self.initConvNet()
        self.initNN()
    
    def initConvNet(self):
        # input layer
        self.input = tf.placeholder("float", [None, 84, 84, 4])
        # Convolutional Neural Network
        # zero-padding
        # 84 x 84 x 4
        # 8 x 8 x 4 with 32 Filters
        # Stride 4 -> Output 21 x 21 x 32 -> max_pool 11 x 11 x 32
        with tf.device('/gpu:0'):
            tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME", use_cudnn_on_gpu=True)
            conv1 = tf.nn.relu(tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME", use_cudnn_on_gpu=True) + self.b1)
            pool = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
            # 11 x 11 x 32
            # 4 x 4 x 32 with 64 Filters
            # Stride 2 -> Output 6 x 6 x 64
            conv2 = tf.nn.relu(tf.nn.conv2d(pool, self.w2, strides = [1, 2, 2, 1], padding = "SAME", use_cudnn_on_gpu=True) + self.b2)  
            # 6 x 6 x 64
            # 3 x 3 x 64 with 64 Filters
            # Stride 1 -> Output 6 x 6 x 64
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.w3, strides = [1, 1, 1, 1], padding = "SAME", use_cudnn_on_gpu=True) + self.b3)
            # 6 x 6 x 64 = 2304
            conv3_to_reshaped = tf.reshape(conv3, [-1, 2304])
            # Matrix (1, 2304) * (2304, 512)
            #fullyconnected = tf.nn.relu(tf.matmul(conv3_to_reshaped, self.wfc) + self.bfc)
            # Matrix (1, 2304) * (2304, 512)

            # Matrix (1, 2304) * (2304, 512)
            #fullyconnected = tf.nn.relu(tf.matmul(conv3_to_reshaped, self.wfc) + self.bfc)
            # Matrix (1, 2304) * (2304, 512)

        with tf.device('/gpu:1'):
            fullyconnected1 = tf.nn.relu(tf.matmul(conv3_to_reshaped, self.wfc1) + self.bfc1)     
            # Matrix (1, 2304) * (2304, 512)
            
            fullyconnected2 = tf.nn.relu(tf.matmul(fullyconnected1, self.wfc2) + self.bfc2)     
            
            fullyconnected3 = tf.nn.relu(tf.matmul(fullyconnected2, self.wfc3) + self.bfc3)
            
            # output(Q) layer
            # Matrix (1, 512) * (512, ACTIONS) -> (1, ACTIONS)
            self.output = tf.matmul(fullyconnected3, self.wto) + self.bto
        

    def initNN(self):
        self.a = tf.placeholder("float", [None, self.actions])
        self.y = tf.placeholder("float", [None]) 
        out_action = tf.reduce_sum(tf.multiply(self.output, self.a), axis = 1)       
            # Minimize error using cross entropy  
        self.cost = tf.reduce_mean(tf.square(self.y - out_action))
            # Gradient Descent   
        self.optimize = tf.train.AdamOptimizer(0.00001).minimize(self.cost) #(learning_rate=0.01, beta1=0.93, beta2=0.999, epsilon=1).minimize(self.cost)
        #tf.train.MonitoredTrainingSession()
        self.saver = tf.train.Saver(max_to_keep=3) #max_to_keep=3, keep_checkpoint_every_n_hours=2
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' #use GPU with ID=0
        conf = tf.ConfigProto()
        # conf.gpu_options.per_process_gpu_memory_fraction = 1
        
        

        # sess = tf.Session(config = config)
        self.session = tf.InteractiveSession(config=conf)
        self.session.run(tf.global_variables_initializer())
        # self.writer.add_graph(self.session.graph)
        # Write logs at every iteration
        checkpoint = tf.train.get_checkpoint_state(str(self.model_path))
        # For fresh start, comment below 2 lines
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        
    def addValue(self):
        print("was here +")
        self.discft += 0.1
        cfg.discft += 0.1

    def subtractValue(self):
        print("was here -")
        self.discft -= 0.1
        cfg.discft -= 0.1

    def saveCheckPoint(self):
        self.saver.save(self.session, self.model_path + 'snake.ckpt', global_step = self.epoch)
        print("Training Model saving..complete")
        time.sleep(1)
        print("..stopped training and closing app.")
    
    def train(self):
        # DQN
        minibatch = random.sample(self.repmem, self.batchsize)
        s_batch = [data[0] for data in minibatch]
        a_batch = [data[1] for data in minibatch]
        r_batch = [data[2] for data in minibatch]
        s_t1_batch = [data[3] for data in minibatch]
        this_index = [data[6] for data in minibatch]
        # this_qv = [data[7] for data in minibatch]
        this_replayCount = [data[5] for data in minibatch]
        y_batch = []
        Q_batch = self.output.eval(feed_dict={self.input : s_t1_batch})
        for i in range(0,self.batchsize):
            done = minibatch[i][4]

            # if (len(self.repmem) >= self.batchsize*2) and (len(self.repmem) <= self.REPLAYMEM):
            #     #print (this_index[i])
            #     self.repmem[this_index[i]] = (s_batch[i],a_batch[i],r_batch[i],s_t1_batch[i],minibatch[i][4],int(this_replayCount[i])+1,this_index[i], minibatch[i][7])

            if done:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + cfg.discft * np.max(Q_batch[i]))

        self.optimize.run(feed_dict={self.y : y_batch, self.a : a_batch, self.input : s_batch})

        # ------SAVE Training Model -------
        if (self.epoch % (self.step*self.multiplier) == 0):

            self.saver.save(self.session, self.model_path + 'snake.ckpt', global_step = self.epoch)
            print("Training Model saving..complete")
            # os.system("start C:\\Windows\\WinSxS\\amd64_microsoft-windows-shell-sounds_31bf3856ad364e35_6.3.9600.16384_none_07d0dc3d89809d9b\\Ring10.wav /b")
            # pygame.mixer.music.load("C:\\Users\\chefo\\Documents\\LiClipse Workspace\\deep_learner\\qlearner\\Snake-Reinforcement-Deep-Q-Learning-master\\Ring10v.wav")
            # pygame.mixer.music.play()
            cfg.lastSave = time.strftime("%d.%m.%Y %H:%M:%S")
            # --------------------------------- REPLAYMEMORY reduktion (MAGIC) --------------------------------------
            # i=len(self.repmem)-1
            # ll = []
            # qq = []
            # for a in range(0,len(self.repmem)):
            #     ll.append(self.repmem[a][5])
            #     qq.append(self.repmem[a][7])
            # count = 0
            # limit = int(np.max(ll) - (np.mean(ll)/3.14159265359))
            # qlimit = float(np.min(qq) + (np.mean(qq)/2)) #3.14159265359))
            # if (int(limit) > 5):
            #     for a in range(0,len(self.repmem)):
            #         #print (np.min(qlimit))
            #         if (self.repmem[i][7] <= qlimit) or (self.repmem[i][5] >= limit): 
                        
            #             print ("********** Memory-Entry deleted ! *************  nr.",i,"count:",str(self.repmem[i][5])," qv:",str(qlimit))  #  
            #             del self.repmem[i]
            #             count +=1
            #         else:
            #             pass
            #         i -=1
            # print ("llmax, llmean: ", np.max(ll),",", np.mean(ll))
            # print ("qqmin, qqmean: ", np.min(qq),",", np.mean(qq))
            
            # print (count,"eintraege aus dem 'repmem' geloescht.")
            # if not (count == 0):
            #     print ("indiziere replay memory neu..")
            #     for i in range(0,len(self.repmem)):
            #         self.repmem[i] = (self.repmem[i][0],self.repmem[i][1],self.repmem[i][2],self.repmem[i][3],self.repmem[i][4],self.repmem[i][5],i,self.repmem[i][7])
            #         print ("nr.",i,end="\r")
            #     print ("...fertig")


        self.episode += 1
        

    def addReplay(self, s_t1, action, reward, done):
        global eps
        # action[0] up    #action[1] down    #action[2] left    #action[3] right
        tmp = np.append(self.s_t[:,:,1:], s_t1, axis = 2)
        self.repmem.append((   self.s_t, action, reward, tmp, done, int(0), int(len(self.repmem)-1), np.max(self.qv)   ))

        if len(self.repmem) > self.REPLAYMEM:
            self.repmem.popleft()
            # print (len(self.memSaid))
        self.thisFrameWasTrained = False

        if (self.epoch > self.step):
            if (len(self.repmem) > self.batchsize * 2) and (self.epsilon > self.finep):
                self.train()
                self.thisFrameWasTrained = True


        self.s_t = tmp
        self.epoch += 1
        self.run = game.run   
        return self.epoch, np.max(self.qv)
        
    def getAction(self):
        global eps

        # action[0] up    #action[1] down    #action[2] left    #action[3] right

        Q_val = self.output.eval(feed_dict={self.input : [self.s_t]})[0]

        self.qv = Q_val
        self.myQV.append(self.qv)
        cfg.qValue = float(np.max(self.myQV))

        if  (len(self.repmem) > 5000): # (cfg.qValue > 0) and 
            self.startReduce = True
        else:
            self.startReduce = False #False
          

        self.fpl.append(cfg.framesPerLife)
        # self.score.append(cfg.totalScore)


        # action array
        action = np.zeros(self.actions)
        idx = 0

        # epsilon greedily
        if random.random() <= self.epsilon:
            idx = random.randrange(self.actions)
            print (" ---- random   action ----    last model saved: " + cfg.lastSave,  " frameTrained: "+ str(self.thisFrameWasTrained), end="\r")   
            print("*", end="\r")
            action[idx] = 1

        else:
            print (" ---- computed action ----    last model saved: " + cfg.lastSave, " frameTrained: "+ str(self.thisFrameWasTrained), end="\r")   
            print("*", end="\r")  
            idx = np.argmax(Q_val)
            action[idx] = 1

        # change episilon
        if self.epsilon > self.finep and self.epoch > self.step: #self.finep
            if (self.startReduce == True):
                self.epsilon -= (1 - self.finep) / self.reduce #self.step*5

        elif (self.epsilon <= self.finep):
            self.epsilon = self.epsilon_main
            pass
            
                # ------------------------------------------------------- #
        if (self.run % 10 == 0) and (self.epoch > self.step) and not (self.run_old == self.run):  # 
                # ---------------------------------------  save trainingstates  --------------------------------------------------
                fl = open(self.states_path +'training_states.txt',"a") #states.txt' + time.strftime("%d.%m.%Y_%H_%M_%S" + '
                # time  |  epsilon  |  Q-value  |  run  |  fpl  |  epoch  |  totalscore
                fl.write(time.strftime("%d.%m.%Y %H:%M:%S"))
                fl.write("|")
                fl.write(str(self.epsilon))
                fl.write("|")
                sqv = str(np.max(self.myQV))
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
                fl.write("\r\n") 
                fl.close()  
                self.observe += self.step   
                self.myQV = []
                self.fpl = []
                self.score = []
                self.run_old = self.run

               # print ("training_states.txt updated.",end="\n")

        eps = self.epsilon
        cfg.qv = np.max(self.qv)
        return action

    def initState(self, state):
        self.s_t = np.stack((state, state, state, state), axis=2)

class agent:
    global epsilon, thread1, thread2

    def __init__(self):
        self.ts_old = int(0)
        self.rew = int(0)
        self.stepTime = float(0)

    def on_press(self, key):
        global ag
        sys.stdout.flush()
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
           # if (str(format(key.char)) == "+"):
           #     ag.addValue()

          #  if (str(format(key.char)) == "-"):
           #     ag.subtractValue()
            
            if (str(format(key.char)) == "ESC"):
                ag.saveCheckPoint()
                sys.exit()

        except AttributeError:
            print('special key {0} pressed'.format(
                key))

            if (str(format(key)) == "ESC"):
                sys.exit()

        


    def on_release(self, key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False


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
        ag = DQN(   .956,   0,       1,           0.0001,     300000,        64,         4)
                        # 2.0004979999997333e-06
        g = game.gameState()

        a_0 = np.array([1, 0, 0, 0])
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
            s_t1, r, done = g.frameStep(a) # get infos (state) from game
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
                # print(ts,",",qv,",",ep, ",", sc)
                print("discft:", round(cfg.discft,4), "repmem:", len(ag.repmem), "run:", ag.run,"episode:",ag.episode,"score:", sc,"rew:", r,"frame:", ts,"qv:", round(qv,3),"fpl:",fpl,"epsilon:",round(ag.epsilon,6), "stepTime:", self.stepTime)
                # print ("                                  lastSave: ", cfg.lastSave)
                # print("                                                                                         **Flop!**", end="\r")
                print(end="\r")
                ag.fpl = []
                ag.score = []
                
            else:
                if (r>=10) and (sc):          
                    print("discft:", round(cfg.discft,4), "repmem:", len(ag.repmem), "run:", ag.run,"episode:",ag.episode,"score:", sc,"rew:", r,"frame:", ts,"qv:", round(qv,3),"fpl:",fpl,"epsilon:",round(ag.epsilon,6), "stepTime:", self.stepTime)
                # print ("                                         lastSave: ", cfg.lastSave)
                    print("                                                                                                               **Apple!**", end="\r")
                    print(end="\r")
                #   sys.stdout.write("\033[F") # Cursor up one line
                    
    

def Main():
    # global thread1, thread2
    a = agent()
    # Collect events until released
    # listener.join()
    a.run()
    

if __name__ == '__main__':
    Main()
    
  