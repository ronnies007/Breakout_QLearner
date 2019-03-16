#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt #, mpld3
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import time
from dateutil import parser
from datetime import datetime
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import config as cfg
#import seaborn
from matplotlib.font_manager import FontProperties
#from PointBrowser import PointBrowser
import config as cfg
from matplotlib import style
import sys
import subprocess
from sklearn import preprocessing





#print(plt.style.available)
# 'seaborn-darkgrid', 'Solarize_Light2', 'seaborn-notebook', 'classic', 'seaborn-ticks',
# 'grayscale', 'bmh', 'seaborn-talk', 'dark_background', 'ggplot',
# 'fivethirtyeight', '_classic_test', 'seaborn-colorblind', 'seaborn-deep',
#  'seaborn-whitegrid', 'seaborn-bright', 'seaborn-poster', 'seaborn-muted',
# 'seaborn-paper', 'seaborn-white', 'fast', 'seaborn-pastel', 'seaborn-dark',
# 'seaborn', 'seaborn-dark-palette'
style.use('seaborn-paper')


def getMinMax(df_):
    global dmaxVal, dminVal
    
    cols1 = []
    cols1.append(int(df_['totalScore'].max()))
    cols1.append(df_['qvalue'].max())
    cols1.append(int(df_['fpl'].max())/10)
    cols1.append(df_['epsilon'].max())
    cols1.append(np.max((df_['totalScore']/df_['fpl'])/10))
    cols1.append(df_['repmem'].max())
    dmaxVal = float(np.max(cols1))

    # df_['totalScore'] = df_['totalScore']*-1
    # df_['qvalue'] = df_['qvalue']*-1
    # df_['epsilon'] = df_['epsilon']*-1
    # df_['fpl'] = df_['fpl']*-1
    # df_['repmem'] = df_['repmem']*-1

    cols2 = []
    cols2.append(int(df_['totalScore'].min()))
    cols2.append(df_['qvalue'].min())
    cols2.append(int(df_['fpl'].min())/10)
    cols2.append(df_['epsilon'].min())
    cols2.append(np.min((df_['totalScore']/df_['fpl'])/10))
    cols1.append(df_['repmem'].min())
    dminVal = float(np.min(cols2))
    

    # df_['totalScore'] = df_['totalScore']*-1
    # df_['qvalue'] = df_['qvalue']*-1
    # df_['epsilon'] = df_['epsilon']*-1
    # df_['fpl'] = df_['fpl']*-1
    # df_['repmem'] = df_['repmem']*-1
    #df_['repmem'] = df_['repmem'] / 10000

    return dmaxVal+1, dminVal-1


normON = False
# init everything first
fig = plt.figure(figsize=(7, 4))
fig.fontsize = 9
latest = 400 #1440 #int(cfg.latest)
sw = False
states_path = 'r:\\temp\\' #'C:\\Users\\me\\Documents\\LiCliPseWorkspace\\CNN_Snake_SelfLearning\\temp\\'
f=0
## ---------- wait for data file to fill ------------
while (f == 0):  
    try:
        df = pd.read_csv(states_path + 'training_states.txt', sep="|", header=0, encoding="utf8", parse_dates=True)
        if (len(df.index) > 3): 
            f = 1
            df.fillna(0)
        else:
            print ("waiting for data...", end="\r")        
    except:     
        pass
    time.sleep(2)
    

lastEpisode = int(df['epoch'].max()) #int(df['epoch'].iloc[-1])
lastRun = int(df['run'].max())
df.totalScore = df.totalScore/10
df.repmem = df.repmem/10
df.fpl = df.fpl/100
#df['totalscore'] = df['totalscore'] / 4
df['epsilon'] = df['epsilon'] * 10
dmaxVal, dminVal = getMinMax(df)

print ("dmaxVal, dminVal:", dmaxVal,",", dminVal)
zoomX1, zoomX2, zoomY1, zoomY2 = -10, lastEpisode + 10, dminVal-10, dmaxVal+10  # specify the limits
xview = lastEpisode - df['epoch'].iloc[-2]
lastEpisode_old = lastEpisode
lastRun_old = lastRun
qlive = []

# Declare and register callbacks
def on_xlims_change(axes):
    global ax1, zoomX1, zoomX2, zoomY1, zoomY2, sw, xview
    zoomX1, zoomX2 = ax1.get_xlim()
    #xview = float()
    sw = True

    # print ("updated xlims: ", zoomX1, zoomX2)


def on_ylims_change(axes):
    global ax1, zoomX1, zoomX2, zoomY1, zoomY2, sw
    zoomY1, zoomY2 = ax1.get_ylim()
    sw = True
    # print ("updated ylims: ", zoomY1, zoomY2)


def resetView():
    global ax1, zoomX1, zoomX2, zoomY1, zoomY2, plt, sw, lastEpisode, dmaxVal, dminVal, df, states_path
    df = pd.read_csv(states_path + 'training_states.txt', sep="|", header=0, encoding="utf8", parse_dates=True)
    df.fillna(0)
    df.totalScore = df.totalScore/10
    df.repmem = df.repmem/10
    df.fpl = df.fpl/100
    #df['totalscore'] = df['totalscore'] / 4
    df['epsilon'] = df['epsilon'] * 10

    dmaxVal, dminVal = getMinMax(df)

    zoomX1, zoomX2, zoomY1, zoomY2 = -10, lastEpisode + 10, dminVal-10, dmaxVal+10  # specify the limits
    sw = False
    plt.xlim(zoomX1, zoomX2)
    plt.ylim(zoomY1, zoomY2)
    plt.show()

def normalizeData():
    global df, normON
    
    df.qvalue = df.qvalue.reshape(1, -1) 
   # x = df['Q-value'] #returns a numpy array

  #  min_max_scaler = preprocessing.MinMaxScaler()
  #  x_scaled = min_max_scaler.fit_transform(x)
  #  df = df(x_scaled)
    return df

def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == ' ':
        resetView()
    if event.key == 'n':
        normalizeData()


def animate(i):
    global dminVal, lastRun_old, lastRun, dmaxVal, qlive, ax1, zoomX1, zoomX2, zoomY1, zoomY2, plt, df, sw, xview, lastEpisode_old, states_path, lastEpisode, normON
    #ax1 = fig.add_subplot(1,1,1)
    #ax1.xaxis_date()
    #plt.savefig('test.pdf')
    print(str(cfg.qValue))
    #ax1.set_xlim(xlim)
    #ax1.set_ylim(ylim)
    #dataLOG einlesen
    swt = False
    try:
        df = pd.read_csv(states_path + 'training_states.txt', sep="|", header=0, encoding="utf8", parse_dates=True)
        df.fillna(0)
        df.totalScore = df.totalScore/10
        swt = True

    except:
        print ("file problem..")
        swt = False
        pass
    if (swt == True):
        if (normON==True):    
            normalizeData()
        lastEpisode = int(df['epoch'].max()) #int(df['epoch'].iloc[-1])
        # korrekturen
        lastRun = int(df['run'].max())
        df.repmem = df.repmem/10
        df.fpl = df.fpl/100
        #df['totalscore'] = df['totalscore'] / 4
        df['epsilon'] = df['epsilon'] * 10
        dmaxVal, dminVal = getMinMax(df)

        # print ("was here")
        xview = int(lastEpisode - df['epoch'].iloc[-2])   
        print (xview)   
        # - alles im file anzeigen
        latest = len(df.index)
        #angezeigte data reduzieren auf d. letzten 24h
        if len(df.index) > int(latest):        # Drop alle index´ vor den letzten 24h = 1440
            print('alte tabellenLaenge :'+str(len(df.index)))
            ld = len(df.index) - int(latest)
            df.drop(df.index[:ld], inplace=True)
            print('dropped '+str(ld)+' entries.')
            print('neue tabellenLaenge :'+str(len(df.index)))
        else:                           # do not drop (log kleiner als 24h = 1440)
            print('nothing to drop')
            print('\r\n')
        #                                      time | epsilon | Q-value | run | fpl
        # start setup Plotting
        #ax1.clear()    #alten plot loeschen
        fig.clear()
        ax1 = fig.add_subplot(1,1,1)
        #axins = zoomed_inset_axes(ax, 2.5, loc=2) # zoom-factor: 2.5, location: upper-left

        #plt.set_xlim(x1, x2) # apply the x-limits
        #plt.set_ylim(y1, y2) # apply the y-limits
        #ax1.xaxis_date()
        #ax1.xaxis(df.epoch)
        ax1.set_xlabel("epochs = trained frames") #("runs = games (trained games)")    #x-achse setzen
        ax1.set_ylabel("qvalue")  #y-achse setzen
        ax1.yaxis.grid(color='gray', linestyle='dashed') #hintergrund gitter
        ax1.xaxis.grid(color='gray', linestyle='dashed') #hintergrund gitter
        #ax1.autoscale_view()
        ax1.set_facecolor((0.8, 0.8, 0.8))   #Hintergrund grau

        df.stateTime = pd.to_datetime(df['stateTime'], format='%d.%m.%Y %H:%M:%S') # zeit irgendwie parsen oder so etwas..
        #df.frameTime = pd.to_datetime(df['stateTime'], format='%M:%S') # zeit irgendwie parsen oder so etwas..
        #thisFrameTime = df.frameTime.iloc[-1] - df.frameTime.iloc[-2]
        #frameTime = time.strftime("%M:%S")
        #print ("frameTime: " + thisFrameTime)
        # d = datetime.datetime.strptime(str(df['stateTime'].iloc[-1]-df['stateTime'].iloc[-2]), '%MM:%SS')
        this=str(df.stateTime[-1:])
        print (this)
        this=pd.to_timedelta(str(df['stateTime'].iloc[-1]-df['stateTime'].iloc[-2]))
        print ("Delta:" , str(this))

        ax1.set_title('cnn training state  @trained frame '+ str(int(np.max(df.epoch)+2)) + ' in ' + str(df.stateTime.iloc[-1] - df.stateTime.iloc[0])) # + "frameTime: " + str(thisFrameTime))

        for label in ax1.xaxis.get_ticklabels(): label.set_rotation(20)  #x-achse beschriftung rotieren
        print ("hello here")
        
        #df.M1State = df.M1State/10
        #df.newPos = df.newPos/10
        #df.P1State = df.P1State*10
        #ratio = (df['totalscore']/df['fpl'])/10
        plt.tight_layout()
        #print (df['fpl'])
        
        #lastEpsilon = df['epsilon'].iloc[-1]
        print ("lastEpisode: ", lastEpisode)

        #         time | epsilon | Q-value | run | fpl | epoch | totalscore

        #zoomX1, zoomX2, zoomY1, zoomY2 = 0, lastEpisode, -1, 8 # specify the limits
        if (sw==False):        # nicht gezoomt oder bewegt = reset
            if not (lastRun_old == lastRun): 
                dmaxVal, dminVal = getMinMax(df)         
                zoomX1, zoomX2, zoomY1, zoomY2 = -10, lastEpisode + 10, dminVal-10, dmaxVal+10  # specify the limits
                lastRun_old = lastRun
                #xview = int(df['epoch'].iloc[-1]) - int(df['epoch'].iloc[-2]) 
                plt.xlim(zoomX1, zoomX2)
                plt.ylim(zoomY1, zoomY2)
                #zoomX1, zoomX2 = ax1.get_xlim()
                #plt.show()
            else:
                plt.xlim(zoomX1, zoomX2)
                plt.ylim(zoomY1, zoomY2)

        elif (sw==True):      # gezoomt oder bewegt
            if not (lastRun_old == lastRun):              
                lastRun_old = lastRun
                xdiff = int(df['epoch'].iloc[-1]) - int(df['epoch'].iloc[-2]) 
                print ("xdiff:",xdiff)
                zoomX1 += xdiff#*(1/(zoomX2-zoomX1))
                zoomX2 += xdiff#*(1/(zoomX2-zoomX1))
                plt.xlim(int(zoomX1), int(zoomX2))
                plt.ylim(zoomY1, zoomY2)
                zoomX1, zoomX2 = ax1.get_xlim()
                #plt.show()
            else:
                plt.xlim(zoomX1, zoomX2)
                plt.ylim(zoomY1, zoomY2)
                zoomX1, zoomX2 = ax1.get_xlim()

        plot7, = plt.plot(0, 0, label="rundZt.", marker="o", picker=3, linewidth=0, color='k')
        for a,b in zip(df.epoch, df['rundenZeit']): 
            if not (b==0):
                plt.text(a, b, str(b))
                plot7, = plt.plot(df.epoch, df['rundenZeit']/100, label="rundZt.", marker="o", picker=3, linewidth=0, color='k')
        plot2, = plt.plot(df.epoch, df['qvalue'], label="qv.", marker="", picker=3, linewidth=2, color='b')
        plot6, = plt.plot(df.epoch, df['repmem'], label="repmem", marker="+", picker=5, linewidth=.8, color='w')
        plot5, = plt.plot(df.epoch, (df['totalScore']/df['fpl'])/10, label="eat-ratio", marker="", picker=3, linewidth=.8, color='m')
        plot4, = plt.plot(df.epoch, df['fpl'], label="fpl", marker="", picker=3, linewidth=1, color='grey')
        plot1, = plt.plot(df.epoch, df['epsilon'], label="eps.", marker="", picker=3, linewidth=1.6, color='r')
        plot3, = plt.plot(df.epoch, df['totalScore'], label="score", marker="", picker=1, linewidth=2, color='c')

        

        
        
        #print(str(cfg.qValue))
        #plot7, = plt.plot(df.stateTime, df['cfg.maxVTemp'], label="maxV", marker="x", linestyle='dashed', picker=3)
        #plot8, = plt.plot(df.stateTime, df['fehler'], label="F1", marker="o", picker=5)

        #ax2 = ax1.twinx()
        #ax2.set_ylim([-30,30])

        plt.legend([plot1,plot2,plot3,plot4,plot5,plot6,plot7],['epsilon *10', 'avg. qv','score / 10','avg. fpl /100','eat-ratio /10','repmem /100000',"rundZt.(sec.)/100"], loc="upper left",  bbox_transform=fig.transFigure)
        #fig.savefig('test.pdf')

        #ax2.plot(df.time, df['fehler'], label="F1", marker="o", color='C7', picker=5)
        #ax2.set_ylabel('fehler', color='C7', font="Arial")
        #ax2.tick_params('f', color='C7', font="Arial")
        
        ax1.callbacks.connect('xlim_changed', on_xlims_change)
        ax1.callbacks.connect('ylim_changed', on_ylims_change)
        fig.canvas.mpl_connect('key_press_event', press)

    

ani = animation.FuncAnimation(fig, animate, interval=3000)
#fig.canvas.mpl_connect('button_press_event', onpress)
plt.show()
