import time, datetime

FPS = 60
WINDOWWIDTH = 360 # 500
SCREENWIDTH = 360 # 500
WINDOWHEIGHT = 420 # 800
FIELDHEIGHT = 420 # 650
ballReleased = False
myList = []
dateString = '%d.%m.%Y %H:%M:%S'
qValue = float(0)
stateTime = time.strftime(dateString)
frame = int(0)
gameRun = int(0)
framesPerLife = int(0)
trainingRun = int(0)
totalScore = int(0)
print (datetime.datetime.now().strftime(dateString))
stepTime = float(0)
aliveGameTime = time.time()
lastSave = time.strftime("%d.%m.%Y %H:%M:%S")
model_path = 'R:\\temp\\'
states_path = 'R:\\temp\\'
#mainPath = 'C:\\Users\\me\\Documents\\LiCliPseWorkspace\\CNN_Snake_SelfLearning\\'
lastRun = 0
qLive = []
fplLive = []
scoreLive = []
epsilonLive = []
lastRun = 0
diagON = int(0)
discft = int(0)
