import os
import multiprocessing

MAX_COUNT = 1000000000
ITERATION = 50000000

def whoami(what):
    #単純な加算
    count = 0
    for n in range(MAX_COUNT):
        if count % ITERATION ==0:
            #実行中のプロセスIDと,現在のcount数を表示
            print("{} Process {} count {}".format(what,os.getpid(),count))
        count +=1
    #どのIDのプロセスが終了したかを表示
    print("end {} Process {}".format(what,os.getpid()))
#現在のプロセスのidを表示
print("Main Process ID is {}".format(os.getpid()))
#メインのプロセスで実行
whoami("main program")

print("-----------------------------------------------------")
#プロセスを10作りスタートさせる.
for n in range(10):
    p = multiprocessing.Process(target=whoami,args=("Process {}".format(n),))
    p.start()
print("end of program")
