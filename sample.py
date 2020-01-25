import time
import concurrent.futures


def func1():
    while True:
        print("func1")
        #time.sleep(0.01)


def func2():
    while True:
        print("func2")
        #time.sleep(0.01)


if __name__ == "__main__":
    #print('start')
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    executor.submit(func1)
    executor.submit(func2)
    #print('end')