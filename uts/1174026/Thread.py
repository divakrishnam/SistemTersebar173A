# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:45:18 2020

@author: ACER
"""

# In[]

import threading
import time



def sleeper(n, name):
    print('Hi, I am {}. Going to sleep for 5 seconds \n'.format(name))
    time.sleep(n)
    print('{} has woken up from sleep \n'.format(name))
    
    
    
    
t = threading.Thread(target = sleeper, name = 'thread1', args =(5, 'thread1') )
 
 
t.start()
 
 
 
t.join()
 
 
print('hello')
print('what is going on')

# In[]

import threading 


def MyFunction(num1, num2):
	"""This is user defined thread function"""
	print ("Given numbers= %s, %s" %(num1, num2))
	print ("Result = %d" %(int(num1)+int(num2)))
	return


def Main():
	t = threading.Thread(
			target=MyFunction, 
			args=(12,13)
		)
	t.start()


if __name__ == '__main__':
	Main()
    
# In[]
    
import threading
import time



def sleeper(n, name):
    print('Hi, I am {}. Going to sleep for 5 seconds \n'.format(name))
    time.sleep(n)
    print('{} has woken up from sleep \n'.format(name))
    
    
 
start = time.time()
threads = []

for i in range(5):
    t = threading.Thread(target = sleeper, name = 'thread{}'.format(i), args =(5,'thread{}'.format(i) ) )
    threads.append(t)
    t.start()
    print('{} has started \n'.format(t.name))
 
 
 
    
for i in threads:

    i.join()
    
end = time.time()



print('time is {}'.format(end - start))






#example 2 without threads


import time

def sleeper(n, i):
    print ('Hi, I am function {}. Going to sleep for 5 seconds \n'.format(i))
    time.sleep(n)
    print('function{} has woken up from sleep \n'.format(i))




start = time.time()

for i in range(5):
    print('{} has started \n'.format(i))
    x = sleeper(5, i)
    
    
end  = time.time()


print('time taken: {}'.format(end - start))
    
# In[]
    
import threading
import time

total = 4

def creates_items():
    global total
    for i in range(10):
        time.sleep(2)
        print('added item')
        total += 1
    print('creation is done')


def creates_items_2():
    global total
    for i in range(7):
        time.sleep(1)
        print('added item')
        total += 1
    print('creation is done')


def limits_items():
    
    #print('finished sleeping')
    
    global total
    while True:
        if total > 5:

            print ('overload')
            total -= 3
            print('subtracted 3')
        else:
            time.sleep(1)
            print('waiting')


creator1 = threading.Thread(target  = creates_items)
creator2 = threading.Thread(target = creates_items_2)
limitor = threading.Thread(target = limits_items, daemon = True)

print(limitor.isDaemon())


creator1.start()
creator2.start()
limitor.start()


creator1.join()
creator2.join()




print('our ending value of total is' , total)

# In[]
    
# Gabisa di run eror line 177, benerin kalo bisa    
    
import threading 
 
class PrimeNumber(threading.Thread): 
  def __init__(self, number): 
    threading.Thread.__init__(self) 
    self.Number = number
 
  def run(self): 
    counter = 2 
    while counter*counter < self.Number: 
      if self.Number % counter == 0: 
        print ("%d is no prime number, because %d = %d * %d" % ( self.Number, self.Number, counter, self.Number / counter))
                return 
            counter += 1 
        print "%d is a prime number" % self.Number
threads = [] 
while True: 
    input = long(raw_input("number: ")) 
    if input < 1: 
        break 
 
    thread = PrimeNumber(input) 
    threads += [thread] 
    thread.start() 
 
for x in threads: 
    x.join()
    
# In[]

import time
import threading
from threading import Thread


def ThreadFunction(i):
    print("Thread %i going to sleep for 5 seconds." % i)
    time.sleep(5)
    print("Thread %i is awake now." % i)


def Main():
  for i in range(10):
      myThread = Thread(target=ThreadFunction, args=(i, ))
      myThread.start()
      print("Current Thread count: %i." % threading.active_count())
    
    
if __name__ == "__main__":    
  Main()
  
# In[]
  
import threading
import time


def HelloWorld():
	"""User defined Thread function"""
	print ("Hello World")
	return


def Main():
	threads = [] # Threads list needed when we use a bulk of threads
	print ("Program started.  This program will print Hello World five times...")
	for i in range(5):
		mythread = threading.Thread(target=HelloWorld)
		threads.append(mythread)
		time.sleep(2)
		mythread.start()
	print ("Done! Program ended")


if __name__ == "__main__":
	Main()
    
# In[]
    
import threading
import datetime


class myThread (threading.Thread):
    def __init__(self, name, counter):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name
        self.counter = counter

    def run(self):
        print ("Starting " + self.name)
        print_date(self.name, self.counter)
        print ("Exiting " + self.name)

def print_date(threadName, counter):
    datefields = []
    today = datetime.date.today()
    datefields.append(today)
    print ("%s[%d]: %s" % ( threadName, counter, datefields[0] ))


# Create new threads
thread1 = myThread("Thread", 1)
thread2 = myThread("Thread", 2)


# Start new Threads
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print ("Exiting the Program!!!")

# In[]

import threading
import time


def ThreadFunction():
	print (threading.currentThread().getName(), "Starting")
	time.sleep(2)
	print (threading.currentThread().getName(), "Exiting")


def Main():
	w = threading.Thread(target=ThreadFunction)
	w.start()


if __name__ == '__main__':
	Main()

# In[]
    
import random
import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def student():
    """thread student function"""
    t = threading.currentThread()
    pause = random.randint(1,5)
    logging.debug('sleeping %s', pause)
    time.sleep(pause)
    logging.debug('ending')
    return

for i in range(3):
    t = threading.Thread(target=student)
    t.setDaemon(True)
    t.start()

main_thread = threading.currentThread()
for t in threading.enumerate():
    if t is main_thread:
        continue
    logging.debug('joining %s', t.getName())
    t.join()
    
# In[]
    
from thread import start_new_thread

threadId = 1


def factorial(n):
   global threadId
   if n < 1:   # base case
       print ("%s: %d" % ("Thread", threadId ))
       threadId += 1
       return 1
   else:
       returnNumber = n * factorial( n - 1 )  # recursive call
       print(str(n) + '! = ' + str(returnNumber))
       return returnNumber

start_new_thread(factorial,(5, ))
start_new_thread(factorial,(4, ))

c = raw_input("Waiting for threads to return...\n")

# In[]

import time
from threading import Thread


def ThreadFunction(x):
    print "sleeping 5 sec from thread %d" % x
    time.sleep(5)
    print "finished sleeping from thread %d" % x

def Main():
  for i in range(10):
      myThread = Thread(target=ThreadFunction, args=(i,))
      myThread.start()
      
if __name__ == "__main__":
  Main()
  
# In[]
  
import threading


def MyFunction():
	"""This is a user defined function"""
	print "Hello World"
	return


def Main():
	"""This is where we create a thread. 
	Target means run this function when a thread is initiated."""
	myThread = threading.Thread(target=MyFunction) 	
	myThread.start() 	# Starting a thread


if __name__ == '__main__':
	Main()
    
# In[]
    
import threading

localSchool = threading.local()


def processStudent():
    std = localSchool.student
    print('Hello, %s (in %s)' % (std, threading.current_thread().name))

def processThread(name):
    localSchool.student = name
    processStudent()

t1 = threading.Thread(target=processThread, args=('Alice', ), name='T-a')
t2 = threading.Thread(target=processThread, args=('Bob', ), name='T-b')

t1.start()
t2.start()

t1.join()
t2.join()
  
# In[]
  
import time, threading

# 假定这是你的银行存款:
balance = 0
lock = threading.Lock()

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(100000):
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
    
# In[]
    
import logging
import random
import threading
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s (%(threadName)-2s) %(message)s',
                    )

class ActivePool(object):
    def __init__(self):
        super(ActivePool, self).__init__()
        self.active = []
        self.lock = threading.Lock()
    def makeActive(self, name):
        with self.lock:
            self.active.append(name)
            logging.debug('Running: %s', self.active)
    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)
            logging.debug('Running: %s', self.active)

def worker(s, pool):
    logging.debug('Waiting to join the pool')
    with s:
        name = threading.currentThread().getName()
        pool.makeActive(name)
        time.sleep(0.1)
        pool.makeInactive(name)

pool = ActivePool()
s = threading.Semaphore(2)
for i in range(4):
    t = threading.Thread(target=worker, name=str(i), args=(s, pool))
    t.start()
    
# In[]
    
## This function returns a list of all active threads. 


import threading


def Main():
	for thread in threading.enumerate():
	    print("Thread name is %s." % thread.getName())
	    

if __name__ == '__main__':
	Main()
    
# In[]
    
#Python multithreading example to demonstrate locking.
#1. Define a subclass using Thread class.
#2. Instantiate the subclass and trigger the thread. 
#3. Implement locks in thread's run method. 

import threading
import datetime

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, name, counter):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name
        self.counter = counter
    def run(self):
        print "Starting " + self.name
        # Acquire lock to synchronize thread
        threadLock.acquire()
        print_date(self.name, self.counter)
        # Release lock for the next thread
        threadLock.release()
        print "Exiting " + self.name

def print_date(threadName, counter):
    datefields = []
    today = datetime.date.today()
    datefields.append(today)
    print "%s[%d]: %s" % ( threadName, counter, datefields[0] )

threadLock = threading.Lock()
threads = []

# Create new threads
thread1 = myThread("Thread", 1)
thread2 = myThread("Thread", 2)

# Start new Threads
thread1.start()
thread2.start()

# Add threads to thread list
threads.append(thread1)
threads.append(thread2)

# Wait for all threads to complete
for t in threads:
    t.join()
print "Exiting the Program!!!"

# In[]

import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
                    
def lock_holder(lock):
    logging.debug('Starting')
    while True:
        lock.acquire()
        try:
            logging.debug('Holding')
            time.sleep(0.5)
        finally:
            logging.debug('Not holding')
            lock.release()
        time.sleep(0.5)
    return
                    
def worker(lock):
    logging.debug('Starting')
    num_tries = 0
    num_acquires = 0
    while num_acquires < 3:
        time.sleep(0.5)
        logging.debug('Trying to acquire')
        have_it = lock.acquire(0)
        try:
            num_tries += 1
            if have_it:
                logging.debug('Iteration %d: Acquired',  num_tries)
                num_acquires += 1
            else:
                logging.debug('Iteration %d: Not acquired', num_tries)
        finally:
            if have_it:
                lock.release()
    logging.debug('Done after %d iterations', num_tries)


lock = threading.Lock()

holder = threading.Thread(target=lock_holder, args=(lock,), name='LockHolder')
holder.setDaemon(True)
holder.start()

worker = threading.Thread(target=worker, args=(lock,), name='Worker')
worker.start()

# In[]

import threading
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def worker_with(lock):
    with lock:
        logging.debug('Lock acquired via with')
        
def worker_no_with(lock):
    lock.acquire()
    try:
        logging.debug('Lock acquired directly')
    finally:
        lock.release()

lock = threading.Lock()
w = threading.Thread(target=worker_with, args=(lock,))
nw = threading.Thread(target=worker_no_with, args=(lock,))

w.start()
nw.start()

# In[]

import os, re

received_packages = re.compile(r"(\d) received")
status = ("no response","alive but losses","alive")

for suffix in range(20,30):
   ip = "192.168.178."+str(suffix)
   ping_out = os.popen("ping -q -c2 "+ip,"r")
   print "... pinging ",ip
   while True:
      line = ping_out.readline()
      if not line: break
      n_received = received_packages.findall(line)
      if n_received:
         print ip + ": " + status[int(n_received[0])]
         
# In[]
        
import os, re, threading

class ip_check(threading.Thread):
   def __init__ (self,ip):
      threading.Thread.__init__(self)
      self.ip = ip
      self.__successful_pings = -1
   def run(self):
      ping_out = os.popen("ping -q -c2 "+self.ip,"r")
      while True:
        line = ping_out.readline()
        if not line: break
        n_received = re.findall(received_packages,line)
        if n_received:
           self.__successful_pings = int(n_received[0])
   def status(self):
      if self.__successful_pings == 0:
         return "no response"
      elif self.__successful_pings == 1:
         return "alive, but 50 % package loss"
      elif self.__successful_pings == 2:
         return "alive"
      else:
         return "shouldn't occur"
received_packages = re.compile(r"(\d) received")

check_results = []
for suffix in range(20,70):
   ip = "192.168.178."+str(suffix)
   current = ip_check(ip)
   check_results.append(current)
   current.start()

for el in check_results:
   el.join()
   print "Status from ", el.ip,"is",el.status()
   
# In[]
   
## This program creates a thread, 
## officially names it and 
## tries to print the name


import threading
import time


def ThreadFunction():
	print threading.currentThread().getName(), "Starting"
	time.sleep(2)
	print threading.currentThread().getName(), "Exiting"

def ServiceFunction():
	print threading.currentThread().getName(), "Starting"
	time.sleep(3)
	print threading.currentThread().getName(), "Exiting"


def Main():
	myThread = threading.Thread(
		name='Service Function', 
		target=ServiceFunction
	)
	w = threading.Thread(
		name='Thread function',
		target=ThreadFunction
	)
	w2 = threading.Thread(
		target=ThreadFunction
	)

	w.start()
	w2.start()
	myThread.start()


if __name__ == "__main__":
	Main()
    
# In[]
    
import threading

lock = threading.RLock()

print 'First try :', lock.acquire()
print 'Second try:', lock.acquire(0)

# In[]

import threading

lock = threading.Lock()

print 'First try :', lock.acquire()
print 'Second try:', lock.acquire(0)

# In[]

import threading
import logging
import time


logging.basicConfig(
	level=logging.DEBUG,
	format = '(%(threadName)-10s) %(message)s',
)


class MyThread(threading.Thread):
    def run(self):
        logging.debug('This thread is running')
        return

for x in range(5):
	z=MyThread()
	z.start()
	time.sleep(1)
    
# In[]
    
import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
                    
def wait_for_event(e):
    """Wait for the event to be set before doing anything"""
    logging.debug('wait_for_event starting')
    event_is_set = e.wait()
    logging.debug('event set: %s', event_is_set)

def wait_for_event_timeout(e, t):
    """Wait t seconds and then timeout"""
    while not e.isSet():
        logging.debug('wait_for_event_timeout starting')
        event_is_set = e.wait(t)
        logging.debug('event set: %s', event_is_set)
        if event_is_set:
            logging.debug('processing event')
        else:
            logging.debug('doing other work')


e = threading.Event()
t1 = threading.Thread(name='block', 
                      target=wait_for_event,
                      args=(e,))
t1.start()

t2 = threading.Thread(name='non-block', 
                      target=wait_for_event_timeout, 
                      args=(e, 2))
t2.start()

logging.debug('Waiting before calling Event.set()')
time.sleep(3)
e.set()
logging.debug('Event is set')

# In[]

# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
 
def print_cube(num):
    """
    function to print cube of given num
    """
    print("Cube: {}".format(num * num * num))
 
def print_square(num):
    """
    function to print square of given num
    """
    print("Square: {}".format(num * num))
 
if __name__ == "__main__":
    # creating thread
    t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=print_cube, args=(10,))
 
    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()
 
    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()
 
    # both threads completely executed
    print("Done!")
    
# In[]
    
## This multithreading program prints output 
## of square and cube of a series of numbers


import time
import threading


def PrintSquare(numbers):
  print("Print squares")
  for n in numbers:
    time.sleep(0.2)
    print("Square", n*n)


def PrintCube(numbers):
  print("Print cubes")
  for n in numbers:
    time.sleep(0.2)
    print("Cube", n*n*n)


def Main():
	arr = [2,3,4,5]
	t = time.time()
	PrintSquare(arr)
	PrintCube(arr)
	t1=threading.Thread(Target=PrintSquare, args=(arr,))
	t2=threading.Thread(Target=PrintCube, args=(arr,))
	t1.start()
	t2.start()
	t1.join()
	t2.join()
	print("Program took", time.time()-t)


if __name__ == '__main__':
	Main()
    
# In[]
    
## Above program has been written using multithreading module 
## and the following one has been written in conventional 
## way where threading module hasnot been used.
## A comparison between the above and folling program 
## would help the readers understand the difference 
## a normal program and a thread program.


import time


def PrintSquare(num):
  print("Print squares of the given numbers")
  for n in numbers:
    time.sleep(0.2)
    print("Square", n*n)


def PrintCube(num):
  print("Print cubes of the given numbers")
  for n in numbers:
    time.sleep(0.2)
    print("Cube", n*n*n)


arr = [2,3,4,5]
t = time.time()
PrintSquare(arr)
PrintCube(arr)
print("Program took", time.time()-t)

# In[]

import threading
import time 
# global variable x
x = 0
 
def Tfunc(i,lock):
 lock.acquire()
 print('from thread %d' %i)
 lock.release()
 
 
 # creating a lock
lock = threading.Lock()
 
 # creating threads
t1 = threading.Thread(target=Tfunc, args=(1,lock,))
t2 = threading.Thread(target=Tfunc, args=(2,lock,))
 
 # start threads
t1.start()
t2.start()
 
 # wait until threads finish their job
t1.join()
time.sleep(5)
t2.join()

# In[]

import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s (%(threadName)-2s) %(message)s',
                    )

def consumer(cond):
    """wait for the condition and use the resource"""
    logging.debug('Starting consumer thread')
    t = threading.currentThread()
    with cond:
        cond.wait()
        logging.debug('Resource is available to consumer')

def producer(cond):
    """set up the resource to be used by the consumer"""
    logging.debug('Starting producer thread')
    with cond:
        logging.debug('Making resource available')
        cond.notifyAll()

condition = threading.Condition()
c1 = threading.Thread(name='c1', target=consumer, args=(condition,))
c2 = threading.Thread(name='c2', target=consumer, args=(condition,))
p = threading.Thread(name='p', target=producer, args=(condition,))

c1.start()
time.sleep(2)
c2.start()
time.sleep(2)
p.start()

# In[]

import threading
import time 


tLock = threading.Lock()


def timer(name, delay, repeat):
  print ("Timer: " + name + " Started")
  
  tLock.acquire()
  print name + " has acquired lock"
  while repeat > 0:
    time.sleep(delay)
    print (name + ":" + str(time.ctime(time.time())))
    repeat -= 1
  print name + "is releasing the lock"
  tLock.release()
  print ("Timer: " + name + "Completed")

  
def Main():
  thread1 = threading.Thread(
    target=timer, 
    args=("Timer1", 1, 5)
  )
  thread2 = threading.Thread(
    target=timer, 
    args=("Timer2", 2, 5)
  )
  thread1.start()
  thread2.start()

  print ("Main function completed")


if __name__ == '__main__':
  Main()
  
# In[]
  import time
import threading

def get_food(c_id):
    print("Customer %s's food is cooking.... "%c_id)
    time.sleep(5)
    print("Customer %s's food is ready "%c_id)
    print("======Ending time: " + str(time.ctime()))

def make_order(c_id):
    print("Running Thread: " + str(threading.current_thread().name))
    print("Customer %s's food is Ordering.... "%c_id)
    get_food(c_id)


if __name__ == "__main__":

    cust = 5
    threads = []

    print("======Starting time: " + str(time.ctime()))

    for c in range(cust):
        thread = threading.Thread(target=make_order, args=(c,))
        thread.start()


    for t in threads:
        t.join()
# In[]
    
