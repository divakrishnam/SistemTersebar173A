# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:28:02 2020

@author: choi
"""

import logging

import concurrent.futures

import threading

import random

import time

import Queue



def thread_function(name):
    
    logging.info("Thread %s: starting", name)
    
    
    time.sleep(2)
    
    
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    
    
    format = "%(asctime)s: %(message)s"
    
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,))
    
    
    logging.info("Main    : before running thread")
    x.start()
    
    logging.info("Main    : wait for the thread to finish")
    # x.join()
    
    logging.info("Main    : all done")
    
    
    
# In[2]:
def thread_function(name):
    
    logging.info("Thread %s: starting", name)
    
    time.sleep(2)
    
    logging.info("Thread %s: finishing", name)
    

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    

    threads = list()
    
    for index in range(3):
        
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        
        logging.info("Main    : before joining thread %d.", index)
        
        thread.join()
        
        
        logging.info("Main    : thread %d done", index)
        

# In[3]:
        
# [rest of code]

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        
        executor.map(thread_function, range(3))

# In[4]
        
class FakeDatabase:
    
    
    def __init__(self):
        
        self.value = 0

    def update(self, name):
        logging.info("Thread %s: starting update", name)
        local_copy = self.value
        local_copy += 1
        
        
        time.sleep(0.1)
        self.value = local_copy
        logging.info("Thread %s: finishing update", name)
        
        if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    
    
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    database = FakeDatabase()
    
    logging.info("Testing update. Starting value is %d.", database.value)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            
            executor.submit(database.update, index)
            
    logging.info("Testing update. Ending value is %d.", database.value)
    
# In[5]

class FakeDatabase:
    
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()


    def locked_update(self, name):
        logging.info("Thread %s: starting update", name)
        
        logging.debug("Thread %s about to lock", name)
        
        
        with self._lock:
            
            logging.debug("Thread %s has lock", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.debug("Thread %s about to release lock", name)
            
            
        logging.debug("Thread %s after release", name)
        
        logging.info("Thread %s: finishing update", name)
        
        logging.getLogger().setLevel(logging.DEBUG)
        
# In[6]
        
l = threading.Lock()
print("before first acquire")


l.acquire()
print("before second acquire")


l.acquire()
print("acquired lock twice")

# In[7]

SENTINEL = object()

def producer(pipeline):
    """Pretend we're getting a message from the network."""
    for index in range(10):
        message = random.randint(1, 101)
        
        logging.info("Producer got message: %s", message)
        pipeline.set_message(message, "Producer")

    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")
    
    
    def consumer(pipeline):
    """Pretend we're saving a number in the database."""
    message = 0
    while message is not SENTINEL:
        message = pipeline.get_message("Consumer")
        
        if message is not SENTINEL:
            logging.info("Consumer storing message: %s", message)
            
            
            if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        
                        datefmt="%H:%M:%S")
    # logging.getLogger().setLevel(logging.DEBUG)

    pipeline = Pipeline()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        
        executor.submit(producer, pipeline)
        executor.submit(consumer, pipeline)
        
# In[8]
        
class Pipeline:
    
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        
        
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()
        

    def get_message(self, name):
        logging.debug("%s:about to acquire getlock", name)
        self.consumer_lock.acquire()
        
        logging.debug("%s:have getlock", name)
        message = self.message
        
        logging.debug("%s:about to release setlock", name)
        self.producer_lock.release()
        
        logging.debug("%s:setlock released", name)
        return message


    def set_message(self, message, name):
        
        logging.debug("%s:about to acquire setlock", name)
        self.producer_lock.acquire()
        
        logging.debug("%s:have setlock", name)
        self.message = message
        
        logging.debug("%s:about to release getlock", name)
        self.consumer_lock.release()
        
        
        logging.debug("%s:getlock released", name)
        
# In[9]
class Pipeline:
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self, name):
        
        self.consumer_lock.acquire()
        message = self.message
        
        self.producer_lock.release()
        return message

    def set_message(self, message, name):
        
        self.producer_lock.acquire()
        self.message = message
        
        self.consumer_lock.release()
        
# In[10]
        
def consumer(pipeline, event):
    
    """Pretend we're saving a number in the database."""
    while not event.is_set() or not pipeline.empty():
        
        message = pipeline.get_message("Consumer")
        logging.info(
                
            "Consumer storing message: %s  (queue size=%s)",
            message,
            pipeline.qsize(),
        )
        
        
    logging.info("Consumer received EXIT event. Exiting")
    
    
# In[11]
    
class Pipeline(queue.Queue):
    
    def __init__(self):
        super().__init__(maxsize=10)

    def get_message(self, name):
        
        
        logging.debug("%s:about to get from queue", name)
        value = self.get()
        
        logging.debug("%s:got %d from queue", name, value)
        return value

    def set_message(self, value, name):
        
        
        logging.debug("%s:about to add %d to queue", name, value)
        self.put(value)
        
        logging.debug("%s:added %d to queue", name, value)
        
# In[12]
        
def producer(queue, event):
    
    
    """Pretend we're getting a number from the network."""
    while not event.is_set():
        
        
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        queue.put(message)

    logging.info("Producer received event. Exiting")

def consumer(queue, event):
    
    
    """Pretend we're saving a number in the database."""
    while not event.is_set() or not queue.empty():
        message = queue.get()
        
        
        logging.info(
            "Consumer storing message: %s (size=%d)", message, queue.qsize()
        )

    logging.info("Consumer received event. Exiting")

if __name__ == "__main__":
    
    
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")


    pipeline = queue.Queue(maxsize=10)
    event = threading.Event()
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline, event)
        executor.submit(consumer, pipeline, event)


        time.sleep(0.1)
        logging.info("Main: about to set event")
        event.set()
        
# In[13]
        
class myThread (threading.Thread):
    
    
   def __init__(self, threadID, name, q):
      threading.Thread.__init__(self)
      
      self.threadID = threadID
      self.name = name
      
      self.q = q
   def run(self):
       
       
      print "Starting " + self.name
      process_data(self.name, self.q)
      
      
      print "Exiting " + self.name

def process_data(threadName, q):
    
    
   while not exitFlag:
       
      queueLock.acquire()
      
         if not workQueue.empty():
             
             
            data = q.get()
            queueLock.release()
            print "%s processing %s" % (threadName, data)
            
            
         else:
            queueLock.release()
         time.sleep(1)


threadList = ["Thread-1", "Thread-2", "Thread-3"]
nameList = ["One", "Two", "Three", "Four", "Five"]


queueLock = threading.Lock()
workQueue = Queue.Queue(10)


threads = []
threadID = 1


for tName in threadList:
    
   thread = myThread(threadID, tName, workQueue)
   thread.start()
   
   
   threads.append(thread)
   threadID += 1


queueLock.acquire()

for word in nameList:
    
   workQueue.put(word)
queueLock.release()


while not workQueue.empty():
    
   pass


exitFlag = 1


for t in threads:
    
   t.join()
print "Exiting Main Thread"



# In[14]


class myThread (threading.Thread):
    
    
   def __init__(self, threadID, name, counter):
       
       
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
      
      
   def run(self):
       
       
      print "Starting " + self.name
   
    
      threadLock.acquire()
      print_time(self.name, self.counter, 3)
    
      threadLock.release()

def print_time(threadName, delay, counter):
    
   while counter:
       
       
      time.sleep(delay)
      print "%s: %s" % (threadName, time.ctime(time.time()))
      counter -= 1



threadLock = threading.Lock()
threads = []


thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)


thread1.start()
thread2.start()


threads.append(thread1)
threads.append(thread2)


for t in threads:
    
    t.join()
    
print "Exiting Main Thread"


# In[15]


exitFlag = 0

class myThread (threading.Thread):
    
    
   def __init__(self, threadID, name, counter):
       
       
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      
      
      self.counter = counter
      
   def run(self):
       
       
      print "Starting " + self.name
      print_time(self.name, 5, self.counter)
      print "Exiting " + self.name



def print_time(threadName, counter, delay):
    
    
   while counter:
       
      if exitFlag:
          
         threadName.exit()
      time.sleep(delay)
      print "%s: %s" % (threadName, time.ctime(time.time()))
      counter -= 1


thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)


thread1.start()
thread2.start()

print "Exiting Main Thread"

# In[16]
class mythread(threading.Thread):
  def __init__(self, i):
  threading.Thread.__init__(self)
self.h = i
def run(self):
  print“ Value send“, self.h
thread1 = mythread(1)
thread1.start()
# In[16]
def fun1(a, b):
  c = a + b
print(c)
thread1 = threading.Thread(target = fun1, args = (12, 10))
thread1.start()
# In[16]
def fun1(a, b):
  time.sleep(1)
c = a + b
print(c)
thread1 = threading.Thread(target = fun1, args = (12, 10))
thread1.start()
thread2 = threading.Thread(target = fun1, args = (10, 17))
thread2.start()
print(“Total number of threads”, threading.activeCount())
print(“List of threads: “, threading.enumerate())
# In[16]
list1 = []
def fun1(a):
  time.sleep(1)# complex calculation takes 1 seconds
list1.append(a)
thread1 = threading.Thread(target = fun1, args = (1, ))
thread1.start()
thread2 = threading.Thread(target = fun1, args = (6, ))
thread2.start()
print(“List1 is: “, list1)
# In[16]
list1 = []
def fun1(a):
  time.sleep(1)# complex calculation takes 1 seconds
list1.append(a)
thread1 = threading.Thread(target = fun1, args = (1, ))
thread1.start()
thread2 = threading.Thread(target = fun1, args = (6, ))
thread2.start()
thread1.join()
thread2.join()
print(“List1 is: “, list1)
# In[16]
t1 = datetime.datetime.now()
list1 = []
def fun1(a):
  time.sleep(1)# complex calculation takes 1 seconds
list1.append(a)
list_thread = []
for each in range(10):
  thread1 = threading.Thread(target = fun1, args = (each, ))
list_thread.append(thread1)
thread1.start()
for th in list_thread:
  th.join()
print(“List1 is: “, list1)
t2 = datetime.datetime.now()
print(“Time taken”, t2 - t1)
# In[16]

t1 = datetime.datetime.now()
list1 = []
def fun1(a):
  time.sleep(1)# complex calculation takes 1 seconds
list1.append(a)
for each in range(10):
  thread1 = threading.Thread(target = fun1, args = (each, ))
thread1.start()
thread1.join()
print(“List1 is: “, list1)
t2 = datetime.datetime.now()
print(“Time taken”, t2 - t1)

# In[16]
def fun1(a):
  time.sleep(3)# complex calculation takes 3 seconds
thread1 = threading.Thread(target = fun1, args = (1, ))
thread1.start()
thread1.join()
print(thread1.isAlive())
# In[16]
def auto(self, start_message, end_message):
        """
        Auto progress.
        """
        self._auto_running = threading.Event()
        self._auto_thread = threading.Thread(target=self._spin)

        self.start(start_message)
        self._auto_thread.start()

        try:
            yield self
        except (Exception, KeyboardInterrupt):
            self._io.error_line("")

            self._auto_running.set()
            self._auto_thread.join()

            raise

        self.finish(end_message, reset_indicator=True) 
# In[16]
def CheckConCursor(self):
        def run(con, errors):
            try:
                cur = con.cursor()
                errors.append("did not raise ProgrammingError")
                return
            except sqlite.ProgrammingError:
                return
            except:
                errors.append("raised wrong exception")

        errors = []
        t = threading.Thread(target=run, kwargs={"con": self.con, "errors": errors})
        t.start()
        t.join()
        if len(errors) > 0:
            self.fail("\n".join(errors)) 
# In[16]
            def CheckConCommit(self):
        def run(con, errors):
            try:
                con.commit()
                errors.append("did not raise ProgrammingError")
                return
            except sqlite.ProgrammingError:
                return
            except:
                errors.append("raised wrong exception")

        errors = []
        t = threading.Thread(target=run, kwargs={"con": self.con, "errors": errors})
        t.start()
        t.join()
        if len(errors) > 0:
            self.fail("\n".join(errors)) 
# In[16]
def CheckConRollback(self):
        def run(con, errors):
            try:
                con.rollback()
                errors.append("did not raise ProgrammingError")
                return
            except sqlite.ProgrammingError:
                return
            except:
                errors.append("raised wrong exception")

        errors = []
        t = threading.Thread(target=run, kwargs={"con": self.con, "errors": errors})
        t.start()
        t.join()
        if len(errors) > 0:
            self.fail("\n".join(errors)) 
# In[16]
def CheckCurImplicitBegin(self):
        def run(cur, errors):
            try:
                cur.execute("insert into test(name) values ('a')")
                errors.append("did not raise ProgrammingError")
                return
            except sqlite.ProgrammingError:
                return
            except:
                errors.append("raised wrong exception")

        errors = []
        t = threading.Thread(target=run, kwargs={"cur": self.cur, "errors": errors})
        t.start()
        t.join()
        if len(errors) > 0:
            self.fail("\n".join(errors)) 
# In[16]
            def CheckCurClose(self):
        def run(cur, errors):
            try:
                cur.close()
                errors.append("did not raise ProgrammingError")
                return
            except sqlite.ProgrammingError:
                return
            except:
                errors.append("raised wrong exception")

        errors = []
        t = threading.Thread(target=run, kwargs={"cur": self.cur, "errors": errors})
        t.start()
        t.join()
        if len(errors) > 0:
            self.fail("\n".join(errors)) 
            
# In[]
            
def CheckCurIterNext(self):
        def run(cur, errors):
            try:
                row = cur.fetchone()
                errors.append("did not raise ProgrammingError")
                return
            except sqlite.ProgrammingError:
                return
            except:
                errors.append("raised wrong exception")

        errors = []
        self.cur.execute("insert into test(name) values ('a')")
        self.cur.execute("select name from test")
        t = threading.Thread(target=run, kwargs={"cur": self.cur, "errors": errors})
        t.start()
        t.join()
        if len(errors) > 0:
            self.fail("\n".join(errors)) 

# In[]
def test_threads_write(self):
        # Issue6750: concurrent writes could duplicate data
        event = threading.Event()
        with self.open(support.TESTFN, "w", buffering=1) as f:
            def run(n):
                text = "Thread%03d\n" % n
                event.wait()
                f.write(text)
            threads = [threading.Thread(target=lambda n=x: run(n))
                       for x in range(20)]
            for t in threads:
                t.start()
            time.sleep(0.02)
            event.set()
            for t in threads:
                t.join()
        with self.open(support.TESTFN) as f:
            content = f.read()
            for n in range(20):
                self.assertEqual(content.count("Thread%03d\n" % n), 1)          
# In[]
def test_join_nondaemon_on_shutdown(self):
        # Issue 1722344
        # Raising SystemExit skipped threading._shutdown
        p = subprocess.Popen([sys.executable, "-c", """if 1:
                import threading
                from time import sleep

                def child():
                    sleep(1)
                    # As a non-daemon thread we SHOULD wake up and nothing
                    # should be torn down yet
                    print "Woke up, sleep function is:", sleep

                threading.Thread(target=child).start()
                raise SystemExit
            """],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        stdout, stderr = p.communicate()
        self.assertEqual(stdout.strip(),
            "Woke up, sleep function is: <built-in function sleep>")
        stderr = re.sub(r"^\[\d+ refs\]", "", stderr, re.MULTILINE).strip()
        self.assertEqual(stderr, "") 
# In[]
def test_is_alive_after_fork(self):
        # Try hard to trigger #18418: is_alive() could sometimes be True on
        # threads that vanished after a fork.
        old_interval = sys.getcheckinterval()

        # Make the bug more likely to manifest.
        sys.setcheckinterval(10)

        try:
            for i in range(20):
                t = threading.Thread(target=lambda: None)
                t.start()
                pid = os.fork()
                if pid == 0:
                    os._exit(1 if t.is_alive() else 0)
                else:
                    t.join()
                    pid, status = os.waitpid(pid, 0)
                    self.assertEqual(0, status)
        finally:
            sys.setcheckinterval(old_interval) 
# In[]
        
        def test_BoundedSemaphore_limit(self):
        # BoundedSemaphore should raise ValueError if released too often.
        for limit in range(1, 10):
            bs = threading.BoundedSemaphore(limit)
            threads = [threading.Thread(target=bs.acquire)
                       for _ in range(limit)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            threads = [threading.Thread(target=bs.release)
                       for _ in range(limit)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertRaises(ValueError, bs.release) 
# In[]
            
            def test_reinit_tls_after_fork(self):
        # Issue #13817: fork() would deadlock in a multithreaded program with
        # the ad-hoc TLS implementation.

        def do_fork_and_wait():
            # just fork a child process and wait it
            pid = os.fork()
            if pid > 0:
                os.waitpid(pid, 0)
            else:
                os._exit(0)

        # start a bunch of threads that will fork() child processes
        threads = []
        for i in range(16):
            t = threading.Thread(target=do_fork_and_wait)
            threads.append(t)
            t.start()

        for t in threads:
            t.join() 