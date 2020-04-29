# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 23:07:25 2020

@author: choi
"""

from multiprocessing import Process, Queue


import multiprocessing as mp

import random

def rand_num():
    num = random.random()
    print(num)

if __name__ == "__main__":
    queue = Queue()

    processes = [Process(target=rand_num, args=()) for x in range(4)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
        
# In[1]:
        
def my_func(x):
  print(x**x)

def main():
  pool = mp.Pool(mp.cpu_count())
  result = pool.map(my_func, [4,2,3])

if __name__ == "__main__":
  main()
  
  
# In[2]
  
  
def my_func(x):
    
    
  print(mp.current_process())
  return x**x

def main():
    
    
  pool = mp.Pool(mp.cpu_count())
  result = pool.map(my_func, [4,2,3,5,3,2,1,2])
  
  
  result_set_2 = pool.map(my_func, [4,6,5,4,6,3,23,4,6])

  print(result)
  
  print(result_set_2)

if __name__ == "__main__":
    
  main()
  
# In[3]
  
  def rand_num(queue):
      
      
    num = random.random()
    
    queue.put(num)

if __name__ == "__main__":
    
    
    queue = Queue()

    processes = [Process(target=rand_num, args=(queue,)) for x in range(4)]

    for p in processes:
        
        p.start()

    for p in processes:
        
        p.join()

    results = [queue.get() for p in processes]

    print(results)
  
  
# In[4]
    
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
    
    
    
    p1 = multiprocessing.Process(target=print_square, args=(10, )) 
    
    p2 = multiprocessing.Process(target=print_cube, args=(10, )) 
  
     
    p1.start() 
    
    p2.start() 
  
   
    p1.join() 
  
    p2.join() 
  
   
    print("Done!") 
  
  
# In[5]
  
def worker1(): 
   
    
    print("ID of process running worker1: {}".format(os.getpid())) 
  
def worker2(): 
   
    
    print("ID of process running worker2: {}".format(os.getpid())) 
  
if __name__ == "__main__": 
  
    
    print("ID of main process: {}".format(os.getpid())) 
  
   
    p1 = multiprocessing.Process(target=worker1) 
    
    p2 = multiprocessing.Process(target=worker2) 
  

    p1.start() 
    p2.start() 
  

    print("ID of process p1: {}".format(p1.pid)) 
    
    print("ID of process p2: {}".format(p2.pid)) 
  
 
    p1.join() 
    p2.join() 
  
    
    print("Both processes finished execution!") 
  
 
    print("Process p1 is alive: {}".format(p1.is_alive())) 
    
    print("Process p2 is alive: {}".format(p2.is_alive())) 
  
# In[6]
  
result = [] 
  
def square_list(mylist): 
    
    
    """ 
    function to square a given list 
    """
    global result 

    for num in mylist: 
        result.append(num * num) 
    
    print("Result(in process p1): {}".format(result)) 
  
if __name__ == "__main__": 
 
    
    mylist = [1,2,3,4] 
  

    p1 = multiprocessing.Process(target=square_list, args=(mylist,)) 
 
    
    p1.start() 
    
    
    p1.join() 
  
    
    print("Result(in main program): {}".format(result)) 
    
  
# In[7]
  
def square_list(mylist, result, square_sum): 
    
    
    """ 
    function to square a given list 
    """
    
    
    for idx, num in enumerate(mylist): 
        
        result[idx] = num * num 
  
  
    square_sum.value = sum(result) 
  
   
    print("Result(in process p1): {}".format(result[:])) 
  
     
    print("Sum of squares(in process p1): {}".format(square_sum.value)) 
  
if __name__ == "__main__": 
    
   
    mylist = [1,2,3,4] 
  
     
    result = multiprocessing.Array('i', 4) 
  
  
    square_sum = multiprocessing.Value('i') 
  
    
    p1 = multiprocessing.Process(target=square_list, args=(mylist, result, square_sum)) 
  
    
    p1.start() 
  
   
    p1.join() 
  
    
    print("Result(in main program): {}".format(result[:])) 
  
    
    print("Sum of squares(in main program): {}".format(square_sum.value)) 
  
# In[8]
  
def print_records(records): 
    
    
    """ 
    function to print record(tuples) in records(list) 
    """
    for record in records: 
        
        
        print("Name: {0}\nScore: {1}\n".format(record[0], record[1])) 
  
def insert_record(record, records): 
    
    
    """ 
    function to add a new record to records(list) 
    """
    records.append(record) 
    
    
    print("New record added!\n") 
  
if __name__ == '__main__': 
    
    
    with multiprocessing.Manager() as manager: 
        
        
     
        
        records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin',9)]) 
        
        
        
       
        new_record = ('Jeff', 8) 
  
        
        p1 = multiprocessing.Process(target=insert_record, args=(new_record, records)) 
        p2 = multiprocessing.Process(target=print_records, args=(records,)) 
  
        
        
        p1.start() 
        p1.join() 
  
        
        
        p2.start() 
        p2.join() 
  
  # In[9]
  
  
def square_list(mylist, q): 
    
    """ 
    function to square a given list 
    """
   
    
    for num in mylist: 
        q.put(num * num) 
  
def print_queue(q): 
    
    
    """ 
    function to print queue elements 
    """
    print("Queue elements:") 
    while not q.empty(): 
        
        
        print(q.get()) 
    print("Queue is now empty!") 
  
if __name__ == "__main__": 
    
    
  
    mylist = [1,2,3,4] 
  
     
    q = multiprocessing.Queue() 
  
   
    p1 = multiprocessing.Process(target=square_list, args=(mylist, q)) 
    
    p2 = multiprocessing.Process(target=print_queue, args=(q,)) 
  
   
    
    p1.start() 
    
    p1.join() 
  
    
    p2.start() 
    
    p2.join()
    
    
    
  
# In[10]
    
    
def sender(conn, msgs): 
    
    
    """ 
    function to send messages to other end of pipe 
    """
    for msg in msgs: 
        
        
        conn.send(msg) 
        print("Sent the message: {}".format(msg)) 
        
    conn.close() 
  
def receiver(conn): 
    
    
    """ 
    function to print the messages received from other 
    end of pipe 
    """
    while 1: 
        
        
        msg = conn.recv() 
        if msg == "END": 
            
            
            break
        print("Received the message: {}".format(msg)) 
        
  
if __name__ == "__main__": 
    
    
    
    msgs = ["hello", "hey", "hru?", "END"] 
  
   
    
    parent_conn, child_conn = multiprocessing.Pipe() 
  
    
    p1 = multiprocessing.Process(target=sender, args=(parent_conn,msgs)) 
    
    p2 = multiprocessing.Process(target=receiver, args=(child_conn,)) 
  
    
    
    p1.start() 
    
    p2.start() 
  
    
    p1.join() 
    
    p2.join() 
    
    
# In[]
    
    
    def withdraw(balance):    
        
    for _ in range(10000): 
        
        balance.value = balance.value - 1
  

def deposit(balance):    
    
    
    for _ in range(10000): 
        
        
        balance.value = balance.value + 1
  
def perform_transactions(): 
  
    
    
    balance = multiprocessing.Value('i', 100) 
  
   
    
    
    p1 = multiprocessing.Process(target=withdraw, args=(balance,)) 
    p2 = multiprocessing.Process(target=deposit, args=(balance,)) 
  
    
    
    p1.start() 
    p2.start() 
  
   
    
    p1.join() 
    p2.join() 
  
  
    
    print("Final balance = {}".format(balance.value)) 
  
if __name__ == "__main__": 
    for _ in range(10): 
  
       
        
        
        perform_transactions() 
    
# In[]
        
        
        def withdraw(balance, lock): 
            
            
    for _ in range(10000): 
        
        lock.acquire() 
        
        balance.value = balance.value - 1
        
        lock.release() 
  


def deposit(balance, lock): 

    
    for _ in range(10000): 
        
        lock.acquire() 
        
        balance.value = balance.value + 1
        
        lock.release() 
  
def perform_transactions(): 
    
  
    
    
    
    balance = multiprocessing.Value('i', 100) 
  
    
    
    
    lock = multiprocessing.Lock() 
  
  
    
    
    p1 = multiprocessing.Process(target=withdraw, args=(balance,lock)) 
    
    
    p2 = multiprocessing.Process(target=deposit, args=(balance,lock)) 
  
    
    
    p1.start() 
    
    p2.start() 
  
   
    
    p1.join() 
    
    p2.join() 
  
 
    
    
    print("Final balance = {}".format(balance.value)) 
  
if __name__ == "__main__": 
    
    for _ in range(10): 
        
  
      
        
        perform_transactions() 
    
# In[]
        
        def square(n): 
            
            
    return (n*n) 
  
if __name__ == "__main__": 
  
    
    
    mylist = [1,2,3,4,5] 
  
    
    
    result = [] 
  
    for num in mylist: 
        
        
        result.append(square(num)) 
  
    print(result) 
    
# In[]
    
    def square(n): 
        
        
    print("Worker process id for {0}: {1}".format(n, os.getpid())) 
    
    return (n*n) 
  
if __name__ == "__main__": 
    
    
    
    mylist = [1,2,3,4,5] 
  
   
    
    p = multiprocessing.Pool() 
  

 
    result = p.map(square, mylist) 
  
    print(result) 
    
# In[]
    def worker():
    """worker function"""
    print 'Worker'
    return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()
# In[]
        
def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()
    
# In[]
        if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=multiprocessing_import_worker.worker)
        jobs.append(p)
        p.start()
        
        def worker():
    """worker function"""
    print 'Worker'
    return
    
# In[]
    def worker():
    name = multiprocessing.current_process().name
    print name, 'Starting'
    time.sleep(2)
    print name, 'Exiting'

def my_service():
    name = multiprocessing.current_process().name
    print name, 'Starting'
    time.sleep(3)
    print name, 'Exiting'

if __name__ == '__main__':
    service = multiprocessing.Process(name='my_service', target=my_service)
    worker_1 = multiprocessing.Process(name='worker 1', target=worker)
    worker_2 = multiprocessing.Process(target=worker) # use default name

    worker_1.start()
    worker_2.start()
    service.start()
# In[]
    def daemon():
    p = multiprocessing.current_process()
    print 'Starting:', p.name, p.pid
    sys.stdout.flush()
    time.sleep(2)
    print 'Exiting :', p.name, p.pid
    sys.stdout.flush()

def non_daemon():
    p = multiprocessing.current_process()
    print 'Starting:', p.name, p.pid
    sys.stdout.flush()
    print 'Exiting :', p.name, p.pid
    sys.stdout.flush()

if __name__ == '__main__':
    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    time.sleep(1)
    n.start()
    
# In[]
    def daemon():
    print 'Starting:', multiprocessing.current_process().name
    time.sleep(2)
    print 'Exiting :', multiprocessing.current_process().name

def non_daemon():
    print 'Starting:', multiprocessing.current_process().name
    print 'Exiting :', multiprocessing.current_process().name

if __name__ == '__main__':
    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    time.sleep(1)
    n.start()

    d.join()
    n.join()
# In[]
    
    def daemon():
    print 'Starting:', multiprocessing.current_process().name
    time.sleep(2)
    print 'Exiting :', multiprocessing.current_process().name

def non_daemon():
    print 'Starting:', multiprocessing.current_process().name
    print 'Exiting :', multiprocessing.current_process().name

if __name__ == '__main__':
    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    n.start()

    d.join(1)
    print 'd.is_alive()', d.is_alive()
    n.join()
    
# In[]
    def slow_worker():
    print 'Starting worker'
    time.sleep(0.1)
    print 'Finished worker'

if __name__ == '__main__':
    p = multiprocessing.Process(target=slow_worker)
    print 'BEFORE:', p, p.is_alive()
    
    p.start()
    print 'DURING:', p, p.is_alive()
    
    p.terminate()
    print 'TERMINATED:', p, p.is_alive()

    p.join()
    print 'JOINED:', p, p.is_alive()
    
# In[]
    def exit_error():
    sys.exit(1)

def exit_ok():
    return

def return_value():
    return 1

def raises():
    raise RuntimeError('There was an error!')

def terminated():
    time.sleep(3)

if __name__ == '__main__':
    jobs = []
    for f in [exit_error, exit_ok, return_value, raises, terminated]:
        print 'Starting process for', f.func_name
        j = multiprocessing.Process(target=f, name=f.func_name)
        jobs.append(j)
        j.start()
        
    jobs[-1].terminate()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)
    
# In[]
    def worker():
    print 'Doing some work'
    sys.stdout.flush()

if __name__ == '__main__':
    multiprocessing.log_to_stderr(logging.DEBUG)
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
    # In[]
    
    def worker():
    print 'Doing some work'
    sys.stdout.flush()

if __name__ == '__main__':
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
    
    # In[]
    
    class Worker(multiprocessing.Process):

    def run(self):
        print 'In %s' % self.name
        return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = Worker()
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    
    # In[]
    
    data = (['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
        ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7']
)


def mp_handler(var1):
    for indata in var1:
        p = multiprocessing.Process(target=mp_worker, args=(indata[0], indata[1]))
        p.start()


def mp_worker(inputs, the_time):
    print " Processs %s\tWaiting %s seconds" % (inputs, the_time)
    time.sleep(int(the_time))
    print " Process %s\tDONE" % inputs

if __name__ == '__main__':
    mp_handler(data)
# In[]
    data = (
    ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
    ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7']
)

def mp_worker((inputs, the_time)):
    print " Processs %s\tWaiting %s seconds" % (inputs, the_time)
    time.sleep(int(the_time))
    print " Process %s\tDONE" % inputs

def mp_handler():
    p = multiprocessing.Pool(2)
    p.map(mp_worker, data)

if __name__ == '__main__':
    mp_handler()
# In[]
    class Renderer:
    queue = None

    def __init__(self, nb_workers=2):
        self.queue = JoinableQueue()
        self.processes = [Process(target=self.upload) for i in range(nb_workers)]
        for p in self.processes:
            p.start()

    def render(self, item):
        self.queue.put(item)

    def upload(self):
        while True:
            item = self.queue.get()
            if item is None:
                break

            # process your item here

            self.queue.task_done()

    def terminate(self):
        """ wait until queue is empty and terminate processes """
        self.queue.join()
        for p in self.processes:
            p.terminate()

r = Renderer()
r.render(item1)
r.render(item2)
r.terminate()
# In[]
    THREADS = 3

# Used to prevent multiple threads from mixing thier output
GLOBALLOCK = multiprocessing.Lock()


def func_worker(args):
    """This function will be called by each thread.
    This function can not be a class method.
    """
    # Expand list of args into named args.
    str1, str2 = args
    del args

    # Work
    # ...



    # Serial-only Portion
    GLOBALLOCK.acquire()
    print(str1)
    print(str2)
    GLOBALLOCK.release()


def main(argp=None):
    """Multiprocessing Spawn Example
    """
    # Create the number of threads you want
    pool = multiprocessing.Pool(THREADS)

    # Define two jobs, each with two args.
    func_args = [
        ('Hello', 'World',), 
        ('Goodbye', 'World',), 
    ]


    try:
        # Spawn up to 9999999 jobs, I think this is the maximum possible.
        # I do not know what happens if you exceed this.
        pool.map_async(func_worker, func_args).get(9999999)
    except KeyboardInterrupt:
        # Allow ^C to interrupt from any thread.
        sys.stdout.write('\033[0m')
        sys.stdout.write('User Interupt\n')
    pool.close()

if __name__ == '__main__':
    main()
# In[]
    
# In[]
    
# In[]