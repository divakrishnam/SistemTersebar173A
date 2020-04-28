# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:08:25 2020

@author: Damara
"""
# In 1
import time 

start = time.perf_counter()


def do_something():
    print('Sleeping 1 second....')
    time.sleep(1)
    print('Done sleeping...')
    
    
do_something()


finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 2

import time 

start = time.perf_counter()



def do_something():
    print('Sleeping 1 second....')
    time.sleep(1)
    print('Done sleeping...')
    
    
do_something()
do_something()



finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 3 
# threading

import threading
import time 

start = time.perf_counter()



def do_something():
    print('Sleeping 1 second....')
    time.sleep(1)
    print('Done sleeping...')
    
    
t1 = threading.Thread(target=do_something)
t2 = threading.Thread(target=do_something)

t1.start()
t2.start()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 4 
import threading
import time 

start = time.perf_counter()



def do_something():
    print('Sleeping 1 second....')
    time.sleep(1)
    print('Done sleeping...')
    
    
t1 = threading.Thread(target=do_something)
t2 = threading.Thread(target=do_something)

t1.start()
t2.start()


t1.join()
t2.join()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 5

import threading
import time 

start = time.perf_counter()



def do_something():
    print('Sleeping 1 second....')
    time.sleep(1)
    print('Done sleeping...')
    
threads = []
    
    
for _ in range(10):
    t = threading.Thread(target=do_something)
    t.start()
    threads.append(t)
    
for thread in threads :
    thread.join()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 6 
import threading
import time 

start = time.perf_counter()



def do_something(seconds):
    print(f'Sleeping {seconds} second(s)....')
    time.sleep(seconds)
    print('Done sleeping...')
    
threads = []
    
    
for _ in range(10):
    t = threading.Thread(target=do_something, args=[1.8])
    t.start()
    threads.append(t)
    
for thread in threads :
    thread.join()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 7
import concurrent.futures
import time 

start = time.perf_counter()



def do_something(seconds):
    print(f'Sleeping {seconds} second(s)....')
    time.sleep(seconds)
    return 'Done sleeping...'
    
    
with concurrent.futures.ThreadPoolExecutor() as executor:
    f1 = executor.submit(do_something, 1)
    f2 = executor.submit(do_something, 1)
    print(f1.result())
    print(f2.result())
    
#threads = []
    
    
#for _ in range(10):
   # t = threading.Thread(target=do_something, args=[1.8])
   # t.start()
   # threads.append(t)
    
#for thread in threads :
    #thread.join()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 8
import concurrent.futures
import time 

start = time.perf_counter()



def do_something(seconds):
    print(f'Sleeping {seconds} second(s)....')
    time.sleep(seconds)
    return f'Done sleeping...{seconds}'
    
    
with concurrent.futures.ThreadPoolExecutor() as executor:
    secs = [5,4,3,2,1]
    results = [executor.submit(do_something, sec) for sec in secs]
    
    for f in concurrent.futures.as_completed(results):
        print(f.result())
    

    
#threads = []
    
    
#for _ in range(10):
   # t = threading.Thread(target=do_something, args=[1.8])
   # t.start()
   # threads.append(t)
    
#for thread in threads :
    #thread.join()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 8 

import concurrent.futures
import time 

start = time.perf_counter()



def do_something(seconds):
    print(f'Sleeping {seconds} second(s)....')
    time.sleep(seconds)
    return f'Done sleeping...{seconds}'
    
    
with concurrent.futures.ThreadPoolExecutor() as executor:
    secs = [5,4,3,2,1]
    results = executor.map(do_something, secs)
    
    
    #for result in results:
    #   print (result)
    
    
    
#threads = []
    
    
#for _ in range(10):
   # t = threading.Thread(target=do_something, args=[1.8])
   # t.start()
   # threads.append(t)
    
#for thread in threads :
    #thread.join()

finish = time.perf_counter()


print(f'Finished in {round(finish-start, 2)} second(s)')

# In 9
# Menunjukkan beberapa thread yang akan menampilkan nama pada interval yang berbeda.
import threading
import random
import time

class PrintThread(threading.Thread):
    """Subclass dari threading.Thread"""
    
    def __init__(self, threadName):
        """Inisialisasi thread, set sleep time, print data"""
        
        threading.Thread.__init__(self, name=threadName)
        self.sleepTime = random.randrange(1, 6)
        print 
            (self.getName(), self.sleepTime)
            
#overridden Thread run method
def run(self):
        """Slee untuk 1-5 detik"""
        print self.getName(), "going to sleep"
        time.sleep(self.sleepTime)
        print self.getName(), "done sleeping"
thread1 = PrintThread("thread1")
thread2 = PrintThread("thread2")
thread3 = PrintThread("thread3")
thread4 = PrintThread("thread4")

print "\nStarting thread"
thread1.start() #menjalankan thread
thread2.start() #menjalankan thread
thread3.start() #menjalankan thread
thread4.start() #menjalankan thread
print "Thread started\n"

#  In 10 
# menunjukkan multiple thread mengakses shared object

from UnsynchronizedInteger import UnsynchronizedInteger
from ProduceInteger import ProduceInteger
from ConsumeInteger import ConsumeInteger

# initialize integer and threads
number = UnsynchronizedInteger()
producer = ProduceInteger( "Producer", number )
consumer = ConsumeInteger( "Consumer", number )
print "Starting threads...\n"

# start threads
producer.start()
consumer.start()

# wait for threads to terminate
producer.join()
consumer.join()

print "\nAll threads have terminated."

# In 11

import threading 
import time

def func():
    print('ran')
    time.sleep(1)
    print ("done")
    
    
x = threading.Thread(target=func)
x.start()
print (threading.activeCount())

# In 12 

import threading 
import time

def func():
    print('ran')
    time.sleep(1)
    print ("done")
    time.sleep(1)
    print ("now done")
    
    
x = threading.Thread(target=func)
x.start()
print (threading.activeCount())
time.sleep(1)
print ("finally")
    
# In 13

import threading 
import time

def func():
    print('ran')
    time.sleep(1)
    print ("done")
    time.sleep(0.85)
    print ("now done")
    
x = threading.Thread(target=func)
x.start()
print (threading.activeCount())
time.sleep(0.9)
print ("finally")

# In 14

import threading 
import time

def count(n):
    for i in range (1,n+1):
        print(i)
        time.sleep(0.01)
        
        
for _ in range(2):
    x = threading.Thread(target= count, args = (10,))
    x.start()
    
    
print("Done")

# In 15 
import threading 
import time

def count(n):
    for i in range (1,n+1):
        print(i)
        time.sleep(0.01)
        
def count(n):
    for i in range (1,n+1):
        print(i)
        time.sleep(0.02)
        

x = threading.Thread(target= count, args = (10,))
x.start()

y = threading.Thread(target= count, args = (10,))
y.start()

print("Done")

# In 16
import threading 
import time

ls = []

def count(n):
    for i in range (1,n+1):
        ls.append(i)
        time.sleep(0.5)
        
def count(n):
    for i in range (1,n+1):
        ls.append(i)
        time.sleep(0.5)

x = threading.Thread(target= count, args = (5,))
x.start()

y = threading.Thread(target= count, args = (5,))
y.start()


print(ls)

# In 17
import threading 
import time

ls = []

def count(n):
    for i in range (1,n+1):
        ls.append(i)
        time.sleep(0.5)
        
def count(n):
    for i in range (1,n+1):
        ls.append(i)
        time.sleep(0.5)

x = threading.Thread(target= count, args = (5,))
x.start()

y = threading.Thread(target= count, args = (5,))
y.start()


time.sleep(0.5)
print(ls)

# In 18
import threading 
import time

ls = []

def count(n):
    for i in range (1,n+1):
        ls.append(i)
        time.sleep(0.5)
        
def count(n):
    for i in range (1,n+1):
        ls.append(i)
        time.sleep(0.5)

x = threading.Thread(target= count, args = (5,))
x.start()

y = threading.Thread(target= count, args = (5,))
y.start()


x.join()
y.join()
print(ls)

# In 19  

"""Thread module emulating a subset of Java's threading model."""

import os as _os
import sys as _sys
import _thread
import functools

from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
try:
    from _collections import deque as _deque
except ImportError:
    from collections import deque as _deque

# Note regarding PEP 8 compliant names
#  This threading model was originally inspired by Java, and inherited
# the convention of camelCase function and method names from that
# language. Those original names are not in any imminent danger of
# being deprecated (even for Py3k),so this module provides them as an
# alias for the PEP 8 compliant names
# Note that using the new PEP 8 compliant names facilitates substitution
# with the multiprocessing module, which doesn't provide the old
# Java inspired names.

__all__ = ['get_ident', 'active_count', 'Condition', 'current_thread',
           'enumerate', 'main_thread', 'TIMEOUT_MAX',
           'Event', 'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Thread',
           'Barrier', 'BrokenBarrierError', 'Timer', 'ThreadError',
           'setprofile', 'settrace', 'local', 'stack_size',
           'excepthook', 'ExceptHookArgs']

# Rename some stuff so "from threading import *" is safe
_start_new_thread = _thread.start_new_thread
_allocate_lock = _thread.allocate_lock
_set_sentinel = _thread._set_sentinel
get_ident = _thread.get_ident
try:
    get_native_id = _thread.get_native_id
    _HAVE_THREAD_NATIVE_ID = True
    __all__.append('get_native_id')
except AttributeError:
    _HAVE_THREAD_NATIVE_ID = False
ThreadError = _thread.error
try:
    _CRLock = _thread.RLock
except AttributeError:
    _CRLock = None
TIMEOUT_MAX = _thread.TIMEOUT_MAX
del _thread


# Support for profile and trace hooks

_profile_hook = None
_trace_hook = None

def setprofile(func):
    """Set a profile function for all threads started from the threading module.
    The func will be passed to sys.setprofile() for each thread, before its
    run() method is called.
    """
    global _profile_hook
    _profile_hook = func

def settrace(func):
    """Set a trace function for all threads started from the threading module.
    The func will be passed to sys.settrace() for each thread, before its run()
    method is called.
    """
    global _trace_hook
    _trace_hook = func

# Synchronization classes

Lock = _allocate_lock

def RLock(*args, **kwargs):
    """Factory function that returns a new reentrant lock.
    A reentrant lock must be released by the thread that acquired it. Once a
    thread has acquired a reentrant lock, the same thread may acquire it again
    without blocking; the thread must release it once for each time it has
    acquired it.
    """
    if _CRLock is None:
        return _PyRLock(*args, **kwargs)
    return _CRLock(*args, **kwargs)

class _RLock:
    """This class implements reentrant lock objects.
    A reentrant lock must be released by the thread that acquired it. Once a
    thread has acquired a reentrant lock, the same thread may acquire it
    again without blocking; the thread must release it once for each time it
    has acquired it.
    """

    def __init__(self):
        self._block = _allocate_lock()
        self._owner = None
        self._count = 0

    def __repr__(self):
        owner = self._owner
        try:
            owner = _active[owner].name
        except KeyError:
            pass
        return "<%s %s.%s object owner=%r count=%d at %s>" % (
            "locked" if self._block.locked() else "unlocked",
            self.__class__.__module__,
            self.__class__.__qualname__,
            owner,
            self._count,
            hex(id(self))
        )

    def _at_fork_reinit(self):
        self._block._at_fork_reinit()
        self._owner = None
        self._count = 0

    def acquire(self, blocking=True, timeout=-1):
        """Acquire a lock, blocking or non-blocking.
        When invoked without arguments: if this thread already owns the lock,
        increment the recursion level by one, and return immediately. Otherwise,
        if another thread owns the lock, block until the lock is unlocked. Once
        the lock is unlocked (not owned by any thread), then grab ownership, set
        the recursion level to one, and return. If more than one thread is
        blocked waiting until the lock is unlocked, only one at a time will be
        able to grab ownership of the lock. There is no return value in this
        case.
        When invoked with the blocking argument set to true, do the same thing
        as when called without arguments, and return true.
        When invoked with the blocking argument set to false, do not block. If a
        call without an argument would block, return false immediately;
        otherwise, do the same thing as when called without arguments, and
        return true.
        When invoked with the floating-point timeout argument set to a positive
        value, block for at most the number of seconds specified by timeout
        and as long as the lock cannot be acquired.  Return true if the lock has
        been acquired, false if the timeout has elapsed.
        """
        me = get_ident()
        if self._owner == me:
            self._count += 1
            return 1
        rc = self._block.acquire(blocking, timeout)
        if rc:
            self._owner = me
            self._count = 1
        return rc

    __enter__ = acquire

    def release(self):
        """Release a lock, decrementing the recursion level.
        If after the decrement it is zero, reset the lock to unlocked (not owned
        by any thread), and if any other threads are blocked waiting for the
        lock to become unlocked, allow exactly one of them to proceed. If after
        the decrement the recursion level is still nonzero, the lock remains
        locked and owned by the calling thread.
        Only call this method when the calling thread owns the lock. A
        RuntimeError is raised if this method is called when the lock is
        unlocked.
        There is no return value.
        """
        if self._owner != get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count = count = self._count - 1
        if not count:
            self._owner = None
            self._block.release()

    def __exit__(self, t, v, tb):
        self.release()

    # Internal methods used by condition variables

    def _acquire_restore(self, state):
        self._block.acquire()
        self._count, self._owner = state

    def _release_save(self):
        if self._count == 0:
            raise RuntimeError("cannot release un-acquired lock")
        count = self._count
        self._count = 0
        owner = self._owner
        self._owner = None
        self._block.release()
        return (count, owner)

    def _is_owned(self):
        return self._owner == get_ident()

_PyRLock = _RLock


class Condition:
    """Class that implements a condition variable.
    A condition variable allows one or more threads to wait until they are
    notified by another thread.
    If the lock argument is given and not None, it must be a Lock or RLock
    object, and it is used as the underlying lock. Otherwise, a new RLock object
    is created and used as the underlying lock.
    """

    def __init__(self, lock=None):
        if lock is None:
            lock = RLock()
        self._lock = lock
        # Export the lock's acquire() and release() methods
        self.acquire = lock.acquire
        self.release = lock.release
        # If the lock defines _release_save() and/or _acquire_restore(),
        # these override the default implementations (which just call
        # release() and acquire() on the lock).  Ditto for _is_owned().
        try:
            self._release_save = lock._release_save
        except AttributeError:
            pass
        try:
            self._acquire_restore = lock._acquire_restore
        except AttributeError:
            pass
        try:
            self._is_owned = lock._is_owned
        except AttributeError:
            pass
        self._waiters = _deque()

    def _at_fork_reinit(self):
        self._lock._at_fork_reinit()
        self._waiters.clear()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __repr__(self):
        return "<Condition(%s, %d)>" % (self._lock, len(self._waiters))

    def _release_save(self):
        self._lock.release()           # No state to save

    def _acquire_restore(self, x):
        self._lock.acquire()           # Ignore saved state

    def _is_owned(self):
        # Return True if lock is owned by current_thread.
        # This method is called only if _lock doesn't have _is_owned().
        if self._lock.acquire(False):
            self._lock.release()
            return False
        else:
            return True

    def wait(self, timeout=None):
        """Wait until notified or until a timeout occurs.
        If the calling thread has not acquired the lock when this method is
        called, a RuntimeError is raised.
        This method releases the underlying lock, and then blocks until it is
        awakened by a notify() or notify_all() call for the same condition
        variable in another thread, or until the optional timeout occurs. Once
        awakened or timed out, it re-acquires the lock and returns.
        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof).
        When the underlying lock is an RLock, it is not released using its
        release() method, since this may not actually unlock the lock when it
        was acquired multiple times recursively. Instead, an internal interface
        of the RLock class is used, which really unlocks it even when it has
        been recursively acquired several times. Another internal interface is
        then used to restore the recursion level when the lock is reacquired.
        """
        if not self._is_owned():
            raise RuntimeError("cannot wait on un-acquired lock")
        waiter = _allocate_lock()
        waiter.acquire()
        self._waiters.append(waiter)
        saved_state = self._release_save()
        gotit = False
        try:    # restore state no matter what (e.g., KeyboardInterrupt)
            if timeout is None:
                waiter.acquire()
                gotit = True
            else:
                if timeout > 0:
                    gotit = waiter.acquire(True, timeout)
                else:
                    gotit = waiter.acquire(False)
            return gotit
        finally:
            self._acquire_restore(saved_state)
            if not gotit:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass

    def wait_for(self, predicate, timeout=None):
        """Wait until a condition evaluates to True.
        predicate should be a callable which result will be interpreted as a
        boolean value.  A timeout may be provided giving the maximum time to
        wait.
        """
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = _time() + waittime
                else:
                    waittime = endtime - _time()
                    if waittime <= 0:
                        break
            self.wait(waittime)
            result = predicate()
        return result

    def notify(self, n=1):
        """Wake up one or more threads waiting on this condition, if any.
        If the calling thread has not acquired the lock when this method is
        called, a RuntimeError is raised.
        This method wakes up at most n of the threads waiting for the condition
        variable; it is a no-op if no threads are waiting.
        """
        if not self._is_owned():
            raise RuntimeError("cannot notify on un-acquired lock")
        all_waiters = self._waiters
        waiters_to_notify = _deque(_islice(all_waiters, n))
        if not waiters_to_notify:
            return
        for waiter in waiters_to_notify:
            waiter.release()
            try:
                all_waiters.remove(waiter)
            except ValueError:
                pass

    def notify_all(self):
        """Wake up all threads waiting on this condition.
        If the calling thread has not acquired the lock when this method
        is called, a RuntimeError is raised.
        """
        self.notify(len(self._waiters))

    notifyAll = notify_all


class Semaphore:
    """This class implements semaphore objects.
    Semaphores manage a counter representing the number of release() calls minus
    the number of acquire() calls, plus an initial value. The acquire() method
    blocks if necessary until it can return without making the counter
    negative. If not given, value defaults to 1.
    """

    # After Tim Peters' semaphore class, but not quite the same (no maximum)

    def __init__(self, value=1):
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._cond = Condition(Lock())
        self._value = value

    def acquire(self, blocking=True, timeout=None):
        """Acquire a semaphore, decrementing the internal counter by one.
        When invoked without arguments: if the internal counter is larger than
        zero on entry, decrement it by one and return immediately. If it is zero
        on entry, block, waiting until some other thread has called release() to
        make it larger than zero. This is done with proper interlocking so that
        if multiple acquire() calls are blocked, release() will wake exactly one
        of them up. The implementation may pick one at random, so the order in
        which blocked threads are awakened should not be relied on. There is no
        return value in this case.
        When invoked with blocking set to true, do the same thing as when called
        without arguments, and return true.
        When invoked with blocking set to false, do not block. If a call without
        an argument would block, return false immediately; otherwise, do the
        same thing as when called without arguments, and return true.
        When invoked with a timeout other than None, it will block for at
        most timeout seconds.  If acquire does not complete successfully in
        that interval, return false.  Return true otherwise.
        """
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        rc = False
        endtime = None
        with self._cond:
            while self._value == 0:
                if not blocking:
                    break
                if timeout is not None:
                    if endtime is None:
                        endtime = _time() + timeout
                    else:
                        timeout = endtime - _time()
                        if timeout <= 0:
                            break
                self._cond.wait(timeout)
            else:
                self._value -= 1
                rc = True
        return rc

    __enter__ = acquire

    def release(self, n=1):
        """Release a semaphore, incrementing the internal counter by one or more.
        When the counter is zero on entry and another thread is waiting for it
        to become larger than zero again, wake up that thread.
        """
        if n < 1:
            raise ValueError('n must be one or more')
        with self._cond:
            self._value += n
            for i in range(n):
                self._cond.notify()

    def __exit__(self, t, v, tb):
        self.release()


class BoundedSemaphore(Semaphore):
    """Implements a bounded semaphore.
    A bounded semaphore checks to make sure its current value doesn't exceed its
    initial value. If it does, ValueError is raised. In most situations
    semaphores are used to guard resources with limited capacity.
    If the semaphore is released too many times it's a sign of a bug. If not
    given, value defaults to 1.
    Like regular semaphores, bounded semaphores manage a counter representing
    the number of release() calls minus the number of acquire() calls, plus an
    initial value. The acquire() method blocks if necessary until it can return
    without making the counter negative. If not given, value defaults to 1.
    """

    def __init__(self, value=1):
        Semaphore.__init__(self, value)
        self._initial_value = value

    def release(self, n=1):
        """Release a semaphore, incrementing the internal counter by one or more.
        When the counter is zero on entry and another thread is waiting for it
        to become larger than zero again, wake up that thread.
        If the number of releases exceeds the number of acquires,
        raise a ValueError.
        """
        if n < 1:
            raise ValueError('n must be one or more')
        with self._cond:
            if self._value + n > self._initial_value:
                raise ValueError("Semaphore released too many times")
            self._value += n
            for i in range(n):
                self._cond.notify()


class Event:
    """Class implementing event objects.
    Events manage a flag that can be set to true with the set() method and reset
    to false with the clear() method. The wait() method blocks until the flag is
    true.  The flag is initially false.
    """

    # After Tim Peters' event class (without is_posted())

    def __init__(self):
        self._cond = Condition(Lock())
        self._flag = False

    def _at_fork_reinit(self):
        # Private method called by Thread._reset_internal_locks()
        self._cond._at_fork_reinit()

    def is_set(self):
        """Return true if and only if the internal flag is true."""
        return self._flag

    isSet = is_set

    def set(self):
        """Set the internal flag to true.
        All threads waiting for it to become true are awakened. Threads
        that call wait() once the flag is true will not block at all.
        """
        with self._cond:
            self._flag = True
            self._cond.notify_all()

    def clear(self):
        """Reset the internal flag to false.
        Subsequently, threads calling wait() will block until set() is called to
        set the internal flag to true again.
        """
        with self._cond:
            self._flag = False

    def wait(self, timeout=None):
        """Block until the internal flag is true.
        If the internal flag is true on entry, return immediately. Otherwise,
        block until another thread calls set() to set the flag to true, or until
        the optional timeout occurs.
        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof).
        This method returns the internal flag on exit, so it will always return
        True except if a timeout is given and the operation times out.
        """
        with self._cond:
            signaled = self._flag
            if not signaled:
                signaled = self._cond.wait(timeout)
            return signaled


# A barrier class.  Inspired in part by the pthread_barrier_* api and
# the CyclicBarrier class from Java.  See
# http://sourceware.org/pthreads-win32/manual/pthread_barrier_init.html and
# http://java.sun.com/j2se/1.5.0/docs/api/java/util/concurrent/
#        CyclicBarrier.html
# for information.
# We maintain two main states, 'filling' and 'draining' enabling the barrier
# to be cyclic.  Threads are not allowed into it until it has fully drained
# since the previous cycle.  In addition, a 'resetting' state exists which is
# similar to 'draining' except that threads leave with a BrokenBarrierError,
# and a 'broken' state in which all threads get the exception.
class Barrier:
    """Implements a Barrier.
    Useful for synchronizing a fixed number of threads at known synchronization
    points.  Threads block on 'wait()' and are simultaneously awoken once they
    have all made that call.
    """

    def __init__(self, parties, action=None, timeout=None):
        """Create a barrier, initialised to 'parties' threads.
        'action' is a callable which, when supplied, will be called by one of
        the threads after they have all entered the barrier and just prior to
        releasing them all. If a 'timeout' is provided, it is used as the
        default for all subsequent 'wait()' calls.
        """
        self._cond = Condition(Lock())
        self._action = action
        self._timeout = timeout
        self._parties = parties
        self._state = 0 #0 filling, 1, draining, -1 resetting, -2 broken
        self._count = 0

    def wait(self, timeout=None):
        """Wait for the barrier.
        When the specified number of threads have started waiting, they are all
        simultaneously awoken. If an 'action' was provided for the barrier, one
        of the threads will have executed that callback prior to returning.
        Returns an individual index number from 0 to 'parties-1'.
        """
        if timeout is None:
            timeout = self._timeout
        with self._cond:
            self._enter() # Block while the barrier drains.
            index = self._count
            self._count += 1
            try:
                if index + 1 == self._parties:
                    # We release the barrier
                    self._release()
                else:
                    # We wait until someone releases us
                    self._wait(timeout)
                return index
            finally:
                self._count -= 1
                # Wake up any threads waiting for barrier to drain.
                self._exit()

    # Block until the barrier is ready for us, or raise an exception
    # if it is broken.
    def _enter(self):
        while self._state in (-1, 1):
            # It is draining or resetting, wait until done
            self._cond.wait()
        #see if the barrier is in a broken state
        if self._state < 0:
            raise BrokenBarrierError
        assert self._state == 0

    # Optionally run the 'action' and release the threads waiting
    # in the barrier.
    def _release(self):
        try:
            if self._action:
                self._action()
            # enter draining state
            self._state = 1
            self._cond.notify_all()
        except:
            #an exception during the _action handler.  Break and reraise
            self._break()
            raise

    # Wait in the barrier until we are released.  Raise an exception
    # if the barrier is reset or broken.
    def _wait(self, timeout):
        if not self._cond.wait_for(lambda : self._state != 0, timeout):
            #timed out.  Break the barrier
            self._break()
            raise BrokenBarrierError
        if self._state < 0:
            raise BrokenBarrierError
        assert self._state == 1

    # If we are the last thread to exit the barrier, signal any threads
    # waiting for the barrier to drain.
    def _exit(self):
        if self._count == 0:
            if self._state in (-1, 1):
                #resetting or draining
                self._state = 0
                self._cond.notify_all()

    def reset(self):
        """Reset the barrier to the initial state.
        Any threads currently waiting will get the BrokenBarrier exception
        raised.
        """
        with self._cond:
            if self._count > 0:
                if self._state == 0:
                    #reset the barrier, waking up threads
                    self._state = -1
                elif self._state == -2:
                    #was broken, set it to reset state
                    #which clears when the last thread exits
                    self._state = -1
            else:
                self._state = 0
            self._cond.notify_all()

    def abort(self):
        """Place the barrier into a 'broken' state.
        Useful in case of error.  Any currently waiting threads and threads
        attempting to 'wait()' will have BrokenBarrierError raised.
        """
        with self._cond:
            self._break()

    def _break(self):
        # An internal error was detected.  The barrier is set to
        # a broken state all parties awakened.
        self._state = -2
        self._cond.notify_all()

    @property
    def parties(self):
        """Return the number of threads required to trip the barrier."""
        return self._parties

    @property
    def n_waiting(self):
        """Return the number of threads currently waiting at the barrier."""
        # We don't need synchronization here since this is an ephemeral result
        # anyway.  It returns the correct value in the steady state.
        if self._state == 0:
            return self._count
        return 0

    @property
    def broken(self):
        """Return True if the barrier is in a broken state."""
        return self._state == -2

# exception raised by the Barrier class
class BrokenBarrierError(RuntimeError):
    pass


# Helper to generate new thread names
_counter = _count().__next__
_counter() # Consume 0 so first non-main thread has id 1.
def _newname(template="Thread-%d"):
    return template % _counter()

# Active thread administration
_active_limbo_lock = _allocate_lock()
_active = {}    # maps thread id to Thread object
_limbo = {}
_dangling = WeakSet()
# Set of Thread._tstate_lock locks of non-daemon threads used by _shutdown()
# to wait until all Python thread states get deleted:
# see Thread._set_tstate_lock().
_shutdown_locks_lock = _allocate_lock()
_shutdown_locks = set()





    



       

