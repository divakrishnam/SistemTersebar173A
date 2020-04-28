# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:08:56 2020

@author: Habib
"""

import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import queue
from threading import Thread




SAVE_DIR = r'C:\Users\Habib\Desktop\dog'



def decorator_function(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('dog', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'dog_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link1.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()

###########################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\bird'



def decorator_function1(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('bird', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'bird_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link2.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()

###############################################################


SAVE_DIR = r'C:\Users\Habib\Desktop\reptile'



def decorator_function2(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('reptile', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'reptile_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link3.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()

########################################################################


SAVE_DIR = r'C:\Users\Habib\Desktop\cacing'



def decorator_function3(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('cacing', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'cacing_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link4.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\ampibi'



def decorator_function4(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('ampibi', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'ampibi_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link5.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()

####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\pacuan'



def decorator_function5(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('pacuan', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'pacuan_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link6.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################

####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\invert'



def decorator_function6(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('invert', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'invert_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link7.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################

####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\sport'



def decorator_function7(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('sport', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'sport_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link8.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################


####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\batu'



def decorator_function8(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('batu', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'batu_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link9.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################



####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\fixture'



def decorator_function9(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('fixture', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'fixture_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link10.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################


####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\tanaman'



def decorator_function10(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('tanaman', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'tanaman_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link11.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################


####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\geologi'



def decorator_function11(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('geologi', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'geologi_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link12.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################

####################################################################





SAVE_DIR = r'C:\Users\Habib\Desktop\fungus'



def decorator_function12(func):
    def wrapper(*args,**kwargs):
        session = requests.Session()
        retry = Retry(connect=0, backoff_factor=0.2)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return func(*args, session = session, **kwargs)
    return wrapper








#Using threading:
image_count = 0

#optional decorator_function
#@decorator_function   
def download_image(SAVE_DIR,q, session = None):
        global image_count
        if not session:
                session = requests.Session()
        while not q.empty():
            
            try:

                    r = session.get(q.get(block = False))

            except (requests.exceptions.RequestException, UnicodeError) as e:
                print(e)
                image_count += 1
                q.task_done()
                continue

            image_count += 1
            q.task_done()

            print('fungus', image_count)
            with open(os.path.join(
                        SAVE_DIR, 'fungus_{}.jpg'.format(image_count)),
                        'wb') as f:

                f.write(r.content)
                
               

q =queue.Queue()
with open(r'C:\Users\Habib\Desktop\link13.txt', 'rt') as f:
    for i in range(10):
        line = f.readline()
        q.put(line.strip())
print(q.qsize())




threads = []
start = time.time()
for i in range(10):
     t = Thread(target = download_image, 
                args = (SAVE_DIR,q))
     #t.setDaemon(True)
     threads.append(t)
     t.start()
q.join()


##################################################################
#for t in threads:
#    t.join()
#    print(t.name, 'has joined')



end = time.time()
print('Waktu yang diperlukan untuk mendownload: {:.4f}'.format(end - start))



#time taken: 7.4640
#time taken: 5.0860













                      

                      
