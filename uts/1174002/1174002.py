# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:29:56 2020

@author: Habib
"""




import threading
import time



def thread1(n, name):
    print('Hallo, saya {}. akan melakukan thread dalam 5 detik \n'.format(name))
    time.sleep(n)
    print('{} thread telah dilakukan \n'.format(name))
    
    
    
    
t = threading.Thread(target = thread1, name = 'habib', args =(5, 'habib') )
 
 
t.start()
 
 
 
t.join()
 
 
#Threading part 2

#multiple threads




def thread2(n, name):
    print('Hallo, saya {}. akan melakukan thread secara bersamaan dalam 5 detik \n'.format(name))
    time.sleep(n)
    print('{} thread telah dilakukan \n'.format(name))
    
    
 
start = time.time()
threads = []

for i in range(5):
    t = threading.Thread(target = thread2, name = 'habib{}'.format(i), args =(5,'habib{}'.format(i) ) )
    threads.append(t)
    t.start()
    print('{} thread sedang berjalan \n'.format(t.name))
 
 
 
    
for i in threads:

    i.join()
    
end = time.time()



print('waktu thread {}'.format(end - start))






#example 2 without threads


import time

def thread3(n, i):
    print ('Hallo, ini adalah Fungsi {}. akan membuka fungsi selanjutnya dalam 5 detik \n'.format(i))
    time.sleep(n)
    print('Fungsi{} Telah terbuka \n'.format(i))




start = time.time()

for i in range(5):
    print('{} Dimulai \n'.format(i))
    x = thread3(5, i)
    
    
end  = time.time()


print('Waktu Yang diperlukan: {}'.format(end - start))

 
 
 
 
 