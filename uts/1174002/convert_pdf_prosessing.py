# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:59:45 2020

@author: Habib
"""


from PIL import Image
import os

def convert():
    conv = input("Apakah kamu yakin akan meng-convert gambar ke PDF (y/n)? ")
    if(conv == 'y'):
        filename = "/Users/Habib/Desktop/proses/1.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/1.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/2.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/2.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
                
                
        
        filename = "/Users/Habib/Desktop/proses/3.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/3.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/4.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/4.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
                
        
        filename = "/Users/Habib/Desktop/proses/5.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/5.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/6.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/6.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
                
        
        filename = "/Users/Habib/Desktop/proses/7.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/7.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
                
        
        filename = "/Users/Habib/Desktop/proses/8.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/8.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
                
        
        filename = "/Users/Habib/Desktop/proses/9.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/9.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/10.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/10.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/11.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/11.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/12.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/12.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/13.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/13.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/14.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/14.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/15.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/15.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/16.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/16.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/17.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/17.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        filename = "/Users/Habib/Desktop/proses/18.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/18.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/19.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/19.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/20.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/20.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/21.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/21.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/22.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/22.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/23.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/23.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/24.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/24.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/25.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/25.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/26.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/26.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/27.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/27.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/28.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/28.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/29.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/29.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/30.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/30.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/31.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/31.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/32.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/32.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/33.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/33.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/34.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/34.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/35.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/35.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/36.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/36.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/37.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/37.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/38.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/38.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/39.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/39.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/40.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/40.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/41.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/41.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/42.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/42.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/43.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/43.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/44.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/44.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/45.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/45.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/46.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/46.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/47.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/47.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/48.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/48.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/49.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/49.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/50.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/50.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/51.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/51.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/52.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/52.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/53.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/53.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/54.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/54.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/55.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/55.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/56.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/56.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/57.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/57.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/58.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/58.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/59.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/59.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/60.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/60.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/61.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/61.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/62.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/62.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/63.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/63.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/64.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/64.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/65.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/65.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        filename = "/Users/Habib/Desktop/proses/66.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/66.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
                
                
                
        
        filename = "/Users/Habib/Desktop/proses/67.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/67.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        filename = "/Users/Habib/Desktop/proses/68.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/68.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        filename = "/Users/Habib/Desktop/proses/69.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/69.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/70.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/70.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/71.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/71.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/72.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/72.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        
        
        
        
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/73.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/73.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/74.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/74.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/75.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/75.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/76.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/76.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/78.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/78.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/79.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/79.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/80.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/80.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/81.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/81.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/82.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/82.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/83.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/83.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/84.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/84.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/85.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/85.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/86.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/86.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/87.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/87.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/88.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/88.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/89.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/89.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/90.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/90.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/91.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/91.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/92.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/92.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/93.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/93.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/94.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/94.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/95.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/95.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/96.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/96.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/97.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/97.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/98.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/98.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/99.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/99.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/100.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/100.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/101.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/101.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/102.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/102.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/103.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/103.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/104.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/104.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/105.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/105.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/106.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/106.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/107.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/107.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/108.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/108.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/109.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/109.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/110.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/110.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/111.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/111.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/112.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/112.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        filename = "/Users/Habib/Desktop/proses/113.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/113.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/114.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/114.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/115.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/115.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
        
        
        
        filename = "/Users/Habib/Desktop/proses/116.jpg"
        im = Image.open(filename)
        if im.mode == "RGBA":
            im = im.convert("RGB")
        new_filename = "/Users/Habib/Desktop/proses/116.pdf"
        if not os.path.exists(new_filename):
                im.save(new_filename, "PDF", resolution=100.0)
        
                
                
                
        print("Siap Berhasil boskuuuu")
    
    elif(conv == 'n'):
        print("Oke Boskuuuu ")
    else:
        return ("Pilihan hanya yes or no")
convert()

#filename = "/Users/Habib/Desktop/dog/dog_4.jpg"
#im = Image.open(filename)
#if im.mode == "RGBA":
#    im = im.convert("RGB")
#new_filename = "/Users/Habib/Desktop/dog/dog_2.pdf"
#if not os.path.exists(new_filename):
#    im.save(new_filename, "PDF", resolution=100.0)
#    
#    
#filename = "/Users/Habib/Desktop/dog/dog_5.jpg"
#im = Image.open(filename)
#if im.mode == "RGBA":
#    im = im.convert("RGB")
#new_filename = "/Users/Habib/Desktop/dog/dog_3.pdf"
#if not os.path.exists(new_filename):
#    im.save(new_filename, "PDF", resolution=100.0)