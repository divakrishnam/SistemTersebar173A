# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:55:21 2020

@author: ONIWALDUS
"""


# In[Perhitungan Kalkulator]:

# fungsi penjumlahan




# fungsi pengurangan
   
def subtract(x, y):
   return x - y



# fungsi perkalian
   
def multiply(x, y):
   return x * y



# fungsi pembagian
   
def divide(x, y):
   return x / y



# menu operasi
   
print("Pilih Operasi.")

print("1.Jumlah")

print("2.Kurang")
  
print("3.Kali")

print("4.Bagi")



# Meminta input dari user

choice = input("Masukkan pilihan(1/2/3/4): ")


num1 = int(input("Masukkan bilangan pertama: "))


num2 = int(input("Masukkan bilangan kedua: "))


if choice == '1':
   print(num1,"+",num2,"=", add(num1,num2))


elif choice == '2':
   print(num1,"-",num2,"=", subtract(num1,num2))


elif choice == '3':
   print(num1,"*",num2,"=", multiply(num1,num2))


elif choice == '4':
   print(num1,"/",num2,"=", divide(num1,num2))


else:
   print("Input salah")


   
# In[Python Untuk Menjumlahkan Dua Bilangan]:

# Program Penjumlahan Dua Bilangan 
   
# Meminta inputan dari user 



bil1 = input('Masukkan bilangan pertama: ') 



bil2 = input('Masukkan bilangan kedua: ') 



# Menjumlahkan bilangan 

jumlah = float(bil1) + float(bil2) 



# Menampilkan jumlah 
print('Jumlah {0} + {1} adalah {2}'.format(bil1, bil2, jumlah))



# In[mengirim Pesan Melalui Python]:

import smtplib


from email.mime.multipart import MIMEMultipart


from email.mime.text import MIMEText


from email.mime.base import MIMEBase


from email import encoders


fromaddr = "oniwaldus@gmail.com"


toaddr = "oniwaldus0910@gmail.com"
 

msg = MIMEMultipart()

 

msg['From'] = fromaddr


msg['To'] = toaddr


msg['Subject'] = "uts sistem tersebar"

 
body = "Tolong Cek itu saya ada kirim tugas uts sistem tersebar"
 

msg.attach(MIMEText(body, 'plain'))


# Lampiran, sesuaikan nama filename dengan nama di attachment

filename = "uts.txt"


attachment = open("D:\\KULIAH\\SEMESTER 6\\Sistem Tersebar\\uts.txt", "rb")
 

part = MIMEBase('application', 'octet-stream')


part.set_payload((attachment).read())


encoders.encode_base64(part)


part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
 

msg.attach(part)
 

server = smtplib.SMTP('smtp.gmail.com', 587)


server.starttls()


server.login(fromaddr, "inipassword")


text = msg.as_string()


server.sendmail(fromaddr, toaddr, text)


server.quit()


# In[Menentukan Bilangan Prima dari 1-50]:

# Program untuk menampilkan semua bilangan prima pada interval tertentu 

# Ubah nilai lower dan upper untuk hasil yang lain 


lower = 1 


upper = 50


print("Bilangan prima antara",lower,"and",upper,":") 


for num in range(lower,upper + 1): 

    if num > 1: 

        for i in range(2,num): 
            if (num % i) == 0: 
                break 

        else: 
            print(num) 


# In[Menentukan Faktor Persekutuan Terbesar dengan Dua Bilangan]:

# Program Python untuk menemukan FPB dua buah bilangan

# mendefinisikan fungsi


def hitung_FPB(x, y):

    
    # memilih bilangan yang paling kecil
    if x > y:
        smaller = y
    
    else:
        smaller = x
    
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            fpb = i
            
    
    return fpb


num1 = 100


num2 = 50


print("FPB dari", num1,"dan", num2," =", hitung_FPB(num1, num2))



# In[Berhitung sampai sepuluh. Lama penundaan perhitungan adalah 0.5 detik]:

import time


for i in range(10):
    
    time.sleep(0.5)
    
    
    print ('Berhitung: ' + str(i +1))
    
    




# In[Mengambil Nilai Input dari Keyboard]:

variable = input("Sebuah Teks")


# Mengambil input
nama = input("Siapa Nama Anda: ")


umur = input("Berapa Umur Anda: ")


# Menampilkan output
print ("Hello",nama,"Umur Anda Adalah",umur,"tahun")



# In[Menampilkan Nilai Output]:

print ("Hello World!")


print (variable)


print ('Gabung dengan', nama)



# In[Menampilkan Variabel dan Teks]:

nama = "Oniwaldus Bere Mali"


kelas = "D4 TI 3A"


npm = "1174005"


print ("Hello Nama Saya",nama, "Kelas",kelas, "NPM",npm)



# In[Menggunakan Fungsi Format]:

nama = input("Nama: ")


print ("Hayyy {} Kamu Memang terbaik".format(nama))



# In[Contoh Lainnya]:

nama_mu = input("Nama kamu: ")


nama_dia = input("Nama dia: ")


print ("{} dengan {} sepertinya itu kurang cocok".format(nama_mu, nama_dia))




# In[Membuka File Berbaris ke 0 dan ke 2]:


# buka file
file_referensi = open("D:\\KULIAH\\SEMESTER 6\\Sistem Tersebar\\uts.txt", "rb")



# baca isi file
referensi = file_referensi.readlines()



# cetak baris pertama
print (referensi[0])



# cetak baris kedua
print (referensi[2])



# tutup file
file_referensi.close()



# In[Menampilkan Semua Baris File]:

# buka file
file_referensi = open("D:\\KULIAH\\SEMESTER 6\\Sistem Tersebar\\uts.txt", "rb")



# cetak baris pertama
print (referensi)



# tutup file
file_referensi.close()



# In[Menulis File dan Menyimpannya]:

print ("Selamat Datang Untuk Mau Mengisi  Biodata Anda")

print ("=================================")


# Ambil input dari user

nama = input("Nama: ")

umur = input("Umur: ")

alamat = input("Alamat: ")



# format teks
teks = "Nama: {}\nUmur: {}\nAlamat: {}".format(nama, umur, alamat)



# buka file untuk ditulis
file_bio = open("D:\\KULIAH\\SEMESTER 6\\Sistem Tersebar\\Biodata saya.txt", "w")



# tulis teks ke file
file_bio.write(teks)



# tutup file
file_bio.close()



# In[Menambahkan File Data]:

print ("Selamat Datang Untuk Mau Mengisi  Biodata Anda")

print ("=================================")


# Ambil input dari user

nama = input("Nama: ")

umur = input("Umur: ")

alamat = input("Alamat: ")



# format teks
teks = "Nama: {}\nUmur: {}\nAlamat: {}".format(nama, umur, alamat)



# buka file untuk ditulis, "a" untuk menindih file
file_bio = open("D:\\KULIAH\\SEMESTER 6\\Sistem Tersebar\\Biodata saya.txt", "a")



# tulis teks ke file
file_bio.write(teks)



# tutup file
file_bio.close()




# In[Membaca dan Menulis File dengan Mode r]:

print ("selamat Datang Untuk Mau Mengisi  Biodata Anda")

print ("=================================")


# buka file untuk dibaca dan ditulis
file_bio = open("Biodata saya.txt", "r+")

teks = file_bio.read()


# cetak isi file
print (teks)



# Ambil input dari user

nama = input("Nama: ")

umur = input("Umur: ")

alamat = input("Alamat: ")


# format teks
teks = "Nama: {}\nUmur: {}\nAlamat: {}".format(nama, umur, alamat)



# tulis teks ke file
file_bio.write(teks)



# tutup file
file_bio.close()





# In[Membuat Perulangan]:

ulang = 10


for i in range(ulang):


    print ("Perulangan ke-"+str(i))



# In[Contoh Perulangan Lain dengan List]:

item = ['Sayur','Apel','Susu','Jeruk']


for isi in item:


    print (isi)



# In[Perulangan While]:

jawab = 'ya'


hitung = 0


while(True):

    hitung += 1


    jawab = input("Ulang lagi tidak? ")


    if jawab == 'tidak':
        break


print ("Total perulagan: " + str(hitung))



# In[Operasi Aritmatika Python]:

hasil = 12 + 6 * 4 - 8


print(hasil)



# In[Operasi Aritmatika Python Lain]:

hasil = (12 + 6) * (4 - 8);


print(hasil)



# In[Operator Perbandingan Python]:

x = 7


y = 10

 
print('x =',x)


print('y =',y)


print('\n')
 

print('x == y hasilnya',x==y)


print('x != y hasilnya',x!=y)


print('x > y  hasilnya',x>y)


print('x < y  hasilnya',x<y)


print('x >= y hasilnya',x>=y)


print('x <= y hasilnya',x<=y)



# In[Operator Perbandingan Python String]:

print('Oni' == 'Oni')


print('Oni' == 'Oni')


print('1234' == 1234)


print('1234' != 1234)



# In[Struktur IF]:

a = 10


if (a % 2)==0:

    
    print('Variabel a berisi angka genap')


else:
    
    
    print('Variabel a berisi angka ganjil')


# In[Operator Aritmatika Yang Lain]:
    

print (3 + 2 * 3 - 10 / 5)


# In[]:


print((3 + 2) * (3 - 10 / 5))


# In[]:


print (3 ** 2)



# In[]:


print (100 % 3)



# In[Operator Perbandingan Yang Lain]:


print(10 > 2) #lebih besar, true


print(10 < 2) #lebih kecil, false


print(10 != 2) #tidak sama dengan, true


print(5 >= 5) #lebih besar atau sama dengan, true


print(5<= 4) #lebih kecil atau sama dengan, false


print(5 == 5) #sama dengan, true



# In[Operator Logika]:


print(10 > 2 and 10 > 5)



# In[Operator dan Ekskpresi]:


bilangan1 = 5


bilangan2 = 3


print('bil1 = ', bilangan1)


print('bil2 = ', bilangan2)


print('bil1 + bil2 = ', bilangan1 + bilangan2)


print('bil1 - bil2 = %s' % (bilangan1 - bilangan2))


print('bil1 * bil2 = {0}'.format(bilangan1 * bilangan2))


print('bil1 ** bil2 = ', bilangan1 ** bilangan2)



# In[]:


bilangan1 = 5.0


print('bil1 = ', bilangan1)


print('bil2 = ', bilangan2)


print('bil1 / bil2 = ', bilangan1 / bilangan2)


print('bil1 // bil2 = ', bilangan1 // bilangan2)


print('bil1 % bil2 = ', bilangan1 % bilangan2)


print('-' * 80)


# In[]:


bilangan1 = 5


print('bil1 = ', bilangan1)


print('bil2 = ', bilangan2)


print('bil1 << bil2 = ', bilangan1 << bilangan2)


print('bil1 >> bil2 = ', bilangan1 >> bilangan2)


print('bil1 & bil2 = ', bilangan1 & bilangan2)


print('bil1 | bil2 = ', bilangan1 | bilangan2)


print('bil1 ^ bil2 = ', bilangan1 ^ bilangan2)


print('~bil1 = ', ~bilangan1)


print('-' * 80)



# In[]:


print('bil1 < bil2 = ', bilangan1 < bilangan2)


print('bil1 > bil2 = ', bilangan1 > bilangan2)


print('bil1 <= bil2 = ', bilangan1 <= bilangan2)


print('bil1 >= bil2 = ', bilangan1 >= bilangan2)


print('bil1 == bil2 = ', bilangan1 == bilangan2)


print('bil1 != bil2 = ', bilangan1 != bilangan2)


print('-' * 80)



# In[]:



print('not True = ', not True)


print('True and False = ', True and False)


print('True or False = ', True or False)



# In[Menggunakan IF ELSE]:



a = input("Masukkan Angka = ")


b = int(a)


if( b%2 == 0):
    print("Genap")


else:
    print("Ganjil")




# In[Menggunakan ELIF]:
    
    

y = input("Masukkan Angka = ")


x = int(y)


if(x>50 and x<=60):
    print("Nilai C")


elif(x>60 and x<=80):
    
    print("Nilai B")


elif(x>80 and x<=100):
    print("Nilai A")
    

else:
    print("Nilai E")



# In[Alur Control Python]:
    

nomor_acak = 7


print('tebak nomor acak dari 1 - 10')


tebakan = int(input('Tebakan anda (bil bulat): '))


if tebakan == nomor_acak:
    
    
    print('Selamat tebakan yang anda pilih  benar')
    
    
    print('Tapi tidak ada hadiah untuk anda')


elif tebakan < nomor_acak:
    
    
    print('Tebakan anda terlalu kecil')


else:
    
    
    print('Tebakan anda terlalu besar')
    

print('Selesai')



# In[While]:

nomor_acak = 77


berjalan  = True


print('Tebak nomor acak dari 1 - 100')


while berjalan:
    
    
    tebakan = int(input('Tebakan anda (bil bulat): '))
    
    
    if tebakan == nomor_acak:
        
        
        print('Selamat tebakan anda benar')
        
        
        print('Tapi tidak ada hadiah untuk anda')
        
    
        berjalan  = False
        
    
    elif tebakan < nomor_acak:
    
    
        print('Tebakan anda terlalu kecil')
        

    else:
        
        
        print('Tebakan anda terlalu besar')


else:
    
    
    print('Selesai')



# In[Perulangan FOR]:
    

for i in range(1, 6):
    
    
    print(i)
    

else:
    
    
    print('Perulangan sudah selesai')






# In[Objek dan Class]:
    
daftar_belanja = ['jeruk', 'mangga', 'apel', 'salak']
print('Saya punya %s barang yang akan dibeli' % len(daftar_belanja))


print('Barang tersebut:')


for barang in daftar_belanja:
    
   
    print(barang),
    

print('Saya harus membeli beras')
daftar_belanja.append('beras')


print('daftar belanja sekarang :', daftar_belanja)



print('Saya akan mengurutkan daftar belanja saya')
daftar_belanja.sort()


print('daftar belanja setelah diurutkan', daftar_belanja)


print('Barang yang harus daya beli pertama', daftar_belanja)
barang_pertama = daftar_belanja[0]


del daftar_belanja[0]


print('Saya memberli', barang_pertama)


print('daftar belanja sekarang:', daftar_belanja)