# Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10
บทความนี้สอนเกี่ยวกับวิธีใช้ API การตรวจจับวัตถุของ Tensorflow เพื่อฝึกอบรมแยกประเภทการตรวจจับวัตถุสำหรับวัตถุหลาย ๆ ตัวใน Windows 10 เขียนโดยใช้ Tensorflow เวอร์ชัน 1.15 โดยใช้การรู้จำปลาโดยใช้อัลกอริธึม Faster R-CNN

#### ฉันได้จัดทำวีดีโอบน YouTube ที่ทำตามบทความนี้ เพื่อให้เข้าใจมากยิ่งขึ้น

ที่มา : การติดตั้งโปรแกรม จาก [Jeff Heaton](https://youtu.be/qrkEYf-YDyI) และ การทำ Tensorflow จาก [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

**ขั้นตอนการจัดทำมีดังนี้**

1. 

## ขั้นตอน

### 1. ติดตั้งโปรแกรม
ในส่วนนี้จะทำการติดตั้งโปรแกรมต่างๆ ที่จำเป็นต้องใช้ในการทำ Tensorflow

#### 1a. ติดตั้ง Visual Studio
เข้าไปที่เว็บไซต์ [Visual Studio](https://visualstudio.microsoft.com/downloads/) และเลือกเวอร์ชันที่ต้องการ
#### 1b. ดาวน์โหลด Setup Script
เข้าไปที่เว็บไซต์ [Setup Script](https://github.com/jeffheaton/t81_558_deep_learning) และเลือกไฟล์ tensorflow-gpu.yml
#### 1c. ติดตั้ง Driver Graphics
เข้าไปที่เว็บไซต์ [Driver Graphics](https://www.nvidia.com/Download/index.aspx?lang=th) และเลือกเวอร์ชันไดร์เวอร์ให้ตรงกับ GPU ที่ใช้
#### 1d. ติดตั้ง CUDA 10.0
เข้าไปที่เว็บไซต์ [CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) โดยกดเลือกให้ตรงกับที่ต้องการ ดังรูปภาพ
![54dd1b6d53693fe6823973bf9ed32244.jpg](https://www.img.in.th/images/54dd1b6d53693fe6823973bf9ed32244.jpg)
#### 1e. ติดตั้ง cuDNN 7.6.5
เข้าไปที่เว็บไซต์ [cuDNN](https://developer.nvidia.com/cudnn) โดยกดเลือกให้ตรงกับเวอร์ชันของ [CUDA] ดังรูปภาพ
![734842c8688a364327c01c61176675d0.jpg](https://www.img.in.th/images/734842c8688a364327c01c61176675d0.jpg)
จากนั้นทำการ Set Path
```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin; 
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64; 
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include; 
SET PATH=C:\tools\cuda\bin;.
```
#### 1f. ติดตั้ง Anaconda 3.7
เข้าไปที่เว็บไซต์ [Anaconda](https://www.anaconda.com/distribution/) โดยเลือกเวอร์ชันที่ต้องการใช้ ทำการตั้งค่าให้ตรงกับภาพนี้
![67a92acda91b3d9ea3a97d9525060cae.jpg](https://www.img.in.th/images/67a92acda91b3d9ea3a97d9525060cae.jpg)
#### 1g. ติดตั้ง TensorRT 5.0 GA for Windows
เข้าไปที่เว็บไซต์ [TensorRT](https://developer.nvidia.com/tensorrt) โดยเลือกเวอร์ชันที่ต้องการใช้ และทำการ Set Path ที่อยู่ของโปรแกรม TensorRT
#### 1h. ติดตั้ง Miniconda3 Python 3.7 for Windows 64-bit
เข้าไปที่เว็บไซต์ [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) โดยเลือกเวอร์ชันที่ต้องการใช้ ทำการตั้งค่าให้ตรงกับภาพนี้
![87b212e455b1a03ea0d0adf1a79befc8.jpg](https://www.img.in.th/images/87b212e455b1a03ea0d0adf1a79befc8.jpg)
#### 1i. ติดตั้ง Python 3.7.6
เข้าไปที่เว็บไซต์ [Python](https://www.python.org/downloads/) โดยเลือกเวอร์ชันที่ต้องการใช้ ทำการตั้งค่าให้ตรงกับภาพนี้
![e330a2dcfde9aba39402ed300d867b5a.jpg](https://www.img.in.th/images/e330a2dcfde9aba39402ed300d867b5a.jpg)
#### 1j. ติดตั้ง ติดตั้ง Jupyter
เปิดใช้งานโปรแกรม Command Prompt ขึ้นมา พิมพ์คำสั่งดังนี้
```
C:\Users\Madi> conda install jupyter
```
#### 1k. ประมวลผล Setup Script
- ค้นหาที่อยู่ของไฟล์ .yml
```
C:\Users\Madi> dir *.yml
```
- สร้าง Environment ของ tensorflow.yml
```
C:\Users\Madi> conda env create -v -f tensorflow.yml
```
#### 1l. ติดตั้ง Jupyter Kernel
- เปิดใช้งาน environment tensorflow
```
C:\Users\Madi> Conda activate tensorflow
```
- ติดตั้ง Jupyter Kernel ของ python
```
(tensorflow) C:\Users\Madi> python -m ipykernel install --user --name tensorflow --display-name "Python 3.7 (tensorflow)"
```
- ทดลองเขียน python เพื่อทดสอบการใช้งานของ tensorflow
```
(tensorflow) C:\Users\Madi> python
```
```
>>> import tensorflow as tf
>>> print(tf.__version__)
>>> quit()
```
#### 1m. ทดสอบการทำงานของ Tensorflow
เข้าไปที่ [เว็บไซต์](https://github.com/jeffheaton/t81_558_deep_learning) จากนั้นทำการโหลดไฟล์และแตกไฟล์
- เปลี่ยน directory
```
(tensorflow) C:\Users\Madi> cd t81_558_deep_learning-master
```
- เรียกใช้ jupyter notebook แล้วเลือกไฟล์ t81_558_class_01_1_overview.ipynb
```
(tensorflow) C:\Users\Madi\t81_558_deep_learning-master> jupyter notebook
```
- เปลี่ยนประเภทของ kernel จากเวอร์ชัน 3 เป็นเวอร์ชัน 3.7
- Run code ดังกล่าว เพื่อเรียกใช้งาน GPU

### 2. เตรียมภาพ
เราจะใช้โปรแกรม LabelImg เวอร์ชัน 1.6 ในส่วนของการตีกรอบภาพ สามารถดาวน์โหลดได้จาก[ลิ้งค์](https://www.dropbox.com/s/fc1e9b3jyy9udkm/windows_v1.6.0.zip?dl=1)นี้
