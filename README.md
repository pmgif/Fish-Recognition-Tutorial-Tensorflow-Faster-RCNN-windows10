# Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10
บทความนี้สอนเกี่ยวกับวิธีใช้ API การตรวจจับวัตถุของ Tensorflow เพื่อฝึกอบรมแยกประเภทการตรวจจับวัตถุสำหรับวัตถุหลาย ๆ ตัวใน Windows 10 เขียนโดยใช้ Tensorflow เวอร์ชัน 1.15 โดยใช้การรู้จำปลาโดยใช้อัลกอริธึม Faster R-CNN

#### ฉันได้จัดทำวีดีโอบน YouTube ที่ทำตามบทความนี้ เพื่อให้เข้าใจมากยิ่งขึ้น

ที่มา : การติดตั้งโปรแกรม จาก [Jeff Heaton](https://youtu.be/qrkEYf-YDyI) และ การทำ Tensorflow จาก [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

**ขั้นตอนการจัดทำมีดังนี้**

1. 

### ขั้นตอน

#### 1. ติดตั้งโปรแกรม
ในส่วนนี้จะทำการติดตั้งโปรแกรมต่างๆ ที่จำเป็นต้องใช้ในการทำ Tensorflow

### 1a. ติดตั้ง Visual Studio
เข้าไปที่เว็บไซต์ [Visual Studio](https://visualstudio.microsoft.com/downloads/) และเลือกเวอร์ชันที่ต้องการ
### 1b. ดาวน์โหลด Setup Script
เข้าไปที่เว็บไซต์ [Setup Script](https://github.com/jeffheaton/t81_558_deep_learning) และเลือกไฟล์ tensorflow-gpu.yml
### 1c. ติดตั้ง Driver Graphics
เข้าไปที่เว็บไซต์ [Driver Graphics](https://www.nvidia.com/Download/index.aspx?lang=th) และเลือกเวอร์ชันไดร์เวอร์ให้ตรงกับ GPU ที่ใช้
### 1d. ติดตั้ง CUDA 10.0
เข้าไปที่เว็บไซต์ [CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) โดยกดเลือกให้ตรงกับที่ต้องการ ดังรูปภาพ
![54dd1b6d53693fe6823973bf9ed32244.jpg](https://www.img.in.th/images/54dd1b6d53693fe6823973bf9ed32244.jpg)
### 1e. ติดตั้ง cuDNN 7.6.5
เข้าไปที่เว็บไซต์ [cuDNN](https://developer.nvidia.com/cudnn) โดยกดเลือกให้ตรงกับเวอร์ชันของ [CUDA][### 1d. ติดตั้ง CUDA 10.0] ดังรูปภาพ
![734842c8688a364327c01c61176675d0.jpg](https://www.img.in.th/images/734842c8688a364327c01c61176675d0.jpg)
