# Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10
บทความนี้สอนเกี่ยวกับวิธีใช้ API การตรวจจับวัตถุของ Tensorflow เพื่อฝึกอบรมแยกประเภทการตรวจจับวัตถุสำหรับวัตถุหลาย ๆ ตัวใน Windows 10 เขียนโดยใช้ Tensorflow เวอร์ชัน 1.15 โดยใช้การรู้จำปลาโดยใช้อัลกอริธึม Faster R-CNN

#### ฉันได้จัดทำวีดีโอบน YouTube ที่ทำตามบทความนี้ เพื่อให้เข้าใจมากยิ่งขึ้น
[![cbadf04d8b69b3e3fa68758ce69c33c8.md.jpg](https://www.img.in.th/images/cbadf04d8b69b3e3fa68758ce69c33c8.md.jpg)](https://youtu.be/Rgpfk6eYxJA "Fish Recognition Tutorial Tensorflow Faster-RCNN windows10 - Click to Watch!")

*ที่มา : การติดตั้งโปรแกรม จาก [Jeff Heaton](https://youtu.be/qrkEYf-YDyI) และ การทำ Tensorflow จาก [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)*

**ขั้นตอนการจัดทำมีดังนี้**

  [1. ติดตั้งโปรแกรม](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#1-ติดตั้งโปรแกรม)

  [2. ตั้งค่า Tensorflow และ Anaconda](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#2-ตั้งค่า-Tensorflow-และ-Anaconda)

  [3. เตรียมภาพ](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#3-เตรียมภาพ)

  [4. สร้าง Anaconda virtual environment ใหม่](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#4-สร้าง-Anaconda-virtual-environment-ใหม่)

  [5. สร้างข้อมูลการฝึกอบรม](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#5-สร้างข้อมูลการฝึกอบรม)

  [6. สร้าง Label Map และ กำหนดค่าการฝึกอบรม](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#6-สร้าง-Label-Map-และ-กำหนดค่าการฝึกอบรม)

  [7. ฝึกสอนแบบจำลอง](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#7-ฝึกสอนแบบจำลอง)

  [8. Export Inference Graph](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#8-Export-Inference-Graph)

  [9. ทดสอบแบบจำลอง](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#9-ทดสอบแบบจำลอง)

## แนะนำ
บทความนี้ทำเพื่อให้ข้อมูลการทำ Object Detection จากนิสิตมหาวิทยาลัยบูรพา ได้ทำการถ่ายภาพปลาจากสถาบันวิทยาศาสตร์ทางทะเล มหาวิทยาลัยบูรพา โดยมีข้อมูลภาพปลาทั้งหมด 8 สายพันธุ์ได้แก่
  1. ปลาผีเสื้อครีบยาว (Pennant coralfish)
  2. ปลาสินสมุทร (Blue ring angelfish)
  3. ปลาเขียวพระอินทร์ (Moon wrasse)
  4. ปลาสลิดน้ำเงิน (Sapphire devil) 
  5. ปลาผีเสื้อขอบทอง (Black-backed butterflyfish)
  6. ปลาพยาบาล (Bluesteak cleaner wrasse)
  7. ปลาผีเสื้อเกล็ดมุก (Threadfin butterflyfish)
  8. ปลาขี้ตังเบ็ดหัวเรียบ (Orangespine unicornfish)
  
โดยใช้จำนวนภาพทั้งหมด 10,000 ภาพ โดยแบ่งเป็นสายพันธุ์ละ 1,250 ภาพ และแบ่งเป็นภาพที่ใช้ในการทำแบบจำลอง 1,000 ภาพ ภาพที่ใช้ทดสอบ 250 ภาพ

## ขั้นตอน

### 1. ติดตั้งโปรแกรม
ในส่วนนี้จะทำการติดตั้งโปรแกรมต่างๆ ที่จำเป็นต้องใช้ในการทำ Tensorflow

#### 1a. ติดตั้ง Visual Studio
เข้าไปที่เว็บไซต์ [Visual Studio](https://visualstudio.microsoft.com/downloads/) และเลือกเวอร์ชันที่ต้องการ
#### 1b. ดาวน์โหลด Setup Script
ทำการดาวน์โหลดไฟล์ [tensorflow-gpu.yml](https://www.dropbox.com/s/c7krpl8cp510ath/tensorflow-gpu.yml?dl=1) ไว้ที่ไดร์ฟ C
#### 1c. ติดตั้ง Driver Graphics
เข้าไปที่เว็บไซต์ [Driver Graphics](https://www.nvidia.com/Download/index.aspx?lang=th) และเลือกเวอร์ชันไดร์เวอร์ให้ตรงกับ GPU ที่ใช้
#### 1d. ติดตั้ง CUDA 10.0
เข้าไปที่เว็บไซต์ [CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) โดยกดเลือกให้ตรงกับที่ต้องการ ดังรูปภาพ
![54dd1b6d53693fe6823973bf9ed32244.jpg](https://www.img.in.th/images/54dd1b6d53693fe6823973bf9ed32244.jpg) โดยการทำ Object Detection ครั้งนี้ ได้ใช้ CUDA เวอร์ชัน 10.0
#### 1e. ติดตั้ง cuDNN 7.6.5
เข้าไปที่เว็บไซต์ [cuDNN](https://developer.nvidia.com/cudnn) โดยกดเลือกให้ตรงกับเวอร์ชันของ [CUDA](https://github.com/pmgif/Fish-Recognition-Tutorial-Tensorflow-Faster-RCNN-windows10#1d-ติดตั้ง-cuda-100) ดังรูปภาพ
![734842c8688a364327c01c61176675d0.jpg](https://www.img.in.th/images/734842c8688a364327c01c61176675d0.jpg)
จากนั้นทำการ Set Path
```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin; 
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64; 
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include; 
SET PATH=C:\tools\cuda\bin;.
```
#### 1f. ติดตั้ง Anaconda 3.7
เข้าไปที่เว็บไซต์ [Anaconda](https://www.anaconda.com/distribution/) โดยเลือกเวอร์ชันที่ต้องการใช้ โดยการทำ Object Detection ครั้งนี้ ได้ใช้ Anaconda เวอร์ชัน 3.7 จากนั้นตั้งค่าให้ตรงกับภาพนี้
![67a92acda91b3d9ea3a97d9525060cae.jpg](https://www.img.in.th/images/67a92acda91b3d9ea3a97d9525060cae.jpg)
#### 1g. ติดตั้ง TensorRT 5.0 GA for Windows
เข้าไปที่เว็บไซต์ [TensorRT](https://developer.nvidia.com/tensorrt) โดยเลือกเวอร์ชันที่ต้องการใช้ และทำการ Set Path ที่อยู่ของโปรแกรม TensorRT
#### 1h. ติดตั้ง Miniconda3 Python 3.7 for Windows 64-bit
เข้าไปที่เว็บไซต์ [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) โดยเลือกเวอร์ชันที่ต้องการใช้ ทำการตั้งค่าให้ตรงกับภาพนี้
![87b212e455b1a03ea0d0adf1a79befc8.jpg](https://www.img.in.th/images/87b212e455b1a03ea0d0adf1a79befc8.jpg)
#### 1i. ติดตั้ง Python 3.7.6
เข้าไปที่เว็บไซต์ [Python](https://www.python.org/downloads/) โดยเลือกเวอร์ชันที่ต้องการใช้ โดยการทำ Object Detection ครั้งนี้ ได้ใช้ Python เวอร์ชัน 3.7.6 จากนั้นตั้งค่าให้ตรงกับภาพนี้
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
***
### 2. ตั้งค่า Tensorflow และ Anaconda
#### 2a. ดาวน์โหลด Tensorflow model
- สร้างแฟ้มข้อมูลในไดร์ฟ C: โดยตั้งชื่อว่า tensorflow
- ดาวน์โหลด full TensorFlow object detection จาก [models-master](https://github.com/tensorflow/models)
- เปลี่ยนชื่อแฟ้มข้อมูลจาก “models-master” เป็น “models” และนำแฟ้มข้อมูลดังกล่าว ย้ายมาไว้ในแฟ้มข้อมูล tensorflow
#### 2b. ดาวน์โหลด Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
สามารถเลือกโมเดลอย่างอื่นที่เราต้องการแทนได้ โดยไม่จำเป็นต้องใช้ Faster-RCNN-Inception-V2-COCO model เราสามารถเลือกใช้ให้ตรงกับจุดประสงค์ของเรา โดยสามารถเข้าไปดูได้ที่ [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models) แล้วทำการเลือกโมเดลที่เราต้องการ

หรือจะใช้ Faster-RCNN-Inception-V2-COCO model ก็ได้ โดยสามารถดาวน์โหลดได้โดยตรงจาก [Faster-RCNN](https://www.dropbox.com/s/nrp2xp3bk71zzje/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz?dl=1) จากนั้นย้ายแฟ้มข้อมูล faster_rcnn_inception_v2_coco_2018_01_28 มาไว้ใน C:\tensorflow\models\research\object_detection
# 2c. ดาวน์โหลด model จาก Github
ทำการดาวน์โหลด model เพิ่มเติมจาก GitHub นี้ จากนั้นย้ายข้อมูลในแฟ้มข้อมูล [ของกิ๊ฟ] ลงใน object_detection เนื่องจากมีไฟล์ซ้ำ ให้แทนที่ไฟล์ในแฟ้มข้อมูล จากนั้นทำตามขั้นตอนนี้
- ลบไฟล์ทุกไฟล์ในแฟ้มข้อมูล C:\tensorflow\models\research\object_detection\training
- ลบไฟล์ทุกไฟล์ในแฟ้มข้อมูล C:\tensorflow\models\research\object_detection\inference_graph
- ลบไฟล์ excel ในแฟ้มข้อมูล C:\tensorflow\models\research\object_detection\images
- ลบข้อมูลทั้งหมดในแฟ้มข้อมูล (หากต้องการใช้ข้อมูลของตนเอง  **หรือจะใช้ข้อมูลที่มีให้ก็ได้**)

  C:\tensorflow7\models\research\object_detection\images\test
  
  C:\tensorflow7\models\research\object_detection\images\train
***
### 3. เตรียมภาพ
เราจะใช้โปรแกรม LabelImg เวอร์ชัน 1.6 ในส่วนของการตีกรอบภาพ สามารถดาวน์โหลดได้จาก [LabelImg](https://www.dropbox.com/s/fc1e9b3jyy9udkm/windows_v1.6.0.zip?dl=1)นี้
![546fd110de02cb2d77b5d033fc888664.jpg](https://www.img.in.th/images/546fd110de02cb2d77b5d033fc888664.jpg)
เมื่อเราทำการตีกรอบภาพเสร็จแล้ว โปรแกรม LabelImg จะบันทึกเป็นไฟล์ .xml
***
### 4. สร้าง Anaconda virtual environment ใหม่
- เปิดโปรแกรม Anaconda Prompt โดยเลือกเป็น “Run as Administrator”
- พิมพ์คำสั่งเพื่อสร้าง virtual environment ขึ้นมาใหม่ โดยตั้งชื่อว่า tensorflow พร้อมกับติดตั้ง python เวอร์ชัน 3.7
```
C:\> conda create -n tensorflow pip python=3.7
```
- เปิดใช้งาน environment
```
C:\> activate tensorflow
```
- อัปเดทเวอร์ชันของ pip ให้เป็นเวอร์ชันล่าสุด
```
(tensorflow) C:\>python -m pip install --upgrade pip
```
- ติดตั้ง tensorflow-gpu เวอร์ชัน 1.15
*(เนื่องจากเวอร์ชัน tensorflow-gpu ที่ลงตอนติดตั้ง jupyter เป็นเวอร์ชัน 2.0.0 ไม่สามารถใช้ทดสอบได้ จึงเปลี่ยนเป็น tensorflow-gpu เวอร์ชัน 1.15 สามารถพิมพ์คำสั่งลงไปได้เลย เนื่องจากตัวโปรแกรมจะถอนการติดตั้ง tensorflow-gpu เวอร์ชัน 2.0.0 ให้อัตโนมัติ)*
```
(tensorflow) C:\> pip install --ignore-installed --upgrade tensorflow-gpu==1.15
```
- ติดตั้งแพคเกจย่อยอื่น ๆ
```
(tensorflow) C:\> conda install -c anaconda protobuf 
(tensorflow) C:\> pip install pillow 
(tensorflow) C:\> pip install lxml 
(tensorflow) C:\> pip install Cython 
(tensorflow) C:\> pip install contextlib2 
(tensorflow) C:\> pip install jupyter 
(tensorflow) C:\> pip install matplotlib 
(tensorflow) C:\> pip install pandas 
(tensorflow) C:\> pip install opencv-python
```
- กำหนด PYTHONPATH
```
(tensorflow) C:\> set PYTHONPATH=C:\tensorflow\models;C:\tensorflow\models\research;C:\tensorflow\models\research\slim
```
- สามารถตรวจสอบ PYTHONPATH ได้
```
(tensorflow) C:\> echo %PYTHONPATH%
```
- เปลี่ยน directories
```
(tensorflow) C:\> cd C:\tensorflow\models\research
```
- Tensorflow ใช้ Protobuf เพื่อกำหนดค่าแบบจำลองและพารามิเตอร์ของการฝึกสอนแบบจำลอง โดยสิ่งนี้จะสร้างไฟล์ name_pb2.py ไว้ในโฟลเดอร์ \ object_detection \ protos
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto.\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
- รวบรวมไฟล์ protoc แบบสั้นที่สำหรับการติดตั้ง Object Detection API ของ TensorFlow นั้น ไม่สามารถทำงานบน Windows ได้ ดังนั้นไฟล์ .proto ทุกไฟล์ในไดเร็กทอรี \ object_detection \ protos จึงต้องถูกเรียกใช้โดยคำสั่งทีละรายการ
```
(tensorflow) C:\tensorflow7\models\research> python setup.py build 
(tensorflow) C:\tensorflow7\models\research> python setup.py install
```
***
### 5. สร้างข้อมูลการฝึกอบรม
- เปลี่ยน directories
```
(tensorflow) C:\tensorflow\models\research> cd object_detection
```
- นำไฟล์ข้อมูลภาพ .xml ที่มีข้อมูลทั้งหมดในแฟ้มข้อมูล train และ test ไปแปลงไฟล์เพื่อสร้างไฟล์ .csv
```
(tensorflow) C:\tensorflow\models\research\object_detection> python xml_to_csv.py
```
***
### 6. สร้าง Label Map และ กำหนดค่าการฝึกอบรม
- แก้ไขไฟล์ generate_tfrecord.py จากโฟลเดอร์ \object_detection เพื่อระบุ label map ของตัวรูปแบบจำลอง
```
# TO-DO replace this with label map 
def class_text_to_int(row_label): 
  if row_label == 'Pennant coralfish': 
    return 1 
  elif row_label == 'Moon wrasse': 
    return 2 
  elif row_label == 'Sapphire devil': 
    return 3 
  elif row_label == 'Black-backed butterflyfish': 
    return 4 
  elif row_label == 'Blue ring angelfish':
    return 5 
  elif row_label == 'Threadfin butterflyfish': 
    return 6 
  elif row_label == 'Bluesteak cleaner wrasse': 
    return 7 
  elif row_label == 'Orangespine unicornfish': 
    return 8 
  else: 
    return 0
```
หรือสามารถเปลี่ยนให้เป็นตาม class ชนิดอื่น ๆ ที่ได้ทำการตีกรอบไว้ จาก format นี้
```
# TO-DO replace this with label map 
def class_text_to_int(row_label): 
  if row_label == 'Name1': 
    return 1 
  elif row_label == 'Name2': 
    return 2 
  else: 
    return 0
```
- สร้างไฟล์ TFRecord ของ train.record และ test.record ใน \ object_detection เพื่อใช้ในการฝึกสอนแบบจำลองการตรวจจับวัตถุใหม่
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
- สร้างไฟล์ labelmap.pbtxt ลงในโฟลเดอร์ C: \ tensorflow \ models \ research \ object_detection \ training เพื่อกำหนดว่าแต่ละวัตถุคืออะไร โดยกำหนดชื่อคลาสและหมายเลข ID คลาส ให้ตรงกับชื่อคลาสและหมายเลข ID ของไฟล์ generate_tfrecord.py
```
item {
  id: 1
  name: 'Pennant coralfish'
}

item {
  id: 2
  name: 'Moon wrasse'
}

item {
  id: 3
  name: 'Sapphire devil'
}

item {
  id: 4
  name: 'Black-backed butterflyfish'
}

item {
  id: 5
  name: 'Blue ring angelfish'
}

item {
  id: 6
  name: 'Threadfin butterflyfish'
}

item {
  id: 7
  name: 'Bluesteak cleaner wrasse'
}

item {
  id: 8
  name: 'Orangespine unicornfish'
}

```
หรือสามารถเปลี่ยนให้เป็นตาม class ชนิดอื่น ๆ จาก format นี้
```
item {
  id: 1
  name: 'Name1'
}

item {
  id: 2
  name: 'Name2'
}
```
- คัดลอกไฟล์ faster_rcnn_inception_v2_pets.config จาก C: \ tensorflow7 \ models \ research \ object_detection \ samples \ configs ไปยัง \ object_detection \ training จากนั้นเปิดไฟล์ด้วยโปรแกรม notepad++ และแก้ไข code ซึ่งมีการเปลี่ยนแปลงหลายอย่างในไฟล์ .config
  - บรรทัดที่ 9 เปลี่ยน num_classes (จำนวนของวัตถุต่าง ๆ ที่ต้องการให้ตัวแยกประเภทตรวจจับ) ในการฝึกสอนแบบจำลองมีจำนวนปลาที่ถูกนำมาฝึกสอนแบบจำลองทั้งหมด 8 สายพันธ์ุ
```
num_classes: 8
```
  - บรรทัดที่ 110 เปลี่ยนที่อยู่ของ fine_tune_checkpoint
```
fine_tune_checkpoint : "C:/tensorflow/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
```
  - บรรทัดที่ 116 เปลี่ยนจำนวนรอบการฝึกสอนแบบจำลอง
```
num_steps: 100000
```
  - บรรทัดที่ 126 และ 128 ในส่วน train_input_reader เปลี่ยน input_path และ label_map_path
```
input_path : "C:/tensorflow/models/research/object_detection/train.record"
```
```
label_map_path: "C:/tensorflow/models/research/object_detection/training/labelmap.pbtxt"
```
  - บรรทัดที่ 132 เปลี่ยน num_examples เป็นจำนวนภาพที่มีในแฟ้มข้อมูล \ images \ test
```
num_examples: 1600
```
  - บรรทัดที่ 140 และ 142 ในส่วน eval_input_reader เปลี่ยน input_path และ label_map_path
```
input_path : "C:/tensorflow/models/research/object_detection/test.record"
```
```
label_map_path: "C:/tensorflow/models/research/object_detection/training/labelmap.pbtxt"
```
***
### 7. ฝึกสอนแบบจำลอง
- ย้ายไฟล์ train.py ที่อยู่ในแฟ้มข้อมูล / object_detection / legacy ไปที่แฟ้มข้อมูล / object_detection
- พิมพ์คำสั่งเพื่อฝึกสอนแบบจำลอง
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
TensorFlow จะเริ่มต้นการฝึกสอนแบบจำลอง
![0fc668a9c52296de3bcb95af2a7109b4.jpg](https://www.img.in.th/images/0fc668a9c52296de3bcb95af2a7109b4.jpg)
- ระหว่างการฝึกสอนแบบจำลองสามารถดูความก้าวหน้าโดยใช้ TensorBoard โดยคำสั่งนี้จะต้องเปิดในอินสแตนซ์ใหม่ของ Anaconda Prompt และเปิดใช้งาน environment tensorflow เปลี่ยนเป็นไดเรกทอรี C: \ tensorflow \ models \ research \ object_detection และใช้คำสั่ง
```
(tensorflow7) C:\tensorflow7\models\research\object_detection>tensorboard --logdir=training
```
*โดย code ดังกล่าวจะสร้างเว็บเพจบนเครื่องคอมพิวเตอร์ YourPCName: 6006 สามารถดูได้ผ่านเว็บเบราว์เซอร์ หน้า TensorBoard จะให้ข้อมูลและกราฟที่แสดงว่าการฝึกสอนแบบจำลองมีความก้าวหน้าอย่างไร โดยกราฟนี้จะแสดงค่า loss ที่แสดงค่าความแม่นยำของแต่ละภาพในการฝึกสอนแบบจำลอง โดยแกน x หมายถึง ค่า Num Step (รอบการทำงาน) แกน y หมายถึง ค่า loss*
![27577961e6cb7f0da3ca7b90a7eb481b.jpg](https://www.img.in.th/images/27577961e6cb7f0da3ca7b90a7eb481b.jpg)
- เมื่อการฝึกสอนแบบจำลองเสร็จสิ้นตามจำนวนรอบการฝึกสอนแบบจำลอง จะแสดงข้อความ “Finished training! Saving model to disk.” หรือเราสามารถทำกรหยุดการฝึกสอนแบบจำลองเมื่อเราพอใจในค่า loss ได้ โดยทำการ *Ctrl+C*
![7042b525f9b9c531083b9a2e8cdd7b4e.jpg](https://www.img.in.th/images/7042b525f9b9c531083b9a2e8cdd7b4e.jpg)
***
### 8. Export Inference Graph
- สร้างไฟล์แบบจำลอง (ไฟล์ .pb) จากโฟลเดอร์ \ object_detection โดยที่ “model.ckpt-100000” คือหมายเลขสูงสุดของไฟล์ .ckpt ในโฟลเดอร์ \ training
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-100000 --output_directory inference_graph
```
***
### 9. ทดสอบแบบจำลอง
- เรียกใช้สคริปต์โดยพิมพ์ลงในพรอมต์คำสั่งของ Anaconda (เมื่อเปิดใช้งาน environment “tensorflow”) แล้วกด ENTER ซึ่งคำสั่งนี้จะเป็นการเปิด python shell เพื่อเรียกใช้งานสคริปต์ต่าง ๆ
```
(tensorflow) C:\tensorflow\models\research\object_detection> idle
```
ในขั้นตอนนี้เป็นการทดสอบแบบจำลอง โดยใช้ภาพปลาในการทดสอบสายพันธุ์ละ 250 ภาพ มีกระบวนการ ดังนี้
- เปิดไฟล์ Object_detection_image.py จากแฟ้มข้อมูล \ object_detection
- บรรทัดที่ 50 แก้ไขจำนวน num class ของไฟล์ Object_detection_image.py
```
NUM_CLASSES = 8
```
- บรรทัดที่ 34 แก้ไขชื่อภาพที่ต้องการทดสอบ เมื่อแก้ไขสำเร็จแล้ว กดปุ่ม F5 เพื่อประมวลผลโปรแกรม
```
IMAGE_NAME = 'test1.jpg'
```
- ตัวอย่างผลการประมวลผล code จากไฟล์ Object_detection_image.py สำเร็จ
[![cbadf04d8b69b3e3fa68758ce69c33c8.md.jpg](https://www.img.in.th/images/cbadf04d8b69b3e3fa68758ce69c33c8.md.jpg)](https://www.img.in.th/image/U8UAFO)
