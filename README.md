# MobileNetv2-SSDLite
Caffe implementation of SSD detection on MobileNetv2, converted from tensorflow.<br>
**【As of June 3, 2018】 Be careful as it will not work with Intel Movidius Neural Compute Stick (NCS).**

I fixed a little clone of Chuanqi305/MobileNetv2-SSDLite.<br>
https://github.com/chuanqi305/MobileNetv2-SSDLite.git<br>
I have not confirmed the behavior of the generated model at all.<br>
Pull Request is welcomed.<br>

### Environment
Ubuntu 16.04<br>
Python 2.7.12<br>
Python 3.5.2<br>

### Prerequisites
Tensorflow and Caffe version [SSD](https://github.com/weiliu89/caffe) is properly installed on your computer.

### Procedure

pip command and installation of Tensorflow.

```
$ cd ~
$ sudo apt-get install python-pip python-dev python-launchpadlib   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev python3-launchpadlib # for Python 3.n
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
$ sudo python2 get-pip.py
$ sudo reboot
$ sudo -H pip install --upgrade tensorflow # Python 2.7; CPU support (no GPU support) Ver 1.8.0 [2018.06.03]
$ sudo -H pip3 install --upgrade tensorflow # Python 3.n; CPU support (no GPU support) Ver 1.8.0 [2018.06.03]
```

【Reference】Uninstall Tensorflow

```
$ sudo pip uninstall tensorflow  # for Python 2.7
$ sudo pip3 uninstall tensorflow # for Python 3.n
```

MobileNetv2-SSDLite Clone of learning data generation tool.

```
$ cd ~
$ git clone https://github.com/PINTO0309/MobileNetv2-SSDLite.git
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ tar -zxvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```

Execute the following command.

```
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ python dump_tensorflow_weights.py
```

【Reference】Confirm caffe installation path

```
$ sudo find / -name caffe
```

Execute the following command.

```
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ cp load_caffe_weights.py BK_load_caffe_weights.py
$ nano load_caffe_weights.py
```

```
#Before editing Load_caffe_weights.py

caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
for key in net.params.iterkeys():
print key
```

```
#After editing load_caffe_weights.py

caffe_root = '/path/to/your/ssd-caffe/'
[ex] caffe_root = '/opt/movidius/ssd-caffe/'

for key in net.params.keys():
print(key)
```

Execute the following command.

```
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ python3 load_caffe_weights.py
```

Learning data of the coco model is generated.<br>
1. ssdlite/deploy.caffemodel
2. ssdlite/deploy.prototxt


To generate learning data of the voc model, execute the following.

```
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ cp coco2voc.py BK_coco2voc.py
$ nano coco2voc.py
```

```
#Before editing coco2voc.py

caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
for key in net.params.iterkeys():
x = wt.shape[0] / 91
```

```
#After editing coco2voc.py

caffe_root = '/path/to/your/ssd-caffe/'
[ex] caffe_root = '/opt/movidius/ssd-caffe/'

for key in net.params.keys():
x = int(wt.shape[0] / 91)
```

Execute the following command.

```shell
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ python3 coco2voc.py
```

Learning data of the voc model is generated.
1. ssdlite/deploy_voc.caffemodel
2. ssdlite/voc/deploy.prototxt

<hr>
<hr>
<hr>
<hr>

### Usage
0. Firstly you should download the original model from [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
1. Use gen_model.py to generate the train.prototxt and deploy.prototxt (or use the default prototxt).
```
python gen_model.py -s deploy -c 91 >deploy.prototxt
```
2. Use dump_tensorflow_weights.py to dump the weights of conv layer and batchnorm layer.
3. Use load_caffe_weights.py to load the dumped weights to deploy.caffemodel.
4. Use the code in src to accelerate your training if you have a cudnn7, or add "engine: CAFFE" to your depthwise convolution layer to solve the memory issue.
5. The original tensorflow model is trained on MSCOCO dataset, maybe you need deploy.caffemodel for VOC dataset, use coco2voc.py to get deploy_voc.caffemodel.

### Train your own dataset
1. Generate the trainval_lmdb and test_lmdb from your dataset.
2. Write a labelmap.prototxt
3. Use gen_model.py to generate some prototxt files, replace the "CLASS_NUM" with class number of your own dataset.
```
python gen_model.py -s train -c CLASS_NUM >train.prototxt
python gen_model.py -s test -c CLASS_NUM >test.prototxt
python gen_model.py -s deploy -c CLASS_NUM >deploy.prototxt
```
4. Copy coco/solver_train.prototxt and coco/train.sh to your project and start training.

### Note
There are some differences between caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer. Replace ReLU6 with ReLU cause a bit accuracy drop in ssd-mobilenetv2, but very large drop in ssdlite-mobilenetv2. There is a ReLU6 layer implementation in my fork of [ssd](https://github.com/chuanqi305/ssd).


