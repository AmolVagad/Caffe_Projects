
# Loading the libraries 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

#Using CPU only mode 
caffe.set_mode_cpu()

# Defining a network model with single convolution layer

kernel_size = 5
stride = 1
pad = 0
#output = (input - kernel_size) / stride +1
net = caffe.Net('conv.prototxt', caffe.TEST)

# net.blobs['data'] contains input data [array (1,1,100,100)] and net.blobs['conv'] contains computed data in layer conv [array (1,3,96,96)]

# To print the info 
[(k, v.data.shape) for k, v in net.blobs.items()]
[(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]
print net.blobs['conv'].data.shape


#Read input 
im = np.array(Image.open('/home/amol/caffe-1.0/examples/images/cat_gray.jpg'))
im_input = im[np.newaxis,np.newaxis,:,:]
#Reshaping the data blob from (1,1,100,100) to (1,1,360,480)
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

#Computing the blobs given this input 

net.forward();

#Now net.blobs['conv'] is filled with data, and the 3 pictures inside each of the 3 neurons (net.blobs['conv'].data[0,i]) can be plotted easily.

#Save the net paramaters 

net.save('myfirstmodel.caffemodel')


## Since the model is trained on processed images, need to preprocess the image with peprocessor before saving it in the blob

#load the model
net = caffe.Net('/home/amol/caffe-1.0/models/bvlc_reference_caffenet/deploy.prototxt',
                '/home/amol/caffe-1.0/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/amol/caffe-1.0/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
im = caffe.io.load_image('/home/amol/caffe-1.0/examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#compute
out = net.forward()

# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
print out['prob'].argmax()

#print predicted labels
labels = np.loadtxt("/home/amol/caffe-1.0/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-3:-1]

print labels[top_k]


