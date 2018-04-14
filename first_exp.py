
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



