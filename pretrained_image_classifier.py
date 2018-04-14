
# Loading the libraries 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

#Using CPU only mode 
caffe.set_mode_cpu()

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


