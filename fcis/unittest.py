import _init_paths
import os
import shutil
import mxnet as mx
import numpy as np

from symbols import *

###test symbol
instance_sym = resnext_50_fcis()
a = mx.symbol.Variable('data')
conv4 = instance_sym.get_resnext_conv4(a)
shape1 = conv4.infer_shape(data=(1,3,400,400))[1]
print("conv4 shape:{}".format(shape1))
conv5 = instance_sym.get_resnext_conv5(conv4)
shape2 = conv5.infer_shape(data=(1,3,400,400))[1]
print("conv5_shape:{}".format(shape2))
rpn_cls,rpn_bbox = instance_sym.get_rpn(conv4,9)
shape3 = rpn_cls.infer_shape(data=(1,3,400,400))[1]
shape4 = rpn_bbox.infer_shape(data=(1,3,400,400))[1]
print("rpn cls shape:{},rpn_bbox shape:{}".format(shape3,shape4))

