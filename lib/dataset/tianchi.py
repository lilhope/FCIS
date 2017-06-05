import numpy
import cv2
import os
import cPickle
from rpn.get_GT_box import get_GT_box


class TianChi():
    def __init__(self,name,data_path,mode='train'):
        self.name = name
        self.data_path = data_path
        self.images_path  = os.path.join(data_path,"images/CT")
        self.masks_path = os.path.join(data_path,"masks/CT")
        self.file_list = os.listdir(self.images_path)
        self.cache_path = os.path.join(data_path,"cache")
        self.mode = mode
    def gt_sds_db(self):
        if self.mode == 'train':
            cache_file = os.path.join(self.cache_path,"train_cache.pkl")
        else:
            cache_file = os.path.join(self.cache_path,"test_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file,'rb') as f:
                sdsdb = cPickle.load(f)
        else:
            sbsdb = []
            for file_name in self.file_list:
                image_file = os.path.join(self.images_path,file_name)
                mask_file = os.path.join(self.masks_path,file_name)
                gt_boxes = get_GT_box(mask_file)
                datum = {"img_file":image_file,
                         'mask_file':mask_file,
                         'gt_boxes':gt_boxes,
                         'flipped':False}
                sbsdb.append(datum)
            with open(cache_file,'wb') as f:
                cPickle.dump(sbsdb,f)
        return sdsdb                