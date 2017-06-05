# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haozhi Qi, Guodong Zhang, Yi Li
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_annotator import *
from operator_py.box_parser import *
from operator_py.box_annotator_ohem import *


class resnext_50_fcis(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-4
        self.use_global_stats = True
        self.workspace = 256
        self.bn_mom=0.9
        self.units = [3, 4, 6, 3] # use for resnetx 101
        self.filter_list = [64,128,256,512] #use less channels
        self.bottle_neck = True
    def redusial_unit(self,data,num_filter,stride,dim_match,name,num_group=32,memonger=False):
        if self.bottle_neck:
            conv1 = mx.symbol.Convolution(data=data,num_filter=(int(0.5*num_filter)),kernel=(1,1),stride=(1,1),pad=(0,0),
                                          no_bias=True,workspace=self.workspace,name=name+'_conv1')
            bn1 = mx.symbol.BatchNorm(data=conv1,fix_gamma=False,eps=self.eps,momentum=self.bn_mom,name=name+'_bn1')
            act1 = mx.symbol.Activation(data=bn1,act_type='relu',name=name + '_relu1')
            
            conv2 = mx.symbol.Convolution(data=act1,num_filter=(int(0.5*num_filter)),num_group=num_group,kernel=(3,3),stride=stride,
                                          pad=(1,1),no_bias=True,workspace=self.workspace,name=name+'_conv2')
            bn2 = mx.symbol.BatchNorm(data=conv2,fix_gamma=False,eps=self.eps,momentum=self.bn_mom,name=name+'_bn2')
            act2 = mx.symbol.Activation(data=bn2,act_type='relu',name=name+'_relu2')
            
            conv3 = mx.symbol.Convolution(data=act2,num_filter=num_filter,kernel=(1,1),stride=(1,1),pad=(0,0),no_bias=True,
                                          workspace=self.workspace,name=name+'_conv3')
            bn3 = mx.symbol.BatchNorm(data=conv3,fix_gamma=False,eps=self.eps,momentum=self.bn_mom,name=name + '_bn3')
            if dim_match:
                short_cut = data
            else:
                short_cut = mx.sym.Convolution(data=data,num_filter=num_filter,kernel=(1,1),stride=stride,no_bias=True,
                                               workspace=self.workspace,name=name+'_sc')
                short_cut = mx.symbol.BatchNorm(data=short_cut,fix_gamma=False,eps=self.eps,momentum=self.bn_mom,name=name+'_sc_bn')
            if memonger:
                short_cut._set_attr(mirror_stage='True')
            eltwise = short_cut + bn3
            return mx.symbol.Activation(data=eltwise,act_type='relu',name=name+'_relu')
        else:
            conv1 = mx.symbol.Convolution(data=data,num_filter=num_filter,kernel=(3,3),stride=stride,pad=(1,1),
                                          no_bias=True,workspace=self.workspace,name= name+'_conv1')
            bn1 = mx.symbol.BatchNorm(data=conv1,fix_gamma=False,eps=self.eps,momentum=self.bn_mom,name = name+'_bn1')
            act1 = mx.symbol.Activation(data=bn1,act_type='relu',name=name+'_relu1')
            
            conv2 = mx.symbol.Convolution(data=act1,num_filter=num_filter,kernel=(3,3),stride=(1,1),pad=(1,1),
                                          no_bias=True,workspace=self.workspace,name=name+'_conv2')
            bn2 = mx.symbol.BatchNorm(data=conv2,fix_gamma=False,eps=self.eps,momentum=self.bn_mom,name=name+'_bn2')
            if dim_match:
                short_cut = data
            else:
                short_cut = mx.symbol.Convolution(data=data,num_filter=num_filter,kernel=(1,1),stride=stride,no_bias=True,
                                                  workspace=self.workspace,name=name+'_sc')
                short_cut = mx.symbol.BatchNorm(data=short_cut,fix_gamma=True,eps=self.eps,momentum=self.bn_mom,name = name+'_sc_bn')
            if memonger:
                short_cut._set_attr(mirror_stage='True')
            eltwise = short_cut + bn2
            return mx.symbol.Activation(data=eltwise,act_type='relu',name=name+'_relu')
        

    def get_resnext_conv4(self, data):
        #res1 different from the offical implement,we doesn't do spatial scale here
        body = mx.symbol.Convolution(data=data,num_filter=64,kernel=(3,3),stride=(1,1),pad=(1,1),
                                     no_bias = True,workspace=self.workspace,name="conv0")
        bn0 = mx.symbol.BatchNorm(data=body,fix_gamma=False,eps=self.eps,name='bn0')
        act0 = mx.symbol.Activation(data=bn0,act_type='relu',name='relu0')
        pool0 = mx.symbol.Pooling(data=act0,kernel=(3,3),stride=(2,2),pad=(1,1),pool_type='max',name="pool0")
        #res2
        body = self.redusial_unit(data=pool0,num_filter=self.filter_list[0],stride=(1,1),
                                  dim_match=False,name='stage2_unit1')
        for i in range(2,self.units[0]+1):
            body = self.redusial_unit(data=body,num_filter=self.filter_list[0],stride=(1,1),
                                      dim_match=True,name='stage2_unit%s' % i)
        #res3
        body = self.redusial_unit(data=body,num_filter=self.filter_list[1],stride=(2,2),
                                  dim_match=False,name='stage3_unit1')
        for i in range(2,self.units[1]+1):
            body = self.redusial_unit(data=body,num_filter=self.filter_list[1],stride=(1,1),
                                      dim_match=True,name='stage3_unit%s' % i)
        #res4
        body = self.redusial_unit(data=body,num_filter=self.filter_list[2],stride=(2,2),
                                  dim_match=False,name='stage4_unit1')
        for i in range(2,self.units[2]+1):
            body = self.redusial_unit(data=body,num_filter=self.filter_list[2],stride=(1,1),
                                      dim_match=True,name='stage4_unit%s' % i)
        return body
             
    def get_resnext_conv5(self,conv_feat):
        ##hole algorithm
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=128, pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=128, pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=128, pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu
        
        

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), stride=(2,2),pad=(1, 1), num_filter=256, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS
        # input init
        if is_train:
            data = mx.sym.Variable(name='data')
            im_info = mx.sym.Variable(name='im_info')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            gt_masks = mx.sym.Variable(name='gt_masks')
            rpn_label = mx.sym.Variable(name='proposal_label')
            rpn_bbox_target = mx.sym.Variable(name='proposal_bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='proposal_bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_resnext_conv4(data)
        # res5
        relu1 = self.get_resnext_conv5(conv_feat)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    nms_threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            group = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, gt_masks=gt_masks,
                                  op_type='proposal_annotator',
                                  num_classes=num_reg_classes, mask_size=cfg.MASK_SIZE, binary_thresh=cfg.TRAIN.BINARY_THRESH,
                                  batch_images=cfg.TRAIN.BATCH_IMAGES, cfg=cPickle.dumps(cfg),
                                  batch_rois=cfg.TRAIN.BATCH_ROIS, fg_fraction=cfg.TRAIN.FG_FRACTION)
            rois = group[0]
            label = group[1]
            bbox_target = group[2]
            bbox_weight = group[3]
            mask_reg_targets = group[4]
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    nms_threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)


        # conv new 1
        if cfg.TRAIN.CONVNEW3:
            conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name='conv_new_1', attr={'lr_mult':'3.00'})
        else:
            conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name='conv_new_1')
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')

        fcis_cls_seg = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes*2,
                                          name='fcis_cls_seg')
        fcis_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes,
                                       name='fcis_bbox')

        psroipool_cls_seg = mx.contrib.sym.PSROIPooling(name='psroipool_cls_seg', data=fcis_cls_seg, rois=rois,
                                                        group_size=7, pooled_size=21, output_dim=num_classes*2, spatial_scale=0.125)
        psroipool_bbox_pred = mx.contrib.sym.PSROIPooling(name='psroipool_bbox', data=fcis_bbox, rois=rois,
                                                          group_size=7, pooled_size=21,  output_dim=num_reg_classes*4, spatial_scale=0.125)
        if is_train:
            # classification path
            psroipool_cls = mx.contrib.sym.ChannelOperator(name='psroipool_cls', data=psroipool_cls_seg, group=num_classes, op_type='Group_Max')
            cls_score = mx.sym.Pooling(name='cls_score', data=psroipool_cls, pool_type='avg', global_pool=True, kernel=(21, 21))
            cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
            # mask regression path
            label_seg = mx.sym.Reshape(name='label_seg', data=label, shape=(-1, 1, 1, 1))
            seg_pred = mx.contrib.sym.ChannelOperator(name='seg_pred', data=psroipool_cls_seg, pick_idx=label_seg, group=num_classes, op_type='Group_Pick', pick_type='Label_Pick')
            # bbox regression path
            bbox_pred = mx.sym.Pooling(name='bbox_pred', data=psroipool_bbox_pred, pool_type='avg', global_pool=True, kernel=(21, 21))
            bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))
        else:
            # classification path
            psroipool_cls = mx.contrib.sym.ChannelOperator(name='psroipool_cls', data=psroipool_cls_seg, group=num_classes, op_type='Group_Max')
            cls_score = mx.sym.Pooling(name='cls_score', data=psroipool_cls, pool_type='avg', global_pool=True,
                                       kernel=(21, 21))
            cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            # mask regression path
            score_seg = mx.sym.Reshape(name='score_seg', data=cls_prob, shape=(-1, num_classes, 1, 1))
            seg_softmax = mx.contrib.sym.ChannelOperator(name='seg_softmax', data=psroipool_cls_seg, group=num_classes, op_type='Group_Softmax')
            seg_pred = mx.contrib.sym.ChannelOperator(name='seg_pred', data=seg_softmax, pick_idx=score_seg, group=num_classes, op_type='Group_Pick', pick_type='Score_Pick')
            # bbox regression path
            bbox_pred = mx.sym.Pooling(name='bbox_pred', data=psroipool_bbox_pred, pool_type='avg', global_pool=True,
                                       kernel=(21, 21))
            bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, mask_targets_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM, cfg=cPickle.dumps(cfg),
                                                cls_score=cls_score, seg_pred=seg_pred, bbox_pred=bbox_pred, labels=label,
                                                mask_targets=mask_reg_targets, bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid',
                                                use_ignore=True, ignore_label=-1, grad_scale=cfg.TRAIN.LOSS_WEIGHT[0])
                seg_prob = mx.sym.SoftmaxOutput(name='seg_prob', data=seg_pred, label=mask_targets_ohem, multi_output=True,
                                                normalization='null', use_ignore=True, ignore_label=-1,
                                                grad_scale=cfg.TRAIN.LOSS_WEIGHT[1] / cfg.TRAIN.BATCH_ROIS_OHEM)
                bbox_loss_t = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_t', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_t, grad_scale=cfg.TRAIN.LOSS_WEIGHT[2] / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                                use_ignore=True, ignore_label=-1, grad_scale=cfg.TRAIN.LOSS_WEIGHT[0])
                seg_prob = mx.sym.SoftmaxOutput(name='seg_prob', data=seg_pred, label=mask_reg_targets, multi_output=True,
                                                normalization='null', use_ignore=True, ignore_label=-1,
                                                grad_scale=cfg.TRAIN.LOSS_WEIGHT[1] / cfg.TRAIN.BATCH_ROIS)
                bbox_loss_t = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_t', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_t, grad_scale=cfg.TRAIN.LOSS_WEIGHT[2] / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, seg_prob, mx.sym.BlockGrad(mask_reg_targets), mx.sym.BlockGrad(rcnn_label)])
            #group = mx.sym.Group([rpn_cls_prob,rpn_bbox_loss,cls_score,seg_pred])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            if cfg.TEST.ITER == 2:
                rois_iter2 = mx.sym.Custom(bottom_rois=rois, bbox_delta=bbox_pred, im_info=im_info, cls_prob=cls_prob,
                                           name='rois_iter2', b_clip_boxes=True, bbox_class_agnostic=True,
                                           bbox_means=tuple(cfg.TRAIN.BBOX_MEANS), bbox_stds=tuple(cfg.TRAIN.BBOX_STDS), op_type='BoxParser')
                # rois = mx.sym.Concat(*[rois, rois_iter2], dim=0, name='rois')
                psroipool_cls_seg_iter2 = mx.contrib.sym.PSROIPooling(name='psroipool_cls_seg', data=fcis_cls_seg, rois=rois_iter2,
                                                              group_size=7, pooled_size=21,
                                                              output_dim=num_classes*2, spatial_scale=0.0625)
                psroipool_bbox_pred_iter2 = mx.contrib.sym.PSROIPooling(name='psroipool_bbox', data=fcis_bbox, rois=rois_iter2,
                                                                group_size=7, pooled_size=21,
                                                                output_dim=num_reg_classes*4, spatial_scale=0.0625)

                # classification path
                psroipool_cls_iter2 = mx.contrib.sym.ChannelOperator(name='psroipool_cls', data=psroipool_cls_seg_iter2, group=num_classes,
                                                             op_type='Group_Max')
                cls_score_iter2 = mx.sym.Pooling(name='cls_score', data=psroipool_cls_iter2, pool_type='avg', global_pool=True, kernel=(21, 21), stride=(21,21))

                cls_score_iter2 = mx.sym.Reshape(name='cls_score_reshape', data=cls_score_iter2, shape=(-1, num_classes))
                cls_prob_iter2 = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score_iter2)
                # mask regression path
                score_seg_iter2 = mx.sym.Reshape(name='score_seg', data=cls_prob_iter2, shape=(-1, num_classes, 1, 1))
                seg_softmax_iter2 = mx.contrib.sym.ChannelOperator(name='seg_softmax', data=psroipool_cls_seg_iter2, group=num_classes, op_type='Group_Softmax')
                seg_pred_iter2 = mx.contrib.sym.ChannelOperator(name='seg_pred', data=seg_softmax_iter2, pick_idx=score_seg_iter2, group=num_classes, op_type='Group_Pick', pick_type='Score_Pick')
                # bbox regression path
                bbox_pred_iter2 = mx.sym.Pooling(name='bbox_pred', data=psroipool_bbox_pred_iter2, pool_type='avg', global_pool=True, kernel=(21, 21), stride=(21,21))
                bbox_pred_iter2 = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred_iter2, shape=(-1, 4 * num_reg_classes))

                rois = mx.sym.Concat(*[rois, rois_iter2], dim=0, name='rois')
                cls_prob = mx.sym.Concat(*[cls_prob, cls_prob_iter2], dim=0, name='cls_prob')
                seg_pred = mx.sym.Concat(*[seg_pred, seg_pred_iter2], dim=0, name='seg_pred')
                bbox_pred = mx.sym.Concat(*[bbox_pred, bbox_pred_iter2], dim=0, name='box_pred')
            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred, seg_pred])

        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fcis_cls_seg_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fcis_cls_seg_weight'])
        arg_params['fcis_cls_seg_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fcis_cls_seg_bias'])
        arg_params['fcis_bbox_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fcis_bbox_weight'])
        arg_params['fcis_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fcis_bbox_bias'])

if __name__=="__main__":
    fcis_sym = resnext_50_fcis()
    a = mx.sym.Variable('data')
    conv4 = fcis_sym.get_resnext_conv4(a)
    shape1 = conv4.infer_shape(data=(1,3,400,400))[1]
    print(shape1)

