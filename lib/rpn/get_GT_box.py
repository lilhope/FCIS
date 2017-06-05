import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from generate_anchor import generate_anchors

def get_base_box(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    _,contours,_ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    for contour in contours:
        x_min = np.min(contour[:,:,0])
        y_min = np.min(contour[:,:,1])
        x_max = np.max(contour[:,:,0])
        y_max = np.max(contour[:,:,1])
        bbox.append([x_min,y_min,x_max,y_max])
    return bbox
def generate_GT_bbox(base_box,ratios=[0.75,1,1.25],scales=np.array([2,4,8]),random_drift=True):
    GT_boxes = generate_anchors(base_box,ratios,scales)
    if random_drift:
        diameter = min(base_box[2]-base_box[0],base_box[3]-base_box[1])
        ds = scales * diameter
        for i in range(len(ratios)):
            for j in range(ds.shape[0]):
                GT_boxes[i*len(ratios)+j,:] += np.random.uniform(-0.5,0.5)*ds[j]/2
    return GT_boxes
def clean_GT_boxes(GT_boxes,height,width):
    for i in range(GT_boxes.shape[0]):
        GT_boxes[i,0] = max(GT_boxes[i,0],0)
        GT_boxes[i,1] = max(GT_boxes[i,1],0)
        GT_boxes[i,2] = min(GT_boxes[i,2],width-1)
        GT_boxes[i,3] = min(GT_boxes[i,3],height-1)
def get_GT_box(file_name):
    mask = cv2.imread(file_name)
    width = mask.shape[0]
    height = mask.shape[1]
    base_boxes = get_base_box(mask)
    GT_boxes = np.array([])
    for i in range(len(base_boxes)):
        base_box = np.array(base_boxes[i])
        GT_box = generate_GT_bbox(base_box)
        if i == 0:
            GT_boxes = GT_box.copy()
        else:
            GT_boxes = np.vstack((GT_boxes,GT_box))
    clean_GT_boxes(GT_boxes,height,width)
    return GT_boxes
            
    
            
    
if __name__ == "__main__":
    img = cv2.imread("/home/lilhope/FCIS/data/masks/CT/image_LKDS-00004_0147.jpg")
    ct = cv2.imread("/home/lilhope/FCIS/data/images/CT/image_LKDS-00004_0147.jpg")
    base_boxes = get_base_box(img)    
    plt.figure()
    ax = plt.gca()
    plt.imshow(ct)
    for i in range(len(base_boxes)):
        base_box = np.array(base_boxes[i])
        GT_boxes = generate_GT_bbox(base_box)
        print(GT_boxes)
        for j in range(GT_boxes.shape[0]):
            box = Rectangle((GT_boxes[j][0],GT_boxes[j][1]),GT_boxes[j][2]-GT_boxes[j][0]+1,GT_boxes[j][3]-GT_boxes[j][1]+1,fill=False,edgecolor='green', linewidth=1)
            ax.add_patch(box)
    plt.show()
    