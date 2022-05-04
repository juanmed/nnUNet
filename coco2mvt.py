from pycocotools.coco import COCO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pylab
import cv2
import shutil
import os

image_directory = '/home/fer/repos/detectron2_instance_segmentation_demo/datasets/stent_defect_v2/'
annotation_file = '/home/fer/repos/detectron2_instance_segmentation_demo/datasets/stent_defect_v2/train.json'
example_coco = COCO(annotation_file)
good_image = 'C:\\Users\\lenovo\\Documents\\cursos\\dw_2022_1\\new_data\\good_all\\'

categories = example_coco.loadCats(example_coco.getCatIds())
#print("Categories:", categories)
category_names = [category['name'] for category in categories]
#print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
print(category_names)

#category_names = set([category['supercategory'] for category in categories])
#print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=[])
print("Cat IDS: {}".format(category_ids))

cat_map = {}
for k, (n, ids) in enumerate(zip(category_names, category_ids)):
    cat_map[ids] = n
    
category_ids = example_coco.getCatIds(catNms=['circle'])    
image_ids = example_coco.getImgIds(catIds=category_ids)
print("Total images: ",len(image_ids))

print(cat_map)

output_dir = "./stent_defect_v2_mvt/"
test_dir_good = os.path.join(output_dir,"test","good")
test_dir_def = os.path.join(output_dir,"test","defective")

train_dir_good = os.path.join(output_dir,"train","good")
gt_dir = os.path.join(output_dir,"ground_truth","defective")
os.makedirs(test_dir_good, exist_ok = True)
os.makedirs(test_dir_def, exist_ok = True)
os.makedirs(train_dir_good, exist_ok = True)
os.makedirs(gt_dir, exist_ok = True)


total_anns = 0
num_anns = []
for n ,image_id in enumerate(image_ids[:]):
    image_data = example_coco.loadImgs(image_id)[0]
    #print("Currently viewing: {}".format(image_directory + image_data['file_name']))
    try:
        image = cv2.imread(image_directory + image_data['file_name'], cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("\nImage load Problem:  image_id {}, {}".format(image_id, image_directory + image_data['file_name']))
        print(str(e))
        continue

    height, width, _ = image.shape
    if ( image.shape[0]>0) and (image.shape[1] > 0) and (image is not None):
        pass
    else:
        print( " ##### ESTA IMAGEN NO ESTA ######")
        continue

    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    total_anns += len(annotation_ids)
    assert len(annotation_ids) > 0, "{} has no annotations".format(image_data['file_name'])
    num_anns.append(len(annotation_ids))
    #annotations = example_coco.loadAnns(annotation_ids)
    target = [x for x in example_coco.loadAnns(annotation_ids) if x['image_id'] == image_id]
       
    total_mask = np.zeros((height, width), dtype=np.uint8)   
    for obj in target:
        seg_mask = example_coco.annToMask(obj).reshape(height, width)
        total_mask[seg_mask > 0] = int(obj['category_id'])

    print("Values: ",np.unique(total_mask))
    
    #try:
    #    masks = np.vstack(masks)
    #    masks = masks.reshape(-1, height, width)
    #except Exception as e:
    #    print("\nMask stack Problem:  image_id {}, {}".format(image_id, image_directory + image_data['file_name']))
    #    print(str(e))
    #    continue

    #mask = np.zeros_like(image.shape[0:2])
    #for ann in target:
    #    cv2.drawContours(mask, ann["segmentation"], -1, 255, cv2.FILLED)

    # write mask
    output_file = os.path.join(gt_dir,"{:03.0f}.png".format(n))
    #print(">>",output_file)
    cv2.imwrite(output_file,total_mask)

    # write defective test
    output_file = os.path.join(test_dir_def,"{:03.0f}.png".format(n))
    #print(">>",output_file)
    cv2.imwrite(output_file,image)


    #plt.imshow(cv2.resize(total_mask/np.max(total_mask),(1024,1024)))
    #plt.show()

"""
good = os.listdir(good_image)
train_good = good[:len(good)//2]
test_good = good[len(good)//2:]

for n, image_name in enumerate(train_good):
    image = cv2.imread(os.path.join(good_image, image_name), cv2.IMREAD_UNCHANGED)
    height, width, _ = image.shape
    if ( image.shape[0]>0) and (image.shape[1] > 0) and (image is not None):
        pass
    else:
        print( " ##### ESTA IMAGEN NO ESTA ######")
        continue
    output_file = os.path.join(train_dir_good,"{:03.0f}.png".format(n))
    #print(">>",output_file)
    cv2.imwrite(output_file,cv2.resize(image,(1024,1024),interpolation=cv2.INTER_NEAREST))    

for n, image_name in enumerate(test_good):
    image = cv2.imread(os.path.join(good_image, image_name), cv2.IMREAD_UNCHANGED)
    height, width, _ = image.shape
    if ( image.shape[0]>0) and (image.shape[1] > 0) and (image is not None):
        pass
    else:
        print( " ##### ESTA IMAGEN NO ESTA ######")
        continue
    output_file = os.path.join(test_dir_good,"{:03.0f}.png".format(n))
    #print(">>",output_file)
    cv2.imwrite(output_file,cv2.resize(image,(1024,1024),interpolation=cv2.INTER_NEAREST))    
"""
print("Total annotations", total_anns)