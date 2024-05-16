import numpy as np
import os
from glob import glob
import cv2
import json
from scipy.ndimage.filters import gaussian_filter
import xml.etree.ElementTree as ET


def get_gt_dots(mat_path):
    """
    Load Matlab file with ground truth labels and save it to numpy array.
    ** cliping is needed to prevent going out of the array
    """
    tree = ET.ElementTree(file=mat_path)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    gt = []
    for Object in ObjectSet:
        try:
            Point = Object.find('point')
            x = int(Point.find('x').text)
            y = int(Point.find('y').text)
        except:
            BndBox = Object.find('bndbox')
            x = int(BndBox.find('xmin').text)
            y = int(BndBox.find('ymin').text)
        gt.append([x, y])

    gt = np.array(gt).astype(np.float32).round().astype(int)
    return gt
    
    
def generate_data(label_path):
    rgb_path = label_path.replace('GT_', 'RGB').replace('R.xml', '.jpg')
    t_path = label_path.replace('GT_', 'Infrared').replace('R.xml', 'R.jpg')
    rgb = cv2.imread(rgb_path)
    t = cv2.imread(t_path)
    im_h, im_w, _ = rgb.shape
    # print('rgb and t shape', rgb.shape, t.shape)
    points = get_gt_dots(label_path)
    # print('points', points.shape)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    return rgb, t, points


if __name__ == '__main__':

    root_path = "./data/DroneRGBT/"  # dataset root path
    save_dir = "./data/processedDroneRGBT/"

    for phase in ['train', 'val']:
        if phase in ['train']:
            sub_dir = os.path.join(root_path, 'Train')
        if phase in ['val']:
            sub_dir = os.path.join(root_path, 'Test')
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        gt_list = glob(os.path.join(sub_dir+'/GT_', '*xml'))
        for gt_path in gt_list:
            name = os.path.basename(gt_path)
            print(name)
            rgb, t, points = generate_data(gt_path)
            im_save_path = os.path.join(sub_save_dir, name)
            rgb_save_path = im_save_path.replace('R.xml', '_RGB.jpg')
            t_save_path = im_save_path.replace('R.xml', '_T.jpg')
            cv2.imwrite(rgb_save_path, rgb)
            cv2.imwrite(t_save_path, t)
            gd_save_path = im_save_path.replace('R.xml', '_GT.npy')
            np.save(gd_save_path, points)
            
            density = np.zeros((rgb.shape[0],rgb.shape[1]))
            for i in range(0,len(points)):
                if int(points[i][1])<rgb.shape[0] and int(points[i][0])<rgb.shape[1]:
                    density[int(points[i][1]),int(points[i][0])]=1
            density = gaussian_filter(density, 8)
            density_save_path = gd_save_path.replace('GT', 'density')
            np.save(density_save_path, density)

