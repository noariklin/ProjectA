import scipy, numpy, shutil, os, nibabel
import os.path as osp
import sys, getopt
from PIL import Image
import imageio
import numpy as np
from skimage import img_as_ubyte
from skimage.segmentation import flood_fill, find_boundaries
from tqdm import tqdm
import matplotlib.pyplot as plt

output_folder = '/home/noas/NoaS/Anomaly/images/train'
gt_output_folder = '/home/noas/NoaS/Anomaly/GT'

def nii2jpg(img_path,good_index, output_path=output_folder):
    patient_name = osp.basename(img_path).split('.')[0].rsplit('_',1)[0]
    patient_good_output_folder = osp.join(output_path,patient_name, 'good')
    patient_bad_output_folder = osp.join(output_path,patient_name, 'bad')
    os.makedirs(patient_good_output_folder, exist_ok=True)
    os.makedirs(patient_bad_output_folder, exist_ok=True)
    
    image_array = nibabel.load(img_path).get_data()

    for current_slice in range(image_array.shape[2]):
        current_slice_data = image_array[8:-8, 8:-8, current_slice]
        if current_slice in good_index:
            img_output_path = osp.join(patient_good_output_folder, f'{patient_name}_{current_slice:05d}.png')
        else: 
            img_output_path = osp.join(patient_bad_output_folder, f'{patient_name}_{current_slice:05d}.png')
        imageio.imwrite(img_output_path, current_slice_data)
        
        
        
def create_good_imgs(img_path, output_path=output_folder):
    
    patient_name = osp.basename(img_path).split('.')[0].rsplit('_',1)[0]
    patient_good_output_folder = osp.join(output_path,patient_name, 'good')
    if not osp.exists(patient_good_output_folder):
        return 
    patient_bad_output_folder = osp.join(output_path,patient_name, 'bad')
    os.makedirs(patient_good_output_folder, exist_ok=True)
    os.makedirs(patient_bad_output_folder, exist_ok=True)
    
    image_array = nibabel.load(img_path).get_data()
    mask_array = nibabel.load(img_path.replace('t1ce', 'seg')).get_data()
    
    for current_slice in range(image_array.shape[2]):
        current_slice_data = image_array[8:-8, 8:-8, current_slice]
        h,w = current_slice_data.shape
        img = np.zeros([h,w,3])
        nom = np.max(current_slice_data.T)
        seg = mask_array[8:-8, 8:-8,current_slice]
        for x in range(w):
            for y in range(h):
                if seg[x,y]==4:
                    img[x,y, :]=[30,30,30] #white for et
                    current_slice_data[x,y]=30 #white for et
                # else:
                #     val = current_slice_data.T[y,x]
                #     current_slice_data[x,y, :]=[val,val, val]
        
        img_output_path = osp.join(patient_good_output_folder, f'{patient_name}_{current_slice:05d}.png')
        imageio.imwrite(img_output_path, current_slice_data)
        
        


def seg2jpg(img_path,output_path=gt_output_folder):
    
    patient_name = osp.basename(img_path).split('.')[0].rsplit('_',1)[0]
    if osp.exists(osp.join(output_folder,patient_name)):
        return 
    patient_good_output_folder = osp.join(output_path,patient_name, 'good')
    patient_bad_output_folder = osp.join(output_path,patient_name, 'bad')
    os.makedirs(patient_good_output_folder, exist_ok=True)
    os.makedirs(patient_bad_output_folder, exist_ok=True)
    image_array = nibabel.load(img_path).get_data()
    
    good = []
    
    for current_slice in range(image_array.shape[2]):
        current_slice_good = True
        current_slice_data = image_array[8:-8, 8:-8, current_slice]
        h,w = current_slice_data.shape
        img = np.zeros([h,w,3])
        for x in range(w):
            for y in range(h):
                if current_slice_data[x,y]==4:
                    img[x,y,:]=[255,255,255] #blue for et
                    current_slice_good = False

        if current_slice_good:
            good.append(current_slice)
            img_output_path = osp.join(patient_good_output_folder, f'{patient_name}_{current_slice:05d}.png')
        else:
            img_output_path = osp.join(patient_bad_output_folder, f'{patient_name}_{current_slice:05d}.png')
            # img_bound = np.asfarray(find_boundaries(img,mode='inner'))
            # img_fill_bounds = flood_fill(img_bound,some_x_y,new_value=1)
            # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
            # ax[0].imshow(img)
            # ax[1].imshow(img_fill_bounds)
            # plt.show()
            # a=111
        imageio.imwrite(img_output_path, img) 
    
    orig_img_path = img_path.replace('seg', 't1ce')
    nii2jpg(img_path=orig_img_path,good_index=good)


# call the function to start the program
if __name__ == "__main__":
    data_folder = '/home/noas/NoaS/dataset/Train'
    for root,_,files in os.walk(data_folder):
        segfile_list = [f for f in files if f.endswith('t1ce.nii.gz')]
        if segfile_list:
            # seg2jpg(osp.join(root,segfile_list[0]))
            create_good_imgs(osp.join(root,segfile_list[0]))
    
    # nii2jpg(img_path=r'C:\Users\97250\Desktop\ProjectA\Dataset\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii.gz')
    # seg2jpg(img_path=r'C:\Users\97250\Desktop\ProjectA\Dataset\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii.gz')
#    main(sys.argv[1:])