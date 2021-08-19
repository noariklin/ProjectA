import scipy, numpy, shutil, os, nibabel
import os.path as osp
import sys, getopt
from PIL import Image
import imageio
import numpy as np

output_folder = 'C:\\Users\\97250\\Desktop\\ProjectA\\Anomaly\\images'
gt_output_folder = 'C:\\Users\\97250\\Desktop\\ProjectA\\Anomaly\\GT'

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
        



def seg2jpg(img_path,output_path=gt_output_folder):
    patient_name = osp.basename(img_path).split('.')[0].rsplit('_',1)[0]
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
        imageio.imwrite(img_output_path, img) 
    
    orig_img_path = img_path.replace('seg', 't1ce')
    nii2jpg(img_path=orig_img_path,good_index=good)
    
             
    
    
    


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('nii2png.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('nii2png.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg

    print('Input file is ', inputfile)
    print('Output folder is ', outputfile)

    # set fn as your 4d nifti file
    image_array = nibabel.load(inputfile).get_data()
    print(len(image_array.shape))

    # ask if rotate
    ask_rotate = input('Would you like to rotate the orientation? (y/n) ')

    if ask_rotate.lower() == 'y':
        ask_rotate_num = int(input('OK. By 90° 180° or 270°? '))
        if ask_rotate_num == 90 or ask_rotate_num == 180 or ask_rotate_num == 270:
            print('Got it. Your images will be rotated by {} degrees.'.format(ask_rotate_num))
        else:
            print('You must enter a value that is either 90, 180, or 270. Quitting...')
            sys.exit()
    elif ask_rotate.lower() == 'n':
        print('OK, Your images will be converted it as it is.')
    else:
        print('You must choose either y or n. Quitting...')
        sys.exit()

    # if 4D image inputted
    if len(image_array.shape) == 4:
        # set 4d array dimension values
        nx, ny, nz, nw = image_array.shape

        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            print("Created ouput directory: " + outputfile)

        print('Reading NIfTI file...')

        total_volumes = image_array.shape[3]
        total_slices = image_array.shape[2]

        # iterate through volumes
        for current_volume in range(0, total_volumes):
            slice_counter = 0
            # iterate through slices
            for current_slice in range(0, total_slices):
                if (slice_counter % 1) == 0:
                    # rotate or no rotate
                    if ask_rotate.lower() == 'y':
                        if ask_rotate_num == 90 or ask_rotate_num == 180 or ask_rotate_num == 270:
                            print('Rotating image...')
                            if ask_rotate_num == 90:
                                data = numpy.rot90(image_array[:, :, current_slice, current_volume])
                            elif ask_rotate_num == 180:
                                data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume]))
                            elif ask_rotate_num == 270:
                                data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume])))
                    elif ask_rotate.lower() == 'n':
                        data = image_array[:, :, current_slice, current_volume]
                            
                    #alternate slices and save as png
                    print('Saving image...')
                    image_name = inputfile[:-4] + "_t" + "{:0>3}".format(str(current_volume+1)) + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                    imageio.imwrite(image_name, data)
                    print('Saved.')

                    #move images to folder
                    print('Moving files...')
                    src = image_name
                    shutil.move(src, outputfile)
                    slice_counter += 1
                    print('Moved.')

        print('Finished converting images')

    # else if 3D image inputted
    elif len(image_array.shape) == 3:
        # set 4d array dimension values
        nx, ny, nz = image_array.shape

        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            print("Created ouput directory: " + outputfile)

        print('Reading NIfTI file...')

        total_slices = image_array.shape[2]

        slice_counter = 0
        # iterate through slices
        for current_slice in range(0, total_slices):
            # alternate slices
            if (slice_counter % 1) == 0:
                # rotate or no rotate
                if ask_rotate.lower() == 'y':
                    if ask_rotate_num == 90 or ask_rotate_num == 180 or ask_rotate_num == 270:
                        if ask_rotate_num == 90:
                            data = numpy.rot90(image_array[:, :, current_slice])
                        elif ask_rotate_num == 180:
                            data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice]))
                        elif ask_rotate_num == 270:
                            data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice])))
                elif ask_rotate.lower() == 'n':
                    data = image_array[:, :, current_slice]

                #alternate slices and save as png
                if (slice_counter % 1) == 0:
                    print('Saving image...')
                    image_name = inputfile[:-4] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                    imageio.imwrite(image_name, data)
                    print('Saved.')

                    #move images to folder
                    print('Moving image...')
                    src = image_name
                    shutil.move(src, outputfile)
                    slice_counter += 1
                    print('Moved.')

        print('Finished converting images')
    else:
        print('Not a 3D or 4D Image. Please try again.')

# call the function to start the program
if __name__ == "__main__":
    data_folder = r'C:\Users\97250\Desktop\ProjectA\Dataset\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    for root,_,files in os.walk(data_folder):
        seg_file = [f for f in files if f.endswith('seg.nii.gz')]
        if seg_file:
            seg2jpg(osp.join(root,seg_file[0]))
    
    # nii2jpg(img_path=r'C:\Users\97250\Desktop\ProjectA\Dataset\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii.gz')
    # seg2jpg(img_path=r'C:\Users\97250\Desktop\ProjectA\Dataset\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii.gz')
#    main(sys.argv[1:])