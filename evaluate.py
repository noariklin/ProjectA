import nibabel as nib 
import os
import numpy as np
import argparse
import medpy.metric.binary as medpyMetrics
import matplotlib.pyplot as plt

gtpath = '/home/noas/NoaS/dataset/Val/BraTS20_Training_337'
exps={'base':'/home/noas/NoaS/ProjectA/PartiallyReversibleUnet/prediction/best/'}
# exps={'base':'/home/noas/NoaS/dataset/Test/BraTS20_Training_306'}

def getNSD(pred,target):
    surDist1 = medpyMetrics.__surface_distances(pred, target)
    surDist2 = medpyMetrics.__surface_distances(target, pred)
    hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
    return hd95
    
def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum()
    union = (pred).sum() + (target).sum()
    dice = (2 * intersection) / (union)
    return dice.mean()


def softJI(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum()
    union = (pred).sum() + (target).sum()
    dice = (intersection) / (union-intersection)
    return dice.mean()

def getnames(pdpath):
    pds = os.listdir(pdpath)
    names=[]
    for pd in pds:
        name = os.path.splitext(os.path.splitext(pd)[0])[0]
        names.append(name)
    return names

def trans2gtshape(t,pred):
    s = pred.shape
    if len(s)>len(t):
        pred = pred.squeeze()
    newp = np.zeros(t)
    newp[:min(s[0],t[0]),:min(s[1],t[1]),:min(s[2],t[2])] = pred[:min(s[0],t[0]),:min(s[1],t[1]),:min(s[2],t[2])]
    return newp


def cal_single_dice(gt,pd):
    A = pd == 1
    P = pd == 2
    Amask = gt == 1
    Pmask = gt == 2
    Adice = softDice(A, Amask, 0, True)
    Pdice = softDice(P, Pmask, 0, True)
    return Adice,Pdice


def cal_single_NSD(gt,pd):
    A = pd == 1
    P = pd == 2
    Amask = gt == 1
    Pmask = gt == 2
    Adice = getNSD(A, Amask)
    Pdice = getNSD(P, Pmask)
    return Adice,Pdice


def cal_single_JI(gt,pd):
    A = pd == 1
    P = pd == 2
    Amask = gt == 1
    Pmask = gt == 2
    Adice = softJI(A, Amask, 0, True)
    Pdice = softJI(P, Pmask, 0, True)
    return Adice,Pdice


def cal_all_dice(gtpath,pdpath,names):
    diceA,diceP,avg=[],[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_dice(gt,pd)
        diceA.append(A)
        diceP.append(P)
        avg.append((A+P)/2)
    return diceA,diceP,avg


def cal_all_JI(gtpath,pdpath,names):
    diceA,diceP,avg=[],[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_JI(gt,pd)
        diceA.append(A)
        diceP.append(P)
        avg.append((A+P)/2)
    return diceA,diceP,avg


def cal_all_NSD(gtpath,pdpath,names):
    diceA,diceP,avg=[],[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_NSD(gt,pd)
        diceA.append(A)
        diceP.append(P)
        avg.append((A+P)/2)
    return diceA,diceP,avg

def show_slices(slices,segs):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        h,w = slice.T.shape
        nom = np.max(slice.T)
        seg = segs[i]
        img = np.zeros([h,w,3])
        for x in range(w):
            for y in range(h):
                if seg[x,y]==1:
                    img[y,x,:]=[255,0,0]#red for tc
                elif seg[x,y]==2:
                    img[y,x,:]=[0,255,0]#green for wt
                elif seg[x,y]==4:
                    img[y,x,:]=[0,0,255]#blue for et
                else:
                    val = slice.T[y,x]/nom
                    img[y,x,:]=[val,val,val]
        axes[i].imshow(img, origin="lower")
    plt.show()


def load_exp_img(gt,exp='base',name='BraTS20_Training_306'):
    img = nib.load(os.path.join(gtpath,name+'_t2.nii.gz'))
    # label = nib.load(os.path.join(exps[exp],name+'.nii.gz'))
    label = nib.load(os.path.join(exps[exp],name+'.nii.gz'))
    data = img.get_fdata()
    seg = label.get_fdata()
    h,w,d = data.shape
    seg = trans2gtshape([h,w,d],seg)
    slices = [data[h//2,:,:],data[:,w//2,:],data[:,:,d//2]]
    segs = [seg[h//2,:,:],seg[:,w//2,:],seg[:,:,d//2]]
    show_slices(slices,segs)
    return cal_single_dice(gt,seg)


def predict(imgpath,gtpath):
    names=[os.path.basename(imgpath)]
    exp_names = ['base']
    for name in names:
        img = nib.load(os.path.join(imgpath,name+'_t2.nii.gz'))
        label = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        data = img.get_fdata()
        gt = label.get_fdata()
        h,w,d = data.shape
        slices = [data[h//2,:,:],data[:,w//2,:],data[:,:,d//2]]
        segs = [gt[h//2,:,:],gt[:,w//2,:],gt[:,:,d//2]]
        show_slices(slices,segs)
        for exp in exp_names:
            print(exp)
            res=load_exp_img(gt,exp,name)
            print(res)



def main(args):
    pdpath = exps[args.exp]
    names = getnames(pdpath)
    diceA,diceP,avg=cal_all_dice(gtpath,pdpath,names)
    val1=np.mean(diceA)
    val2=np.mean(diceP)
    val = np.mean(avg)
    print("{}\t{}\t{}".format(val*100,val1*100,val2*100))
    diceA,diceP,avg=cal_all_NSD(gtpath,pdpath,names)
    val1=np.mean(diceA)
    val2=np.mean(diceP)
    val = np.mean(avg)
    print("{}\t{}\t{}".format(val,val1,val2))
    diceA,diceP,avg=cal_all_JI(gtpath,pdpath,names)
    val1=np.mean(diceA)
    val2=np.mean(diceP)
    val = np.mean(avg)
    print("{}\t{}\t{}".format(val,val1,val2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRU Training')
    parser.add_argument('--exp', default='base', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    # main(parser.parse_args())
    predict(imgpath=gtpath, gtpath=gtpath)



