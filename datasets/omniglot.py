# adapted from Brandon Lake
import numpy as np
import os,time,random
from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
	matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from PIL import Image,ImageOps

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as T

from sklearn.cluster import OPTICS,AgglomerativeClustering
from sklearn.manifold import TSNE 
# import seaborn as sns



# convert to str and add leading zero to single digit numbers
def num2str(idx):
	if idx < 10:
		return '0'+str(idx)
	return str(idx)

# Load stroke data for a character from text file
#
# Input
#   fn : filename
#
# Output
#   motor : list of strokes (each is a [n x 3] numpy array)
#      first two columns are coordinates
#	   the last column is the timing data (in milliseconds)
def load_motor(fn):
	motor = []
	with open(fn,'r') as fid:
		lines = fid.readlines()
	lines = [l.strip() for l in lines]
	for myline in lines:
		if myline =='START': # beginning of character
			stk = []
		elif myline =='BREAK': # break between strokes
			stk = np.array(stk)
			motor.append(stk) # add to list of strokes
			stk = [] 
		else:
			arr = np.fromstring(myline,dtype=float,sep=',')
			stk.append(arr)
	return motor

#
# Map from motor space to image space (or vice versa)
#
# Input
#   pt: [n x 2] points (rows) in motor coordinates
#
# Output
#  new_pt: [n x 2] points (rows) in image coordinates
def space_motor_to_img(pt):
	pt[:,1] = -pt[:,1]
	return pt
def space_img_to_motor(pt):
	pt[:,1] = -pt[:,1]
	return


def get_stroke(dir,img,size,lw):
    scale=size/105
    drawing = load_motor(dir)
    drawing = [d[:,0:2] for d in drawing] # strip off the timing data (third column)
    drawing = [space_motor_to_img(d) for d in drawing] # convert to image space
    stks=[]
    for sid in range(len(drawing)): # for each stroke
        mask=np.zeros([size,size])
        drawing[sid]*=scale
        for p in drawing[sid]:
            p=p[::-1]
            i=np.rint(p).astype(int)
            T=max(0,i[0]-lw)
            B=min(104,i[0]+lw)
            L=max(0,i[1]-lw)
            R=min(104,i[1]+lw)
            mask[T:B,L:R]=1
        stks.append((mask*img).astype(int).reshape([784]))
    return stks
        
def process(dir,size=28,lw=1,dset='background'):
    save_dir=os.path.join(dir,str(size)+'x'+str(size))
    if os.path.exists(os.path.join(save_dir,'omniglot.npy')): return
    img_dir = os.path.join(dir,'source','images_'+dset)
    stroke_dir = os.path.join(dir,'source','strokes_'+dset)
    imgs=[]
    strokes=[]
    index=[]
    nreps = 20 # number of renditions for each character
    alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.'] # get folder names
    for a in range(len(alphabet_names)): # for each alphabet
        t0=time.time()
        print('Processing',a+1,'/',str(len(alphabet_names)))
        alpha_name = alphabet_names[a]
        for character_id in range(1,len(os.listdir(os.path.join(img_dir,alpha_name)))+1):
            print('Progress',character_id,'/',str(len(os.listdir(os.path.join(img_dir,alpha_name)))))
            # get image and stroke directories for this character
            img_char_dir = os.path.join(img_dir,alpha_name,'character'+num2str(character_id))
            stroke_char_dir = os.path.join(stroke_dir,alpha_name,'character'+num2str(character_id))
            # get base file name for this character
            if len(os.listdir(img_char_dir))==0: continue
            fn_example = os.listdir(img_char_dir)[0]
            fn_base = fn_example[:fn_example.find('_')] 
            for r in range(1,nreps+1): # for each rendition
                fn_stk = os.path.join(stroke_char_dir,fn_base + '_' + num2str(r) + '.txt')
                fn_img = os.path.join(img_char_dir, fn_base + '_' + num2str(r) + '.png')
                img=255-np.array(Image.open(fn_img))*255
                img=np.array(Image.fromarray(img).resize([size,size]))
                img=(img>50)*np.rint((img-np.min(img))*255/(np.max(img)-np.min(img))).astype(int)
                stk=get_stroke(fn_stk,img,size,lw)
                index.append(os.path.join(alpha_name,str(character_id),str(r)))
                imgs.append(img.reshape([784]))
                strokes.append(stk)
                # return img,stk
        print(time.time()-t0)
    save_dir=os.path.join(dir,str(size)+'x'+str(size))
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    np.save(os.path.join(save_dir,'omniglot_imgs_'+dset+'.npy'),np.array(imgs))
    np.save(os.path.join(save_dir,'omniglot_strokes_'+dset+'.npy'),strokes,allow_pickle=True)
    np.save(os.path.join(save_dir,'omniglot_index_'+dset+'.npy'),index,allow_pickle=True)


class OmniGlot:
    def __init__(self,dir='.', parse=False,size=28):
        self.dir=dir
        self.size=size
        self.parse=parse
        self.load_imgs() 
        self.load_index() 
        self.load_strokes() 
        self.y_sym,self.y_lang,self.langs=self.index2label(self.index)
        
        self.imgs=self.imgs
        self.y_sym=np.array(self.y_sym,dtype=int)
        self.y_lang=np.array(self.y_lang,dtype=int)
        self.index=self.index
        
    def load_imgs(self): 
        if self.parse: self.imgs=np.load(os.path.join(self.dir,'omniglot'+str(self.size)+'_imgs_parse.npy'))
        else: self.imgs=np.load(os.path.join(self.dir,'omniglot'+str(self.size)+'_imgs.npy'))
    def load_index(self): 
        if self.parse: self.index=np.load(os.path.join(self.dir,'omniglot'+str(self.size)+'_inds_parse.npy'))
        else: self.index=np.load(os.path.join(self.dir,'omniglot'+str(self.size)+'_inds.npy'))
    def load_strokes(self): 
        if self.parse: self.stks=np.load(os.path.join(self.dir,'omniglot'+str(self.size)+'_stks_parse.npy'),allow_pickle=True)
        else: self.stks=np.load(os.path.join(self.dir,'omniglot'+str(self.size)+'_stks.npy'),allow_pickle=True)
        # self.stks=[]
        # for i in self.stks_list: self.stks+=i
        # self.stks=np.array(self.stks)
    def index2label(self,index): 
        y_sym=[]
        y_lang=[]
        sym_dict={}
        lang_dict={}
        langs={}
        for i in range(len(index)):
            lang,sym,_=index[i].split('\\')
            sym+=lang
            if lang not in lang_dict: 
                lang_dict[lang]=len(lang_dict)
                langs[len(lang_dict)]=lang
            if sym not in sym_dict: sym_dict[sym]=len(sym_dict)
            y_sym.append(sym_dict[sym])
            y_lang.append(lang_dict[lang])
        return y_sym,y_lang,langs

    def viz(self,idx):
        plt.imshow(self.imgs[idx].reshape([self.size,self.size]),cmap='gray')
        
    def collect_all(self,lang,sym=None):
        imgs=[]
        stks=[]
        index=[]
        label=[]
        if not isinstance(lang,list): lang=[lang]
        if sym and not isinstance(sym,list): sym=[sym]
        for i in range(len(self.index)):
            langi,symi,_=self.index[i].split('\\')
            if langi in lang:
                if sym and int(symi) in sym:
                    imgs.append(self.imgs[i])
                    stks.append(self.stks_list[i])
                    index.append(self.index[i])
        y_sym,y_lang,_=self.index2label(index)
        data=[]
        for i in range(len(imgs)):
            data.append(imgs[i])
            label.append((y_lang[i],y_sym[i],0))
            for j in range(len(stks[i])):
                data.append(stks[i][j])
                label.append((y_lang[i],y_sym[i],j+1))
        return np.array(data),np.array(label)


class OmniGlotDataset(Dataset):
    def __init__(self, dir, mode='sym',split='train', size=28,
                 transform=None,target_transform=None):
        omniglot=OmniGlot(dir,size=size)
        self.size=size
        labels=omniglot.y_sym if mode=='sym' else omniglot.y_lang
        self.transform = transform
        self.target_transform = target_transform
        split_size=1500 
        if split=='test':
            self.imgs=omniglot.imgs[:split_size]
            self.labels=labels[:split_size]
        elif split=='dev':
            self.imgs=omniglot.imgs[split_size:2*split_size]
            self.labels=labels[split_size:2*split_size]
        elif split=='train':
            self.imgs=omniglot.imgs[2*split_size:]
            self.labels=labels[2*split_size:]
        else: 
            self.imgs=omniglot.imgs
            self.labels=labels
        # self.num_classes=np.max(omniglot.y_lang if mode=='lang' else omniglot.y_sym)+1

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image=self.imgs[idx].reshape([1,self.size,self.size])/255.0
        image=torch.tensor(image)
        # label = self.labels[idx] if self.mode=='sym' else self.labels[idx]
        if self.transform: image = self.transform(image)
        # if self.target_transform: label = self.target_transform(label)
        return image, image #label
        
class OmniGlotParse(Dataset):
    def __init__(self, dir,split='train',transform=None,target_transform=None,size=28):
        omniglot=OmniGlot(dir,parse=True,size=size)
        self.size=size
        self.transform = transform
        self.target_transform = target_transform
        split_size=750 
        if split=='test':
            self.imgs=omniglot.imgs[:split_size]
            self.stks=omniglot.stks[:split_size]
        elif split=='dev':
            self.imgs=omniglot.imgs[split_size:2*split_size]
            self.stks=omniglot.stks[split_size:2*split_size]
        elif split=='train':
            self.imgs=omniglot.imgs[2*split_size:]
            self.stks=omniglot.stks[2*split_size:]
        else:
            self.imgs=omniglot.imgs
            self.stks=omniglot.stks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image=self.imgs[idx].reshape([1,1,self.size,self.size])/255.0
        image=torch.tensor(image)
        strokes=self.stks[idx].reshape([-1,1,self.size,self.size])/255.0
        strokes=torch.tensor(strokes)
        if self.transform: 
            together=torch.cat([image,strokes],0)
            together = self.transform(together)
            image,strokes=together[0].unsqueeze(0),together[1:]
        # if self.target_transform: label = self.target_transform(label)
        return image.squeeze(0), strokes, None #label
    
def collate_fn_ogp(list_items):
    x = []
    y = []
    for x_, y_, _ in list_items:
        x.append(x_)
        y.append(y_)
    return torch.stack(x), y, None
    
    
