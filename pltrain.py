import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import time,os,random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as T
import functools as ft
import warnings
warnings.filterwarnings('ignore')

from TAE import TAE,Sampler,TAETrainer,TAETunner
from TAE3D import VoxTAE,VoxSampler,VoxTAETrainer
from datasets.babyarc import ConceptCompositionDataset,NoiseBabyARC,collate_fn_lwp,ParseBabyARC,ConceptDataset
from datasets.omniglot import OmniGlot,OmniGlotDataset,collate_fn_ogp,OmniGlotParse
from datasets.shapenetvox import ShapeNetVoxDataset

 


""" 1. Setting """

dataset='omniglot' # lineworld omniglot shapenet lineworld_parse omniglot_parse
project='TAE_'+dataset
PATH = os.path.join('.','ckpts',dataset)
if not os.path.exists(PATH): os.makedirs(PATH)
ind=1 # Update this for each machine!
name=str(ind)
PATH = os.path.join(PATH,name)
accelerator="cuda"
devices="auto" # [1,2,3,4,5,6] #"auto"
strategy="auto"
log_every_n_steps=10
draw_freq=1
noise=False
use_small=True
mapper_small=True
use_aug=True
PE_mode='rand' # rand fix none
opt=ft.partial(optim.Adam)
sched=ft.partial(optim.lr_scheduler.StepLR,step_size=30,gamma=0.95)
# sched = ft.partial(optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=30, T_mult=1, eta_min=1e-5)

batch_size=96
eval_size=96
epochs=1000
c_m=1
mask_p=0#0.8
mask_ds=2
Niter=10 # kmeans N iters
use_zca=False # use zca in kmeans 
dropout=0.1
use_out_res=False
loss_option='mse_10.0' # focal_1.0-dice_1.0-bce_1.0-smoothl1_1.0-mse_1.0
lambda_r=10 # for tune, p2 ratio

use_attn=False
use_self_attn=True 
n_heads=8
d_head=8
context_dim=64
share_mapper=False
use_ldm=False
ldm_out=8
ldm_ds=[1,2,1,1]

lr=1e-3
sampler_dim=32
embed_dim=256
mapper_mid=256 # clustering cnn feature dim 
method='EM'
blur_sigma=0 # sigma for gaussian blur of masks
quota=32 # quota for each player
N=6
NP1=6
NP2=6
NH1=4
NH2=4
NK=2
K=3
t=100
sigma=2.5
lambda_P=2e-2

focal_alpha=0.95 # alpha for focal loss
alpha_overlap=0.005 # overlap
alpha_l2=1e-6 # l2 to control m scale
alpha_resources=0.1 # resources limit
jump_alpha=25 #25
jump_shift=0.1 # threshhold for jumping to 1
jump_smooth=False

gamma_cluster=1e-2 # clustering
memlen=3000 
threshold=(0.01,0.2)
cluster_start_epoch=0
beta=0 # sde overall

use_rl=False 
use_ae=False
rl_start_epoch=0
critic_mid=128
ppo_gamma=0.95
ppo_lamda=0.95
ppo_epsilon=0.5
ppo_entropy_coef=0.2
ppo_K_epochs=3
ppo_use_grad_clip=True
ppo_use_lr_decay=False
ppo_reward_norm=True
ppo_human_tune=False
ppo_inc_reward=False
ppo_sparse_reward=False
ppo_prefer_last=False
ppo_max_train_steps=1e6
ppo_update_every=3
ppo_pc_rand=10
reward_option='loss_1.0-prefer_0.0'
prefer_option='ct_1.0-cp_1.0-sm_1.0-z_0.0:spl'
lr_a=1e-3
lr_c=1e-3


if use_ldm: name+='-ldm'
if use_rl: 
    name+='-rl'
    if not use_ae: name+='N'
if use_attn: name+='-a'
if ppo_human_tune: name+='-H'

"""  2. Dataset  """

num_workers=16
if dataset=='lineworld':
    c_in=3
    datasize=(16,16)
    lwname='LineWorld40k'
    trainset=NoiseBabyARC(lwname+'_train',datasize,noise=noise)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    devset=NoiseBabyARC(lwname+'_dev',datasize,noise=noise)
    devloader = torch.utils.data.DataLoader(devset, batch_size=eval_size,
                                            shuffle=True, num_workers=num_workers)
    testset=NoiseBabyARC(lwname+'_test',datasize,noise=noise)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_size,
                                            shuffle=True, num_workers=num_workers)
elif dataset=='omniglot':
    c_in=1
    size=28 #16 32
    og_dir='./datasets/files'
    transforms = torch.nn.Sequential(
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 180))
    ) if use_aug else None
    trainset=OmniGlotDataset(og_dir,split='train',transform=transforms,size=size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    testset=OmniGlotDataset(og_dir,split='test',size=size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_size,
                                            shuffle=True, num_workers=num_workers)
    devset=OmniGlotDataset(og_dir,split='dev',size=size)
    devloader = torch.utils.data.DataLoader(devset, batch_size=eval_size,
                                            shuffle=True, num_workers=num_workers)
elif dataset=='shapenet':
    c_in=1
    snv16=np.load('./datasets/files/shapenetvox16.npy')
    seq=list(range(len(snv16)))
    random.shuffle(seq)
    snv16=snv16[seq]
    trainset=ShapeNetVoxDataset(snv16)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    testset=ShapeNetVoxDataset(snv16,split='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_size,
                                            shuffle=True, num_workers=num_workers)
    devset=ShapeNetVoxDataset(snv16,split='dev')
    devloader = torch.utils.data.DataLoader(devset, batch_size=eval_size,
                                            shuffle=True, num_workers=num_workers)
elif dataset=='lineworld_parse':
    c_in=3
    NP2=5 # Need to replace predicate head when finetuning!
    datasize=(16,16)
    lwname='LineWorldParse'
    trainset=ParseBabyARC(lwname+'_train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                shuffle=True, num_workers=num_workers,collate_fn=collate_fn_lwp)
    devset=ParseBabyARC(lwname+'_dev')
    devloader = torch.utils.data.DataLoader(devset, batch_size=eval_size,
                shuffle=True, num_workers=num_workers,collate_fn=collate_fn_lwp)
    testset=ParseBabyARC(lwname+'_test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_size,
                shuffle=True, num_workers=num_workers,collate_fn=collate_fn_lwp)
elif dataset=='omniglot_parse':
    c_in=1
    size=28 #16 32
    og_dir='./datasets/files'
    transforms = torch.nn.Sequential(
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 180))
    ) if use_aug else None
    trainset=OmniGlotParse(og_dir,split='train',transform=transforms,size=size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                shuffle=True, num_workers=num_workers,collate_fn=collate_fn_ogp)
    testset=OmniGlotParse(og_dir,split='test',size=size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_size,
                shuffle=True, num_workers=num_workers,collate_fn=collate_fn_ogp)
    devset=OmniGlotParse(og_dir,split='dev',size=size)
    devloader = torch.utils.data.DataLoader(devset, batch_size=eval_size,
                shuffle=True, num_workers=num_workers,collate_fn=collate_fn_ogp)




    
""" 3. Training """

Sampler_=VoxSampler if dataset=='shapenet' else Sampler
TAE_=VoxTAE if dataset=='shapenet' else TAE
TAETrainer_=VoxTAETrainer if dataset=='shapenet' else TAETrainer
if 'parse' in dataset.split('_'): 
    TAETrainer_=TAETunner
    loss_option='mse_1.0'
    use_rl=False
if use_rl and not use_ae: rl=1e-10
sampler=Sampler_(c_in=c_in,c_m=c_m,dim=sampler_dim,method=method,embed_dim=embed_dim,sigma=sigma,use_small=use_small,K=K,t=t,ldm_ds=ldm_ds,
                 lambda_P=lambda_P,NP1=NP1,NP2=NP2,NH1=NH1,NH2=NH2,NK=NK,memlen=memlen,mapper_mid=mapper_mid,dropout=dropout,use_rl=use_rl,
                 threshold=threshold,use_out_res=use_out_res,Niter=Niter,use_zca=use_zca,use_self_attn=use_self_attn,use_ldm=use_ldm,
                 n_heads=n_heads, d_head=d_head,context_dim=context_dim, share_mapper=share_mapper,critic_mid=critic_mid,ldm_out=ldm_out,mapper_small=mapper_small)
tae=TAE_(sampler,N=N,embed_dim=embed_dim,c_m=c_m,quota=quota,focal_alpha=focal_alpha,alpha_overlap=alpha_overlap,jump_alpha=jump_alpha,jump_smooth=jump_smooth,
        alpha_l2=alpha_l2,alpha_resources=alpha_resources,cluster_start_epoch=cluster_start_epoch,PE_mode=PE_mode,jump_shift=jump_shift,
        beta=beta,gamma_cluster=gamma_cluster,blur_sigma=blur_sigma,mask_ds=mask_ds,mask_p=mask_p,loss_option=loss_option,lambda_r=lambda_r,
        use_rl=use_rl,ppo_gamma=ppo_gamma,ppo_lamda=ppo_lamda,ppo_K_epochs=ppo_K_epochs,ppo_use_grad_clip=ppo_use_grad_clip,lr_a=lr_a,lr_c=lr_c,
        ppo_use_lr_decay=ppo_use_lr_decay,ppo_max_train_steps=ppo_max_train_steps,ppo_update_every=ppo_update_every,reward_option=reward_option,
        ppo_epsilon=ppo_epsilon,ppo_entropy_coef=ppo_entropy_coef,rl_start_epoch=rl_start_epoch,use_ldm=use_ldm,ppo_reward_norm=ppo_reward_norm,
        ppo_inc_reward=ppo_inc_reward,prefer_option=prefer_option,ppo_human_tune=ppo_human_tune,ppo_pc_rand=ppo_pc_rand,
        ppo_sparse_reward=ppo_sparse_reward,ppo_prefer_last=ppo_prefer_last)
model=TAETrainer_(tae,opt=opt,sched=sched,lr=lr,dataset=dataset,draw_freq=draw_freq,use_rl=use_rl)


if use_rl: strategy=pl.strategies.DDPStrategy(find_unused_parameters=True, static_graph=True)
wandb_logger = WandbLogger(name=name,project=project,save_dir=PATH)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
early_stopping = pl.callbacks.EarlyStopping('val_loss', mode="min", patience=100)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')
trainer = pl.Trainer(max_epochs=epochs,default_root_dir=PATH,log_every_n_steps=log_every_n_steps,
                     logger=wandb_logger,accelerator=accelerator, strategy=strategy, devices=devices,
                     callbacks=[lr_monitor,early_stopping,checkpoint_callback])
trainer.fit(model, trainloader, devloader)

