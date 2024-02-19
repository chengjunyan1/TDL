import torch,torchvision
import torch.nn as nn
from torch.nn import functional as F
import functools as ft
import pytorch_lightning as pl 
import torch.optim as optim
import numpy as np
import matplotlib.pylab as plt
from itertools import permutations
import random
# import faiss
from pytorch_metric_learning.distances import LpDistance
from torch.distributions import Bernoulli,Normal
import cv2
import io
from PIL import Image
from scipy.interpolate import splprep, splev

from sampler import UNetSampler
from SDE import ScoreNet,marginal_prob_std,diffusion_coeff,Encoder,Decoder,Upsample

import warnings
warnings.filterwarnings("ignore")

 
def printt(*args):
    print('-'*30)
    print(args)
    print('-'*30)

# Clustering

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = torch.cov(X) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = torch.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = torch.mm(U, torch.mm(torch.diag(1.0/torch.sqrt(S + epsilon)), U.T)) # [M x M]
    return torch.mm(ZCAMatrix, X)

def KMeans(x, K=10, Niter=10, use_zca=False): 
    """Implements Lloyd's algorithm for the Euclidean metric."""
    N, D = x.shape  # Number of samples, dimension of the ambient space
    if use_zca: x=zca_whitening_matrix(x)
    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x.view(N, 1, D) - c.view(1, K, D)) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)
        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
    return cl,c

class Kmeans(object):
    def __init__(self, k, memlen=30000, emb_dim=256, Niter=10, use_zca=False): 
        self.k = k
        self.memlen=memlen
        self.emb_dim=emb_dim
        self.KMeans=ft.partial(KMeans,K=k,Niter=Niter,use_zca=use_zca)
        self.mem=torch.empty([0,emb_dim])
        # self.mem=torch.rand(int(memlen*0.8),emb_dim)

    def clustering(self, data, D):
        self.mem=self.mem.to(data)
        db=torch.cat([data,self.mem])
        self.mem=db[:self.memlen]
        if db.shape[0]<1000 or len(data)<self.k: return None 
        # cluster the data
        # xb = preprocess_features(db)
        # I, loss = run_kmeans(xb, self.k)
        I,_=self.KMeans(db)
        clusters = [[] for i in range(self.k)]
        for i in range(len(data)):
            clusters[I[i]].append(i)
        # centroids
        C=[]
        for i in clusters: 
            if i!=[]: C.append(D[i].mean(0)) # mean code for this class
        C=torch.stack(C)
        assign=self.get_assignment(C)
        S=torch.zeros(D.shape[0]).to(assign[0])
        count=0
        for i in clusters: 
            if i!=[]: 
                S[i]=assign[count]
                count+=1
        return S
    
    def get_assignment(self,C):
        tar=torch.arange(self.k)
        tar=nn.functional.one_hot(tar,num_classes=self.k).to(C)
        DW=LpDistance(p=1)(C,tar)
        rows,cols=DW.shape
        assign=[]
        for _ in range(rows):
            coord=DW.argmin()
            row,col=coord//cols,coord%cols
            assign.append(col)
            DW[row,:]=100
            DW[:,col]=100
        return assign


class Predicates(nn.Module):
    def __init__(self,NP,NK,embed_dim=256,metric='L2',gamma=1,lambda_P=2e-2):
        super().__init__()
        self.NP=NP
        self.NK=NK
        self.P=nn.Parameter(torch.randn(NP*NK, embed_dim))
        nn.init.kaiming_normal_(self.P)
        if metric=='L2': self.metric=LpDistance(p=2)
        self.gamma=gamma
        self.lambda_P=lambda_P
        self.loss_fn=nn.CrossEntropyLoss()

    def pred(self,q):
        D=self.metric(q,self.P)
        p=torch.exp(-self.gamma*D).reshape(-1,self.NP,self.NK)
        pred=p.sum(2)
        pred=pred/pred.sum(1,keepdim=True)
        return pred,D,p

    def loss(self,q,c,pred,D):
        pnn=self.P[D.argmin(1)]
        preg=torch.square(torch.norm(pnn-q,p=2,dim=1)).mean() # L2 dist
        loss=self.loss_fn(pred,c)+self.lambda_P*preg
        return loss,pred,D

class PAE(nn.Module):
    def __init__(self, c_m, mid=128, use_samll=True, use_decoder=False,dropout=0.1):
        super().__init__()
        self.use_small=use_small
        self.use_decoder=use_decoder
        if use_small:
            self.down=nn.Sequential(
                nn.Conv2d(c_m, mid, 3, stride=2),
                nn.GroupNorm(32,mid),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.AdaptiveAvgPool2d(1),
            ) 
        else:
            self.down1=nn.Sequential(
                nn.Conv2d(c_m, mid, 3, stride=2),
                nn.GroupNorm(32,mid),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.down2=nn.Sequential(
                nn.Conv2d(mid, mid, 1, stride=1),
                nn.GroupNorm(32,mid),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.down3=nn.Sequential(
                nn.Conv2d(mid, mid, 3, stride=2),
                nn.GroupNorm(32,mid),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d(1),
            ) 
        if use_decoder:
            if use_small:
                self.decoder=nn.Sequential(
                    Upsample(mid, mid, 3, bias=False, scale_factor=1),
                    nn.GroupNorm(32, mid),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    Upsample(mid, mid, 3, scale_factor=1),
                ) if use_small else nn.Sequential(
                Upsample(mid, mid, 3, bias=False, scale_factor=2),
                nn.GroupNorm(32,mid),
                nn.SiLU(),
                nn.Dropout(dropout),
                Upsample(mid, mid, 1, bias=False, stride=1),
                nn.GroupNorm(32,mid),
                nn.SiLU(),
                nn.Dropout(dropout),
                Upsample(mid, mid, 3, scale_factor=1),
            ) 


class PMapper(nn.Module): # m to predicate space
    def __init__(self,c_m=1,mid=256,embed_dim=256,NP1=6,NP2=6,NH1=6,NH2=6,NK=2,
                 dropout=0.1,memlen=30000,threshold=(0.01,0.1),lambda_P=2e-2,
                Niter=10, use_zca=False,use_small=True):
        super().__init__()
        self.mapper=nn.Sequential(
            nn.Conv2d(c_m, mid, 3, stride=2),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d(1),
        ) if use_small else nn.Sequential(
            nn.Conv2d(c_m, mid, 3, stride=2),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(mid, mid, 1, stride=1),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(mid, mid, 3, stride=2),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d(1),
        ) 
        self.NP1,self.NP2=NP1,NP2
        self.NH1,self.NH2=NH1,NH2
        self.Q1=nn.Linear(mid,embed_dim)
        self.Q2=nn.Linear(mid,embed_dim)
        self.P1=Predicates(NP1,NK,embed_dim,lambda_P=lambda_P)
        self.P2=Predicates(NP2,NK,embed_dim,lambda_P=lambda_P)
        self.K1=Kmeans(NP1,memlen,embed_dim,Niter,use_zca)
        self.K2=Kmeans(NP2,memlen,embed_dim,Niter,use_zca)
        if NH1>0 or NH2>0: # HOL predicates, on groups 
            self.QH1=nn.Linear(mid,embed_dim)
            self.QH2=nn.Linear(mid,embed_dim)
            self.PH1=Predicates(NH1,NK,embed_dim,lambda_P=lambda_P)
            self.PH2=Predicates(NH2,NK,embed_dim,lambda_P=lambda_P)
            self.KH1=Kmeans(NH1,memlen,embed_dim,Niter,use_zca)
            self.KH2=Kmeans(NH2,memlen,embed_dim,Niter,use_zca)
        # self.temperature=embed_dim ** 0.5
        self.threshold=threshold

    def embed(self,m): 
        b,N,c,h,w=m.shape
        q1=self.Q1(self.mapper(m.reshape([-1,c,h,w]))[:,:,0,0])
        mp=m.unsqueeze(1).repeat(1,N,1,1,1,1)+m.unsqueeze(2).repeat(1,1,N,1,1,1)
        mp=(mp-nn.ReLU()(mp-1)).reshape(-1,c,h,w) # remove repeat
        q2=self.Q2(self.mapper(mp)[:,:,0,0])
        return q1,q2
    
    def embedH(self,m):
        b,N,c,h,w=m.shape
        q1,q2=None,None
        if self.NH1>0: q1=self.QH1(self.mapper(m.reshape([-1,c,h,w]))[:,:,0,0])
        if self.NH2>0:
            mp=m.unsqueeze(1).repeat(1,N,1,1,1,1)+m.unsqueeze(2).repeat(1,1,N,1,1,1)
            mp=(mp-nn.ReLU()(mp-1)).reshape(-1,c,h,w) # remove repeat
            q2=self.QH2(self.mapper(mp)[:,:,0,0])
        return q1,q2
    
    def rel_pred(self,m):
        b,N,c,h,w=m.shape
        mp=m.unsqueeze(1).repeat(1,N,1,1,1,1)+m.unsqueeze(2).repeat(1,1,N,1,1,1)
        mp=(mp-nn.ReLU()(mp-1)).reshape(-1,c,h,w) # remove repeat
        q2=self.Q2(self.mapper(mp)[:,:,0,0])
        pred2,dist2,p2=self.P2.pred(q2)
        return pred2

    def forward(self,m,mx=None,loss=False):
        b,N,c,h,w=m.shape
        q1,q2=self.embed(m)
        pred1,dist1,p1=self.P1.pred(q1)
        pred2,dist2,p2=self.P2.pred(q2)
        p1=p1.reshape(b,N,-1)
        pr=p1.matmul(self.P1.P) # p representation
        p2m,_=pred2.max(-1)
        p2m=p2m.reshape(b,N,N)
        # d2d=p2m*(1-torch.eye(N)).unsqueeze(0).to(m) # max as the prob, remvove self
        # gr=torch.einsum('ijkpq,iwj->iwkpq',m,d2d) # group representation
        gr=torch.einsum('ijkpq,iwj->iwkpq',m,p2m) #gr+m
        h1,h2,phr,ghr=None,None,None,None
        if self.NH1>0 or self.NH2>0: 
            gx=nn.Sigmoid()(gr)
            h1,h2=self.embedH(gr) # groups
            predH1,distH1,pH1=self.PH1.pred(h1)
            predH2,distH2,pH2=self.PH2.pred(h2)
            pH1=pH1.reshape(b,N,-1)
            phr=pH1.matmul(self.PH1.P) # p representation
            pH2m,_=predH2.max(-1)
            pH2m=pH2m.reshape(b,N,N)
            # d2dH=pH2m*(1-torch.eye(N)).unsqueeze(0).to(gm) # max as the prob 
            # ghr=torch.einsum('ijkpq,iwj->iwkpq',gm,d2dH) # group representation
            ghr=torch.einsum('ijkpq,iwj->iwkpq',gr,pH2m) #gr+m
        if loss:
            with torch.no_grad():
                ma=mx.reshape(b*N,-1,h,w).sum([1,2,3])
                low,high=self.threshold[0]*h*w*c,self.threshold[1]*h*w*c
                mb =  torch.logical_and(ma>low,ma<high)
                mb1=mb.reshape(b,N).unsqueeze(1).repeat(1,N,1)
                mb2=mb.reshape(b,N).unsqueeze(2).repeat(1,1,N)
                mbb=torch.logical_and(mb1,mb2).reshape(-1)
                ind1 = mb.nonzero().squeeze(1).cpu().detach().tolist()
                ind2 = mbb.nonzero().squeeze(1).cpu().detach().tolist()
                C1=self.K1.clustering(q1[ind1],pred1[ind1]) # get clusters
                if len(ind2)>int(pred2.shape[0]/np.sqrt(2*N)):
                    ind2=random.sample(ind2,int(pred2.shape[0]/np.sqrt(2*N)))
                C2=self.K2.clustering(q2[ind2],pred2[ind2])
            loss_cluster=0
            if C1 is not None: loss_cluster+=self.P1.loss(q1[ind1],C1,pred1[ind1],dist1[ind1])[0]
            if C2 is not None: loss_cluster+=self.P2.loss(q2[ind2],C2,pred2[ind2],dist2[ind2])[0]
            if self.NH1>0 or self.NH2>0:
                with torch.no_grad():
                    ga=gx.reshape(b*N,-1,h,w).sum([1,2,3])
                    low,high=self.threshold[0]*h*w*c,self.threshold[1]*h*w*c*N # higher high 
                    gb =  torch.logical_and(ga>low,ga<high)
                    gb1=mb.reshape(b,N).unsqueeze(1).repeat(1,N,1)
                    gb2=mb.reshape(b,N).unsqueeze(2).repeat(1,1,N)
                    gbb=torch.logical_and(gb1,gb2).reshape(-1)
                    ind1 = gb.nonzero().squeeze(1).cpu().detach().tolist()
                    ind2 = gbb.nonzero().squeeze(1).cpu().detach().tolist()
                    CH1,CH2=None,None
                    if h1 is not None:
                        CH1=self.KH1.clustering(h1[ind1],predH1[ind1]) # get clusters
                    if h2 is not None:
                        if len(ind2)>int(predH2.shape[0]/np.sqrt(2*N)):
                            ind2=random.sample(ind2,int(predH2.shape[0]/np.sqrt(2*N)))
                        CH2=self.KH2.clustering(h2[ind2],predH2[ind2])
                if CH1 is not None: loss_cluster+=self.PH1.loss(h1[ind1],CH1,predH1[ind1],distH1[ind1])[0]
                if CH2 is not None: loss_cluster+=self.PH2.loss(h2[ind2],CH2,predH2[ind2],distH2[ind2])[0]
            return loss_cluster
        else: return pr,gr,phr,ghr


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.gts = [] # env
        self.ms = [] # env
        self.m_in=[]
        self.logprobs = []
        self.advantages = []
        self.is_terminals = []
        self.v_targets=[]
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.gts[:]
        del self.ms[:]
        del self.m_in[:]
        del self.logprobs[:]
        del self.advantages[:]
        del self.is_terminals[:]
        del self.v_targets[:]

class Sampler(nn.Module): # [x;z]->dz
    def __init__(
#region Net arch params
            self,
            c_in,
            c_m,
            dim=32,
            embed_dim=128,
            sigma=5, # the larger the more spread out
            K=6, # diffusion steps
            eps=1e-5, # A tolerance value for numerical stability.
            method='EM', # ODE VAE PC
            snr=0.16,
            t=500, # temprature of step size
            mapper_mid=256, # mapper mid dim
            NP1=6,
            NP2=4, 
            NH1=4,
            NH2=4,
            NK=2,
            lambda_P=2e-2,
            dropout=0.1,
            memlen=30000,
            threshold=(0.01,0.1),
            use_out_res=True,
            use_small=True,
            Niter=10,
            use_zca=False,
            use_ldm=False,
            ldm_out=8,
            ldm_ds=[1,2,1,1],
            mapper_small=False,
            # Attn
            use_attn=False, 
            use_self_attn=False, 
            n_heads=8, 
            d_head=16,
            context_dim=256, 
            share_mapper=False,
#endregion
            # RL
            use_rl=False,
            critic_mid=128,
            **kwargs,
        ):
        super().__init__()
        self.c_in=c_in
        self.c_m=c_m
        self.K=K
        self.t=t
        self.marginal_prob_std_fn = ft.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = ft.partial(diffusion_coeff, sigma=sigma)
        self.use_t=method not in ['VAE']
        mapper_dim=ldm_out if use_ldm else c_m
        self.mapper=PMapper(mapper_dim,mapper_mid,embed_dim,NP1,NP2,NH1,NH2,NK,dropout,memlen,threshold,lambda_P,Niter,use_zca,mapper_small) # share mapper for each layer
        model=ScoreNet if use_small else UNetSampler
        mapper=self.mapper.mapper if share_mapper else None
        self.use_ldm=use_ldm
        mul=4 if NH2>0 else 3
        in_channels=c_in+c_m*mul
        out_channels=c_m
        if use_ldm:
            self.ldm_out=ldm_out
            self.encoder=Encoder(c_in=c_in,dim=dim,dropout=dropout,use_attn=use_attn,n_heads=n_heads,d_head=d_head,ds=ldm_ds)
            self.decoder=Decoder(in_channels=ldm_out,c_out=out_channels,dim=dim,use_attn=use_attn,n_heads=n_heads,dropout=dropout,d_head=d_head,sf=ldm_ds[::-1])
            in_channels,out_channels=dim+ldm_out*mul,ldm_out
        self.net=model(self.marginal_prob_std_fn,in_channels,out_channels, dropout=dropout,use_t=self.use_t,
            dim=dim,embed_dim=embed_dim,use_out_res=use_out_res,use_attn=use_attn,use_self_attn=use_self_attn,
            n_heads=n_heads,d_head=d_head,context_dim=context_dim,mapper=mapper,use_ac=use_rl,mid=critic_mid) # Actor
        self.use_rl=use_rl
        if use_rl: self.buffer=RolloutBuffer()
        self.eps=eps
        self.snr=snr
        self.method=method

    def get_input(self,x,m,N,context):
        b,c,h,w=x.shape
        c_m=self.ldm_out if self.use_ldm else self.c_m
        mcs=m.reshape(-1,1,N,c_m,h,w).repeat(1,N,1,1,1,1) # for each, we get its competetors representation, or only within the class
        mask=1-torch.eye(N).unsqueeze(0).unsqueeze(3).repeat(1,1,1,c_m*h*w).to(x.device)
        mcm=(mask*mcs.reshape(-1,N,N,c_m*h*w)).reshape(-1,N,c_m,h,w) # competetors
        mc=mcm.reshape(b,N,-1).sum(1).reshape(-1,c_m,h,w) # compatitors map
        pr,gr,phr,ghr=self.mapper(m.reshape(-1,N,c_m,h,w)) # d1 d2 clustering
        mg=gr.reshape(-1,c_m,h,w)
        context=context+pr.reshape(b,-1)
        if phr is not None: context=context+phr.reshape(b,-1)
        if ghr is not None: 
            mgh=ghr.reshape(-1,c_m,h,w)
            inp=torch.cat([x,m,mc,mg,mgh],dim=1)
        else: inp=torch.cat([x,m,mc,mg],dim=1)
        return inp, context, mcm 

    def scoring(self,x,m,N,context,t=None,hs=None,m_=None):
        inp, context, mcm = self.get_input(x,m,N,context) # use current state
        hs=[h.detach() for h in hs] if hs is not None else None
        state=[x.detach(), m.detach(), inp.detach(), context.detach(), mcm.detach(), t, hs]
        return self.net(inp, context, t, mcm), state # current state

    @torch.no_grad()
    def get_state(self,x,m,N,context,t=None, hs=None):
        inp, context, mcm = self.get_input(x,m,N,context)
        return [x, m, inp, context, mcm,t, hs]

    def sample_action(self, m):
        mx=nn.Sigmoid()(m)
        dist = Bernoulli(mx) # For c_m>1, categorical
        action = dist.sample()
        entropy=dist.entropy()
        action_logprob = dist.log_prob(action)
        return action,action_logprob,entropy
    
    def actor(self,state, N):
        x, m, inp, context, mcm, t, hs = state
        m, state=self.scoring(x,m,N,context,t)
        if self.use_ldm: m=self.decoder(m,hs[0],hs[1],hs[2])
        return m

    def critic(self,state):
        x, m, inp, context, mcm, t, hs = state
        return self.net.critic(inp,context,t,mcm)

    def rel_pred(self,m):return self.mapper.rel_pred(m)

    def forward(self,x,context,N,gt=None): 
        hs=None
        if self.use_ldm: 
            x,h1,h2,h3=self.encoder(x)
            hs=[h1,h2,h3] 
        if self.method in ['EM','ODE','PC','DDPM']: ms,ss,ts=self.sde(x,context,N,gt,hs)
        elif self.method in ['VAE']: ms,ss,ts=self.vae(x,context,N,gt,hs)
        else: raise
        m_l=ms[-1]
        m_d=self.decoder(m_l,h1,h2,h3) if self.use_ldm else m_l
        return m_d,ss,ts,m_l

    def vae(self,x,context,N,gt=None,hs=None): 
        b,c,h,w=x.shape
        time_steps = torch.linspace(1., self.eps, 2).to(x.device)
        c_m=self.ldm_out if self.use_ldm else self.c_m
        m_init=torch.rand(b,c_m,h,w).to(x.device)*self.marginal_prob_std_fn(time_steps[0],x)#[:, None, None, None]
        m, state=self.scoring(x,m_init,N,context,hs=hs)
        if self.use_rl and self.training: 
            assert gt is not None
            with torch.no_grad():
                next_state=self.get_state(x,m,N,context,hs=hs)
                self.buffer.states.append(state)
                self.buffer.next_states.append(next_state)
                self.buffer.gts.append(gt)
                m_a=m
                if self.use_ldm: m_a=self.decoder(m_a,hs[0],hs[1],hs[2])
                m_in=torch.rand_like(m_a).to(x.device) * self.marginal_prob_std_fn(time_steps[0],x) if self.use_ldm else m_init
                self.buffer.m_in.append(m_in)
                self.buffer.ms.append(m_a)
                action,action_logprob,entropy=self.sample_action(m_a)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.is_terminals.append(1)
        return [m], None, None

    def sde(self,x,context,N,gt=None,hs=None):
        b,c,h,w=x.shape
        time_steps = torch.linspace(1., self.eps, self.K+2).to(x.device)
        c_m=self.ldm_out if self.use_ldm else self.c_m
        m=torch.rand(b,c_m,h,w).to(x.device) * self.marginal_prob_std_fn(time_steps[0],x)#[:, None, None, None]
        m_init=m
        m_in=None 
        step_size = (time_steps[0] - time_steps[1])*self.K/self.t
        ms,ss,ts=[],[],[] 
        m_=None
        for i, time_step in enumerate(time_steps[1:-1]):      
            batch_time_step = torch.ones(b).to(x.device) * time_step
            ts.append(batch_time_step)
            grad, state=self.scoring(x, m, N, context, batch_time_step, hs, m_=m_)
            m_=state[1]
            ss.append(grad)
            if self.method=='DDPM': 
                mean_m=grad
                m=mean_m
            else:
                g = self.diffusion_coeff_fn(batch_time_step, x)
                if self.method=='PC':
                    # Corrector step (Langevin MCMC)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(m.shape[1:]))
                    langevin_step_size = 2 * (self.snr * noise_norm / grad_norm)**2
                    m = m + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(m)    
                mean_m = m + (g**2)[:, None, None, None] * grad * step_size
                if self.method in ['EM','PC']:
                    m = mean_m + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(m)
                elif self.method=='ODE': m = mean_m
            ms.append(mean_m)
            if self.use_rl and self.training: 
                assert gt is not None
                with torch.no_grad():
                    next_time_step = torch.ones(b).to(x.device) * time_steps[1:][i+1]
                    next_state=self.get_state(x,mean_m,N,context,next_time_step,hs)
                    self.buffer.states.append(state)
                    self.buffer.next_states.append(next_state)
                    self.buffer.gts.append(gt)
                    m_a=mean_m
                    if self.use_ldm: m_a=self.decoder(m_a,hs[0],hs[1],hs[2])
                    self.buffer.ms.append(m_a)
                    if m_in is None: m_in=torch.rand_like(m_a).to(x.device) * self.marginal_prob_std_fn(time_steps[0],x) if self.use_ldm else m_init
                    self.buffer.m_in.append(m_in)
                    m_in=m_a
                    action,action_logprob,entropy=self.sample_action(m_a)
                    self.buffer.actions.append(action)
                    self.buffer.logprobs.append(action_logprob)
                    self.buffer.is_terminals.append(1 if i==len(time_steps[1:-1])-1 else 0)
        return ms,torch.cat(ss),torch.cat(ts)

    def loss_fn(self, ss, ts):
        """The loss function for training score-based generative models.

        Args:
            model: A PyTorch model instance that represents a 
            time-dependent score-based model.
            x: A mini-batch of training data.    
            marginal_prob_std: A function that gives the standard deviation of 
            the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        z = torch.rand_like(ss).to(ss)
        std = self.marginal_prob_std_fn(ts,z)
        loss = torch.mean(torch.sum((ss * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss



class FocalLossCE(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x, y):
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss_ce(alpha = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32):
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLossCE(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl


class SoftDiceLoss(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(N, )
        '''
        N=logits.shape[0]
        probs = torch.sigmoid(logits)
        numer = (probs * labels).reshape(N,-1).sum(1)
        denor = (probs.pow(self.p) + labels.pow(self.p)).reshape(N,-1).sum(1)
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

def blur(x,sigma=2):
    if sigma==0: return x
    GB=torchvision.transforms.GaussianBlur(3,sigma)
    x=GB(x)*9
    return x-nn.ReLU()(x-1)

def masking(x,ds=4,p=0.8):
    b,c,h,w=x.shape
    m=(torch.FloatTensor(b,1,h//ds, w//ds).uniform_() > p).to(x)
    return m.unsqueeze(3).unsqueeze(5).repeat(1,1,1,ds,1,ds).reshape(b,1,h,w)

def softjump(x,alpha=25,shift=0.2,smooth=False): # scale up penalty above threshold
    if smooth:
        x_=alpha*nn.ReLU()(x-shift)
        xi=-100*nn.ReLU()(0.1-x_)
        x_=x_+xi
        xj=torch.exp(x_)/(1+torch.exp(x_))
        xnj=nn.ReLU()(shift-x)
        xnj=nn.ReLU()(shift-xnj)
        return xj*(1-shift)+xnj
    else:
        x=alpha*(x-shift)
        return torch.exp(x)/(1+torch.exp(x))

class TAE(nn.Module):
    def __init__(
            self,
            sampler,
            N=8,
            embed_dim=256,
            c_m=1, # channel of mask
            alpha_overlap=0.1, # overlap
            alpha_l2=0.2, # l2 to control m scale
            alpha_resources=0.1, # resources for sparse m
            focal_alpha=0.75, # focal loss alpha
            jump_alpha=None, # alpha for soft jump
            jump_shift=0.2, # shift of jump
            jump_smooth=False,
            blur_sigma=2, # sigma for blurring overlap
            quota=8, # quota for each player
            beta=1.0, # sde overall
            gamma_cluster=1.0, # clustering overall
            lambda_r=0.1, # lambda for rel pred
            mask_p=0.8, # rand mask ratio
            mask_ds=4, # mask downsample ratio, i.e. chunk size 28/ds
            cluster_start_epoch=20, 
            PE_mode='rand', # rand fix none
            loss_option='focal_1.0-dice_1.0', # e.g. focal_1.0-dice_1.0-bce_0.1-smoothl1_1.0:b  mse
            use_ldm=False,
            # rl settings
            use_rl=False,
            rl_start_epoch=50,
            ppo_gamma=0.95,
            ppo_lamda=0.95,
            ppo_epsilon=1.0,
            ppo_entropy_coef=0.2,
            ppo_K_epochs=3,
            ppo_use_grad_clip=True,
            ppo_use_lr_decay=False,
            ppo_reward_norm=True,
            ppo_inc_reward=True,
            ppo_human_tune=False,
            ppo_sparse_reward=False,
            ppo_prefer_last=False,
            ppo_max_train_steps=1e6,
            ppo_update_every=5,
            reward_option='loss_1.0-prefer_1.0', # e.g. loss_1.0-preference_1.0, predict minus loss and human preference 
            prefer_option='ct_1.0-cp_1.0-sm_1.0-z_0.0:spl', # continous, compactness, smoothness, base score
            lr_a=1e-3,
            lr_c=1e-3,
            **kwargs
        ):
        super().__init__()
        self.sampler=sampler
        self.alpha_overlap=alpha_overlap
        self.alpha_l2=alpha_l2
        self.lambda_r=lambda_r
        self.alpha_resources=alpha_resources
        self.focal_alpha=focal_alpha
        self.softjump=None if jump_alpha is None else ft.partial(softjump,alpha=jump_alpha,shift=jump_shift,smooth=jump_smooth)
        self.blur=ft.partial(blur,sigma=blur_sigma) if blur_sigma>0 else None
        self.quota=quota
        self.beta=beta
        self.gamma_cluster=gamma_cluster
        self.c_m=c_m
        self.c_in=sampler.c_in
        self.N=N
        self.embed_dim=embed_dim
        self.act=nn.Softmax(1) if c_m>1 else nn.Sigmoid()
        self.mask_p=mask_p
        self.mask_ds=mask_ds
        self.cluster_start_epoch=cluster_start_epoch
        self.PE_mode=PE_mode
        self.loss_option=loss_option
        loss_option=self.loss_option.split(':')[0].split('-')
        self.loss_dict={}
        for i in loss_option: self.loss_dict[i.split('_')[0]]=float(i.split('_')[1])
        if PE_mode!='none': self.posemb=nn.Embedding(256,embed_dim) # ids, should be far enough
        self.use_ldm=use_ldm

        self.use_rl=use_rl
        self.rl_start_epoch=rl_start_epoch
        if use_rl:
            self.ppo_gamma=ppo_gamma
            self.ppo_lamda=ppo_lamda
            self.ppo_epsilon=ppo_epsilon
            self.ppo_entropy_coef=ppo_entropy_coef
            self.ppo_K_epochs=ppo_K_epochs
            self.ppo_use_grad_clip=ppo_use_grad_clip
            self.ppo_reward_norm=ppo_reward_norm
            self.ppo_inc_reward=ppo_inc_reward
            self.ppo_use_lr_decay=ppo_use_lr_decay
            self.ppo_max_train_steps=ppo_max_train_steps
            self.ppo_update_every=ppo_update_every
            self.ppo_sparse_reward=ppo_sparse_reward
            self.ppo_human_tune=ppo_human_tune
            self.ppo_prefer_last=ppo_prefer_last
            actor_params=get_params_exclude(self.sampler,self.sampler.net.critic_head)
            critic_params=get_params_exclude(self.sampler,self.sampler.net.actor_head)
            self.optimizer_actor = torch.optim.Adam(actor_params, lr=lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(critic_params, lr=lr_c, eps=1e-5)
            self.reward_option=reward_option
            reward_option=self.reward_option.split(':')[0].split('-')
            self.reward_dict={}
            for i in reward_option: self.reward_dict[i.split('_')[0]]=float(i.split('_')[1])
            self.prefer_option=prefer_option
            self.prefer_dict=[float(i.split('_')[1]) for i in prefer_option.split(':')[0].split('-')]
            

    def forward(self,x,gt=None,ret_all=False): # x is image BCHW 
        x=x.to(device=x.device, dtype=torch.float)
        x=x[:,:self.c_in]
        b,c,h,w=x.shape
        if self.PE_mode=='none': PE=0
        else:
            if self.PE_mode=='fix':
                pos=torch.range(0,self.N-1).reshape(1,-1).repeat(b,1).reshape(-1,1)
            elif self.PE_mode=='rand':
                pos=torch.randint(0,256,[b*self.N,1]) # randomly assign player id
            PE=self.posemb(pos.long().to(x.device)).squeeze(1)
        X=x.unsqueeze(1).repeat(1,self.N,1,1,1).reshape(-1,c,h,w) # each batch copy x
        if self.mask_p>0: X=X*masking(X,p=self.mask_p,ds=self.mask_ds)
        if gt is not None:
            if len(gt.shape)==4: GT=gt.unsqueeze(1).repeat(1,self.N,1,1,1).reshape(-1,c,h,w) 
            else: GT=gt.unsqueeze(1).repeat(1,self.N,1,1).reshape(-1,h,w)
        else: GT=None
        m,ss,ts,m_l=self.sampler(X,PE,self.N,GT)
        if ret_all: return m,ss,ts,m_l
        return m
    
    def relpred(self,m): return self.sampler.rel_pred(m)

    def loss_tae(self,m,gt,log=None):
        B,c_m,h,w=m.shape
        b=B//self.N
        mx=self.act(m)
        if self.c_m>1:  # TODO: need a lot update
            rx=mx.reshape(b,-1,self.c_m,h,w).sum([1]) # each one has independent prediction, together make good, allow empty
            loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.1]+[0.7]*(self.c_m-1)).to(x.device))
            r=(self.act(rx)).argmax(1).to(x.device)
            bm=self.blur(m) if self.blur is not None else m
            bmstack=nn.Softmax(2)(bm.reshape(b,-1,self.c_m,h,w))[:,:,1:].sum([1,2]) # sum of non-zero parts
            mxs=nn.ReLU()(mx[:,1:].reshape(b,-1,h,w).sum([2,3])-self.quota).sum(1).mean()
            loss_reconstruct=loss_fn(rx,gt)
        else: 
            mstack=mx.reshape(b,-1,1,h,w).sum([1]).squeeze(1)
            bmx=self.softjump(mx) if self.softjump is not None else mx
            bmx=self.blur(bmx) if self.blur is not None else bmx
            bmstack=bmx.reshape(b,-1,1,h,w).sum([1]).squeeze(1)
            rx=mstack-nn.ReLU()(mstack-1); r=rx # all agreed parts to 1
            mxs=nn.ReLU()(mx.reshape(b,-1,h,w).sum([2,3])-self.quota).sum(1).mean() # exceed resources used for each
            loss_reconstruct=0
            for i in self.loss_dict:
                if i=='focal': loss_i=focal_loss(rx,gt,alpha=self.focal_alpha).mean()*self.loss_dict[i]
                elif i=='smoothl1': loss_i=nn.SmoothL1Loss()(rx,gt).mean()*self.loss_dict[i]
                elif i=='dice': loss_i=SoftDiceLoss()(rx,gt).mean()*self.loss_dict[i]
                elif i=='bce': loss_i=nn.BCELoss()(rx,gt).mean()*self.loss_dict[i]
                elif i=='mse': loss_i=nn.MSELoss()(rx,gt).mean()*self.loss_dict[i]
                log.update({i:loss_i})
                loss_reconstruct+=loss_i
        loss_overlap=self.alpha_overlap*nn.ReLU()(bmstack-1).sum([1,2]).mean() # minimize blured overlap predicate-wise, avoid repeating
        loss_tae_l2=self.alpha_l2*(m**2).mean() if self.alpha_l2!=0 else 0
        loss_tae_resources=self.alpha_resources*mxs.mean() if self.alpha_resources!=0 else 0
        loss_tae=loss_reconstruct+loss_overlap+loss_tae_l2+loss_tae_resources
        if log is not None: log.update({'loss_reconstruct':loss_reconstruct,'loss_overlap':loss_overlap,
                    'loss_tae_resources':loss_tae_resources,'loss_tae_l2':loss_tae_l2})
        return loss_tae,log,r,rx

    def loss(self,x,y,epoch=None): # step*batchsize
        x,y=x[:,:self.c_in],y[:,:self.c_in]
        if self.c_in>1: #TODO need update
            gt=(y.argmax(1)>0).float().to(x.device) if self.c_m==1 else y.argmax(1)
        else: 
            if 'b' in self.loss_option.split(':'): gt=(y>0).squeeze(1).float().to(x.device) 
            else: gt=y.squeeze(1).float().to(x.device) 
        m,ss,ts,m_l=self(x,gt=gt,ret_all=True)
        mx=self.act(m)
        log={}
        loss=0

        #--- reconstruct ---
        loss_tae,log,r,rx=self.loss_tae(m,gt,log)
        loss+=loss_tae

        #--- clustering ---
        if self.gamma_cluster>0 and epoch is not None and epoch>=self.cluster_start_epoch:
            B,c_m,h_m,w_m=m_l.shape
            loss_cluster=self.sampler.mapper(m_l.reshape(-1,self.N,c_m,h_m,w_m),mx,loss=True)*self.gamma_cluster
            loss+=loss_cluster
            log.update({'loss_cluster':loss_cluster})

        # loss SDE        
        if self.beta>0 and self.sampler.use_t:
            loss_sde=self.sampler.loss_fn(ss,ts)*self.beta
            loss+=loss_sde
            log.update({'loss_sde':loss_sde})

        union=r+gt
        union=torch.where(union>1,1,union)
        iou=(r*gt).sum()/union.sum()
        mae=nn.L1Loss()(rx,gt).mean()
        log.update({'loss':loss,'iou':iou,'mae':mae})
        return loss,log,[mx,rx,gt]

    def parse(self,x,gt,o_only=False):
        o,_,_,o_l=self(x,gt=gt,ret_all=True)
        r=None
        if not o_only:        
            if self.use_ldm:
                B,c,h,w=o_l.shape
                r=self.relpred(o_l.reshape(-1,self.N,c,h,w))
            else: 
                B,c,h,w=o.shape
                r=self.relpred(o.reshape(-1,self.N,c,h,w))
        return o,r,o_l
    
    def parse_loss(self,x,o,r=None,epoch=None):
        x=x[:,:self.c_in] #if len(x.shape)==4 else x.unsqueeze(1)
        gt=x.squeeze(1).float().to(x.device) 
        b,c,h,w=x.shape
        o_,r_,o_l_=self.parse(x,gt,o_only=r is None)
        o_=o_.reshape(b,self.N,self.c_m,h,w)
        if r_ is not None:
            r_=r_.reshape(b,self.N,self.N,-1)
        loss1=0
        loss2=0
        loss_option=self.loss_option.split(':')[0].split('-')
        loss_dict={}
        for i in loss_option: loss_dict[i.split('_')[0]]=float(i.split('_')[1])
        N_class=self.sampler.mapper.NP2
        alpha=torch.fill(torch.zeros(N_class),self.focal_alpha)
        alpha[0]=1-self.focal_alpha
        loss_fn2=focal_loss_ce(alpha=alpha,gamma=2,reduction= 'none',device=x.device,dtype=x.dtype) # nn.CrossEntropyLoss
        for i in range(b):
            oi_=o_[i]
            oi=o[i].float().to(x.device) # target
            if r is not None:
                ri_=r_[i]
                ri=r[i]
            with torch.no_grad():
                perms=torch.tensor(list(permutations(range(self.N),len(oi)))).long().to(x.device)
                allperms=oi_[perms]
                mae=torch.abs(allperms-oi.unsqueeze(0)).sum([-1,-2,-3,-4])
                seq=torch.tensor(perms[mae.argmin()])
                # if r is not None:
                #     ind,y=[],[]
                #     for rx in ri:
                #         ind.append(rx[0]*len(oi)+rx[1])
                #         y.append(rx[2])
            unseq=[]
            for oind in range(len(oi_)):
                if oind not in seq.tolist(): unseq.append(oind)
            nseq=seq.tolist()+unseq
            oz=torch.zeros([len(unseq),oi.shape[1],oi.shape[2],oi.shape[3]]).to(oi)
            oi_s=oi_[nseq]
            oi=torch.cat([oi,oz])
            for i in loss_dict:
                if i=='focal': loss_i=focal_loss(oi_s,oi,alpha=self.focal_alpha).mean()*loss_dict[i]
                elif i=='smoothl1': loss_i=nn.SmoothL1Loss()(oi_s,oi).mean()*loss_dict[i]
                elif i=='dice': loss_i=SoftDiceLoss()(oi_s,oi).mean()*loss_dict[i]
                elif i=='bce': loss_i=nn.BCELoss()(oi_s,oi).mean()*loss_dict[i]
                elif i=='mse': loss_i=nn.MSELoss()(oi_s,oi).mean()*loss_dict[i]
                loss1+=loss_i
            # if r is not None:
            #     rel=ri_[seq][:,seq]
            #     print(ri)
            #     rel=rel+rel.permute(1,0,2) 
            #     rel=rel.reshape(-1,self.sampler.mapper.NP2)[ind]
            #     loss2+=loss_fn2(rel,torch.tensor(y).type(torch.LongTensor).to(x.device))
            if r is not None:
                y=torch.zeros([len(nseq),len(nseq)])
                for rx in ri:
                    y[seq[rx[0]],seq[rx[1]]]=rx[2]
                y=y.reshape(-1).type(torch.LongTensor).to(x.device)
                rel=ri_.reshape(-1,N_class)
                loss2+=loss_fn2(rel,y).mean()
        loss1/=b
        loss2/=b
        loss=loss1+self.lambda_r*loss2
        log={'loss1':loss1,'loss2':loss2*self.lambda_r,'loss':loss}
        mx=self.act(o_)
        mstack=mx.reshape(b,-1,1,h,w).sum([1]).squeeze(1)
        rx=mstack-nn.ReLU()(mstack-1)
        bmx=self.softjump(mx) if self.softjump is not None else mx
        bmstack=bmx.reshape(b,-1,1,h,w).sum([1]).squeeze(1)
        loss_overlap=self.alpha_overlap*nn.ReLU()(bmstack-1).sum([1,2]).mean() # minimize blured overlap predicate-wise, avoid repeating
        loss_tae_l2=self.alpha_l2*(o_**2).mean() if self.alpha_l2!=0 else 0
        mxs=nn.ReLU()(mx.reshape(b,-1,h,w).sum([2,3])-self.quota).sum(1).mean() # exceed resources used for each
        loss_tae_resources=self.alpha_resources*mxs.mean() if self.alpha_resources!=0 else 0
        loss=loss+loss_overlap+loss_tae_l2+loss_tae_resources
        #--- clustering ---
        if self.gamma_cluster>0 and epoch is not None and epoch>=self.cluster_start_epoch:
            B,c_m,h_m,w_m=o_l_.shape
            loss_cluster=self.sampler.mapper(o_l_.reshape(-1,self.N,c_m,h_m,w_m),mx,loss=True)*self.gamma_cluster
            loss+=loss_cluster
            log.update({'loss_cluster':loss_cluster})
        log.update({'loss_overlap':loss_overlap,'loss_tae_resources':loss_tae_resources,'loss_tae_l2':loss_tae_l2})
        return loss,log,[mx,rx,gt]

    def ccs_score(self,m, th=0.5):
        m=np.uint8((m>th))
        ret, thresh = cv2.threshold(m, 0.5, 1, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        NCH=len(contours) # continous 
        if NCH==0: return 0
        cnt= max(contours, key=cv2.contourArea) # largest contour
        out_mask = np.zeros_like(m)
        cv2.drawContours(out_mask, [cnt], -1, 1, cv2.FILLED, 1)    
        if np.sum(out_mask)==0: IOU=0
        else: IOU=np.sum(out_mask*m)/np.sum(out_mask) # compactness

        perimeter = cv2.arcLength(cnt,True)
        if perimeter==0: smoothness=0 # point
        else:
            smooth=self.prefer_option.split(':')[1]
            if smooth=='hull':
                approx = cv2.convexHull(cnt)
            elif smooth=='rdp':
                epsilon = 0.1*perimeter
                approx = cv2.approxPolyDP(cnt,epsilon,True)
            elif smooth=='spl':
                try:
                    approx=spl_contour(cnt)
                except: 
                    epsilon = 0.1*perimeter
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
            smooth_perimeter=cv2.arcLength(approx,True)
            smoothness = smooth_perimeter/perimeter # smoothness
        # ZIG=[len(i) for i in contours] # alternative 
        # ct=1/NCH
        areas = [cv2.contourArea(c) for c in contours]
        if np.sum(areas)==0: 
            peris=[max(1,cv2.arcLength(c,True)) for c in contours]
            if np.sum(peris)==0: ct=0
            else: ct=max(1,cv2.arcLength(cnt,True))/np.sum(peris)
        else: ct=cv2.contourArea(cnt)/np.sum(areas)
                    
        pd=self.prefer_dict
        Score=ct*pd[0]+IOU*pd[1]+smoothness*pd[2]+pd[3] # base to encourage explore
        return Score

    def preference(self,mx,logger=None,step=None,idx=None): # from heuristics or human feedback
        ms=mx.squeeze(1).detach().cpu().numpy()
        if self.ppo_human_tune:
            name='Step-'+str(step)+'-'+str(idx)
            im=plot_mats(ms,title=name)
            logger.log_image(key="human tune", images=[im])
            done=False
            bs=len(ms)
            while not done:
                try:
                    x=input('Rating the results of '+name+' (, as seperator):')
                    l=[float(i) for i in x.split(',')]
                    if len(l)>1 and len(l)!=bs: raise Exception('Length must be 1 or batch size')
                    done=True
                except: print('Please retry')
            ratings=l*bs if len(l)==1 else l
        else: 
            ratings = list(map(self.ccs_score, ms))
        scores=torch.tensor(ratings).to(mx)
        return scores

    def reward(self,state,gt,m,logger=None,step=None,use_prefer=1,idx=None): #TODO
        x, _, inp, context, mcm, t, hs = state
        B,c_m,h,w=m.shape
        if self.reward_dict['loss']!=0:
            if self.use_ldm:
                mcs=m.reshape(-1,1,self.N,self.c_m,h,w).repeat(1,self.N,1,1,1,1) # for each, we get its competetors representation, or only within the class
                mask=1-torch.eye(self.N).unsqueeze(0).unsqueeze(3).repeat(1,1,1,self.c_m*h*w).to(x.device)
                mcm=(mask*mcs.reshape(-1,self.N,self.N,self.c_m*h*w)).reshape(-1,self.N,self.c_m,h,w) # competetors
            reward=0
            # estimation of loss per player
            mx=self.act(m) # B c h w
            mc=mcm.reshape(B,self.N,-1).sum(1).reshape(-1,self.c_m,h,w) # compatitors map
            mcx=self.act(mc)
            bmx=self.softjump(mx) if self.softjump is not None else mx
            bmx=self.blur(bmx) if self.blur is not None else bmx
            bmcx=self.softjump(mcx) if self.softjump is not None else mcx
            bmcx=self.blur(bmcx) if self.blur is not None else bmcx
            mstack=(mx+mcx).squeeze(1)
            bmstack=(bmx+bmcx).squeeze(1)
            rx=mstack-nn.ReLU()(mstack-1); r=rx # all agreed parts to 1
            mxs=nn.ReLU()(mx.sum([2,3])-self.quota).sum(1).reshape(B) # exceed resources used for each
            loss_reconstruct=0
            for i in self.loss_dict:
                if i=='focal': loss_i=focal_loss(rx,gt,alpha=self.focal_alpha).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='smoothl1': loss_i=nn.SmoothL1Loss(reduction='none')(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='dice': loss_i=SoftDiceLoss()(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='bce': loss_i=nn.BCELoss(reduction='none')(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='mse': loss_i=nn.MSELoss(reduction='none')(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                loss_reconstruct+=loss_i
            loss_overlap=self.alpha_overlap*nn.ReLU()(bmstack-1).sum([1,2]).reshape(B,-1).mean(1) # minimize blured overlap predicate-wise, avoid repeating
            loss_tae_l2=self.alpha_l2*(m**2).mean([1,2,3]).reshape(B,-1).mean(1) if self.alpha_l2!=0 else 0
            loss_tae_resources=self.alpha_resources*mxs if self.alpha_resources!=0 else 0
            loss_tae=loss_reconstruct+loss_overlap+loss_tae_l2+loss_tae_resources
        else: loss_tae=0
        # preference parts
        r_preference=self.preference(mx,logger,step,idx) if use_prefer and self.reward_dict['prefer']!=0 else 0
        for i in self.reward_dict:
            if i=='loss': reward_i=-loss_tae*self.reward_dict[i]
            if i=='prefer': reward_i=r_preference*self.reward_dict[i]
            reward+=reward_i
        return reward

    def update_ppo(self,total_steps,logger=None):
        if not self.use_rl or total_steps%self.ppo_update_every!=0: return

        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            for i in reversed(range(len(self.sampler.buffer.states))):
                state=self.sampler.buffer.states[i]
                next_state=self.sampler.buffer.next_states[i]
                d=self.sampler.buffer.is_terminals[i]
                if d==1: gae=0
                gt=self.sampler.buffer.gts[i]
                m=self.sampler.buffer.ms[i]
                vs = self.sampler.critic(state)
                vs_ = self.sampler.critic(next_state)
                vlogger=None if logger is None else logger.logger
                if not self.ppo_sparse_reward or d==1:
                    use_prefer=d if self.ppo_prefer_last else 1
                    reward=self.reward(next_state,gt,m,vlogger,total_steps,use_prefer,i)
                    if self.ppo_inc_reward:
                        m_in=self.sampler.buffer.m_in[i]
                        reward=reward-self.reward(state,gt,m_in,vlogger,total_steps,use_prefer,i)
                    if self.ppo_reward_norm:
                        reward = (reward - reward.mean())/(reward.std() + 1e-10) # batch norm, may global norm
                else: reward=0
                delta = reward + self.ppo_gamma * (1.0 - d) * vs_ - vs
                gae = delta + self.ppo_gamma * self.ppo_lamda * gae * (1.0 - d) ###
                gae = ((gae - gae.mean()) / (gae.std() + 1e-5)) # batch norm, may global norm
                self.sampler.buffer.advantages.insert(0,gae)
                v_target = gae + vs
                self.sampler.buffer.v_targets.insert(0,v_target)

        # Optimize policy for K epochs:
        for _ in range(self.ppo_K_epochs):
            seq=list(range(len(self.sampler.buffer.states)))
            random.shuffle(seq)
            for index in seq:
                state=self.sampler.buffer.states[index]
                m=self.sampler.actor(state,self.N)
                _,a_logprob_now,dist_entropy=self.sampler.sample_action(m)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - self.sampler.buffer.logprobs[index])  # shape(mini_batch_size X 1)

                advantage=self.sampler.buffer.advantages[index].reshape(-1,1,1,1)
                surr1 = ratios * advantage  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2) - self.ppo_entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                if logger is not None: logger.log('actor_loss',actor_loss.mean())
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.ppo_use_grad_clip:  # Trick 7: Gradient clip
                    actor_params=get_params_exclude(self.sampler,self.sampler.net.critic_head)
                    torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
                self.optimizer_actor.step()

                v_target=self.sampler.buffer.v_targets[index]
                v_s = self.sampler.critic(state)
                critic_loss = F.mse_loss(v_target, v_s)
                if logger is not None: logger.log('critic_loss',critic_loss.mean())
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.mean().backward()
                if self.ppo_use_grad_clip:  # Trick 7: Gradient clip
                    critic_params=get_params_exclude(self.sampler,self.sampler.net.actor_head)
                    torch.nn.utils.clip_grad_norm_(critic_params, 0.5)
                self.optimizer_critic.step()

        # clear buffer
        self.sampler.buffer.clear()
        
        if self.ppo_use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.ppo_max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.ppo_max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def rl_on(self,value):
        self.use_rl=value
        self.sampler.use_rl=value
        self.sampler.net.use_ac=value


def spl_contour(contour):
    x,y = contour.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x,y], u=None, s=1.0, per=1)
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = np.linspace(u.min(), u.max(), 25)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
    smoothened=np.asarray(res_array, dtype=np.int32)
    return smoothened

def get_params_exclude(net,module): # not safe, be careful
     module.requires_grad=False
     params=filter(lambda p: p.requires_grad, net.parameters())
     module.requires_grad=True
     return params


def y_to_img(y):
    color_dict = {
        0: [1, 1, 1],
        1: [0, 0, 1],
        2: [1, 0, 0],
        3: [0, 1, 0],
        4: [1, 1, 0],
        5: [.5, .5, .5],
        6: [.5, 0, .5],
        7: [1, .64, 0],
        8: [0, 1, 1],
        9: [.64, .16, .16],
        10: [1, 0, 1],
        11: [.5, .5, 0],
    }
    mat=y[0].argmax(0).detach().cpu().numpy()
    img = np.zeros((*mat.shape, 3))
    for k in range(mat.shape[0]):
        for l in range(mat.shape[1]):
            img[k, l] = np.array(color_dict[int(mat[k, l])])
    return img

def mat_to_img(mat):
    vmax=1.2
    vmin=0
    color=np.array([[[220, 178, 88]]])/255
    img = np.expand_dims(mat,2)
    img=((img-vmin)/(vmax-vmin))*color
    return 1-img

def plot_mat(mat,th=0.15,title=None):
    ax = plt.figure().add_subplot()
    mat[mat<th]=0
    ax.matshow(mat,cmap='Greys')
    ax.set_axis_off()
    ax.set_title(title)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

def plot_mats(mats,th=0.15,title=None,N_cols=8,fsize=4):
    bs=len(mats)
    cols=N_cols
    rows=int(np.ceil(bs/N_cols))
    fig = plt.figure(figsize=(fsize*cols,fsize*rows))
    for i in range(len(mats)):
        mat=mats[i]
        ax = fig.add_subplot(rows, cols, i+1)
        mat[mat<th]=0
        ax.matshow(mat,cmap='Greys')
        ax.set_axis_off()
        ax.set_title(title+':'+str(i))
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

def compose_img(model,y,rx,mx,mode='lineworld',N_cols=4):
    b,c,h,w=y.shape
    ind=random.choice(range(b))
    if model.c_m>1: 
        rx=rx[:,1:].sum(1)
        rx[rx>1]=1
        mx=mx.reshape(b,-1,model.c_m,h,w)[:,:,1:].sum(2)
    mats=torch.cat([rx[ind].unsqueeze(0),mx.reshape(b,-1,h,w)[ind]],0)
    mats=mats.detach().cpu().numpy()
    if mode=='lineworld': imgs=[y_to_img(y[ind].unsqueeze(0))]
    elif mode=='omniglot': 
        ymat=y[ind].reshape(h,w).detach().cpu().numpy()
        imgs=[mat_to_img(ymat)]
    imgs+=[mat_to_img(i) for i in mats]
    cols=N_cols
    rows=int(np.ceil(model.N/N_cols))+1
    composed=np.ones([rows*h+rows+1,cols*w+cols+1,3])*0.5
    composed[1:h+1,1:w+1]=imgs[0]
    composed[1:h+1,w+2:w*2+2]=imgs[1]
    count=2
    for row in range(1,rows):
        for col in range(cols):
            if count>=len(imgs): break
            img=imgs[count]
            composed[row*h+row+1:(row+1)*h+row+1,col*w+col+1:(col+1)*w+col+1]=img
            count+=1
    return composed

class TAETrainer(pl.LightningModule):
    def __init__(self,tae,opt,sched=None,lr=1e-3,dataset='lineworld',use_rl=False,**kwargs):
        super().__init__()
        self.model=tae
        self.dataset=dataset
        self.opt=opt # input a partial function, e.g. optim.AdamW
        self.sched=sched
        self.lr=lr
        self.forward=self.model.forward
        self.save_hyperparameters()
        self.use_rl=use_rl

    def training_step(self,batch,batch_idx):
        y, x_ = batch
        if self.current_epoch<self.model.rl_start_epoch: 
            self.model.rl_on(False)
        else: self.model.rl_on(self.use_rl)
        self.model.update_ppo(self.global_step+1,self)
        loss,log,[mx,rx,gt] = self.model.loss(x_,y,self.current_epoch)
        for i in log: self.log(i,log[i])
        with torch.no_grad():
            composed=compose_img(self.model,y,rx,mx,self.dataset)
        self.logger.log_image(key="train samples", images=[composed])
        return loss

    def validation_step(self, batch, batch_idx):
        y, x_ = batch
        loss,log,[mx,rx,gt] = self.model.loss(x_,y)
        composed=compose_img(self.model,y,rx,mx,self.dataset)
        self.logger.log_image(key="valid samples", images=[composed])
        self.log('val_loss',log['loss'])
        self.log('val_iou',log['iou'])
        self.log('val_mae',log['mae'])

    def configure_optimizers(self):
        if self.use_rl:
            params=get_params_exclude(self.model,self.model.sampler.net.critic_head)
        else: params=self.model.parameters()
        opt = self.opt(params, lr=self.lr)
        if self.sched is not None: sched = self.sched(opt)
        else: return opt
        return [opt], [sched]


class TAETunner(pl.LightningModule):
    def __init__(self,tae,opt,sched=None,lr=1e-3,dataset='lineworld_parse',draw_freq=None,use_rl=False):
        super().__init__()
        self.model=tae
        self.dataset=dataset.split('_')[0]
        self.opt=opt # input a partial function, e.g. optim.AdamW
        self.sched=sched
        self.lr=lr
        self.forward=self.model.forward
        self.save_hyperparameters()
        self.model.rl_on(False)

    def training_step(self,batch,batch_idx):
        x, o, r = batch
        loss,log,[mx,rx,gt] = self.model.parse_loss(x,o,r,epoch=self.current_epoch)
        for i in log: self.log(i,log[i])
        with torch.no_grad():
            composed=compose_img(self.model,x,rx,mx,self.dataset)
        self.logger.log_image(key="train samples", images=[composed])
        return loss

    def validation_step(self, batch, batch_idx):
        x, o, r = batch
        loss,log,[mx,rx,gt] = self.model.parse_loss(x,o,r)
        composed=compose_img(self.model,x,rx,mx,self.dataset)
        self.logger.log_image(key="valid samples", images=[composed])
        for i in log: self.log('val_'+i,log[i])

    def configure_optimizers(self):
        opt = self.opt(self.model.parameters(), lr=self.lr)
        if self.sched is not None: sched = self.sched(opt)
        else: return opt
        return [opt], [sched]


if __name__=='__main__':
    c_m=1
    c_in=3
    method='EM'
    embed_dim=256
    sigma=2
    K=3
    t=100
    memlen=3000
    NP1=6
    NP2=5
    NH1=4
    NH2=3
    N=8
    jump_alpha=25
    sampler_dim=32
    mapper_mid=256
    threshold=(0.0,1.0)
    use_small=True
    loss_option='mse_1.0' #'focal_1.0-dice_1.0-bce_1.0-smoothl1_1.0'
    use_ldm=True
    ldm_ds=[1,2,1,1]

    use_rl=True
    rl_start_epoch=0
    critic_mid=128
    ppo_gamma=0.95
    ppo_lamda=0.95
    ppo_K_epochs=3
    ppo_use_grad_clip=True
    ppo_use_lr_decay=False
    ppo_max_train_steps=1e6
    ppo_update_every=5
    reward_option='loss_1.0-prefer_1.0'
    lr_a=1e-3
    lr_c=1e-3

    sampler=Sampler(c_in=c_in,c_m=c_m,dim=sampler_dim,method=method,embed_dim=embed_dim,sigma=sigma,use_rl=use_rl,use_ldm=use_ldm,ldm_ds=ldm_ds,
        K=K,t=t,NP1=NP1,NP2=NP2,NH1=NH1,NH2=NH2,memlen=memlen,mapper_mid=mapper_mid,threshold=threshold,critic_mid=critic_mid)    
    tae=TAE(sampler,c_m=c_m,N=N,embed_dim=embed_dim,loss_option=loss_option,jump_alpha=jump_alpha,rl_start_epoch=rl_start_epoch,use_ldm=use_ldm,
            use_rl=use_rl,ppo_gamma=ppo_gamma,ppo_lamda=ppo_lamda,ppo_K_epochs=ppo_K_epochs,ppo_use_grad_clip=ppo_use_grad_clip,lr_a=lr_a,lr_c=lr_c,
        ppo_use_lr_decay=ppo_use_lr_decay,ppo_max_train_steps=ppo_max_train_steps,ppo_update_every=ppo_update_every,reward_option=reward_option).cuda()
    # model=TAETrainer(tae)
    
    total = sum([param.nelement() for param in tae.parameters()])
    print("T Number of parameter: %.2fM" % (total/1e6))
    total = sum([param.nelement() for param in tae.sampler.parameters()])
    print("S Number of parameter: %.2fM" % (total/1e6))
    if use_ldm:
        total = sum([param.nelement() for param in tae.sampler.encoder.parameters()])
        print("E Number of parameter: %.2fM" % (total/1e6))
        total = sum([param.nelement() for param in tae.sampler.decoder.parameters()])
        print("D Number of parameter: %.2fM" % (total/1e6))
    total = sum([param.nelement() for param in tae.sampler.mapper.parameters()])
    print("PM Number of parameter: %.2fM" % (total/1e6))
    total = sum([param.nelement() for param in tae.sampler.mapper.mapper.parameters()])
    print("PMM Number of parameter: %.2fM" % (total/1e6))
    total = sum([param.nelement() for param in tae.sampler.net.parameters()])
    print("SP Number of parameter: %.2fM" % (total/1e6))
    total = sum([param.nelement() for param in tae.sampler.net.actor_head.parameters()])
    print("AH Number of parameter: %.2fM" % (total/1e6))
    if use_rl:
        total = sum([param.nelement() for param in tae.sampler.net.critic_head.parameters()])
        print("CH Number of parameter: %.2fM" % (total/1e6))

    bs=8
    size=32
    x=torch.rand(bs,c_in,size,size).cuda() 
    y=torch.ones(bs,c_in,size,size).cuda() 
    o,r=[],[]
    for _ in range(bs):
        num=np.random.randint(2,5)
        o.append(torch.rand(num,1,size,size).cuda())
        rel=[]
        for i in range(num):
            for j in range(i+1,num):
                rel.append((i,j,np.random.randint(0,5)))
        r.append(rel)

    # tae.rl_on(False)
    # tae.parse_loss(x,o,r)

    # b,c,h,w=x.shape

    optimizer=optim.Adam(tae.parameters())
    optimizer.zero_grad()
    # m=tae(x)
    # rx=tae.act(m.reshape(b,-1,c_m,h,w).sum(1)) # mask reconstruct, allow empty masks
    # print(m.mean(),m.std())

    epoches=10
    for i in range(epoches):
        tae.update_ppo(i+1)
        # bs=np.random.randint(5,10)
        x=torch.rand(bs,c_in,size,size).cuda() 
        y=torch.ones(bs,c_in,size,size).cuda() 
        loss,log,[mx,rx,gt]=tae.loss(x,y,30000)
        loss.backward()
        optimizer.step()

    # print(mx.shape,rx.shape)
    # print(log)
    # compose_img(tae,y,rx,mx)

    