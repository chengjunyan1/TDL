import torch,torchvision
import torch.nn as nn
from torch.nn import functional as F
import functools as ft
from pytorch_metric_learning.distances import LpDistance
import pytorch_lightning as pl 
import torch.optim as optim
import numpy as np
import matplotlib.pylab as plt
import random
import io
from PIL import Image
# import faiss
from torch.distributions import Bernoulli,Normal
import open3d as o3d
from collections import Counter

from sampler import UNetSampler
from SDE import ScoreNet,marginal_prob_std,diffusion_coeff
from SDE3D import VoxScoreNet,Encoder,Decoder

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
 

def printt(*args):
    print('-'*30)
    print(args)
    print('-'*30)

def to_pc(m,th=0.5,rand=10): # just manu
    pc=[]
    h,w,d=m.shape
    for x in range(h):
        for y in range(w):
            for z in range(d):
                if m[x,y,z]>th:
                    if rand==0:
                        for p in [(x-0.5,y-0.5,z-0.5),(x-0.5,y-0.5,z+0.5),
                                (x-0.5,y+0.5,z-0.5),(x-0.5,y+0.5,z+0.5),
                                (x+0.5,y-0.5,y-0.5),(x+0.5,y-0.5,z+0.5),
                                (x+0.5,y+0.5,z-0.5),(x+0.5,y+0.5,z+0.5)]:
                            if p not in pc: pc.append(p)
                    else:
                        pc.append(np.array((x,y,z))+np.random.randn(3)/rand)
    return np.array(pc)

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
    def __init__(self, k, memlen=30000, emb_dim=256,Niter=10,use_zca=False):
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
        I,_=KMeans(db,self.k)
        clusters = [[] for i in range(self.k)]
        for i in range(len(data)):
            clusters[I[i]].append(i)
        # centroids
        C=[]
        for i in clusters: 
            if i!=[]: C.append(D[i].mean(0))
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

class PMapper(nn.Module): # m to predicate space
    def __init__(self,c_m=1,mid=256,embed_dim=256,NP1=6,NP2=6,NH1=6,NH2=6,NK=2,dropout=0.1,
                 memlen=30000,threshold=(0.01,0.1),lambda_P=2e-2,Niter=10,use_zca=False,use_small=False):
        super().__init__()
        self.mapper=nn.Sequential(
            nn.Conv3d(c_m, mid, 3, stride=2),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d(1),
        ) if use_small else nn.Sequential(
            nn.Conv3d(c_m, mid, 3, stride=2),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(mid, mid, 1, stride=1),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(mid, mid, 3, stride=2),
            nn.GroupNorm(32,mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool3d(1),
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
        b,N,c,h,w,d=m.shape
        q1=self.Q1(self.mapper(m.reshape([-1,c,h,w,d]))[:,:,0,0,0])
        mp=m.unsqueeze(1).repeat(1,N,1,1,1,1,1)+m.unsqueeze(2).repeat(1,1,N,1,1,1,1)
        mp=(mp-nn.ReLU()(mp-1)).reshape(-1,c,h,w,d) # remove repeat
        q2=self.Q2(self.mapper(mp)[:,:,0,0,0])
        return q1,q2
    
    def embedH(self,m):
        b,N,c,h,w,d=m.shape
        q1,q2=None,None
        if self.NH1>0: q1=self.QH1(self.mapper(m.reshape([-1,c,h,w,d]))[:,:,0,0,0])
        if self.NH2>0:
            mp=m.unsqueeze(1).repeat(1,N,1,1,1,1,1)+m.unsqueeze(2).repeat(1,1,N,1,1,1,1)
            mp=(mp-nn.ReLU()(mp-1)).reshape(-1,c,h,w,d) # remove repeat
            q2=self.QH2(self.mapper(mp)[:,:,0,0,0])
        return q1,q2

    def forward(self,m,mx=None,loss=False):
        b,N,c,h,w,d=m.shape
        q1,q2=self.embed(m)
        pred1,dist1,p1=self.P1.pred(q1)
        pred2,dist2,p2=self.P2.pred(q2)
        p1=p1.reshape(b,N,-1)
        pr=p1.matmul(self.P1.P) # p representation
        p2m,_=pred2.max(-1)
        p2m=p2m.reshape(b,N,N)
        # d2d=p2m*(1-torch.eye(N)).unsqueeze(0).to(m) # max as the prob 
        # gr=torch.einsum('ijkpqd,iwj->iwkpqd',m,d2d) # group representation
        gr=torch.einsum('ijkpqd,iwj->iwkpqd',m,p2m) # group representation
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
            # ghr=torch.einsum('ijkpqd,iwj->iwkpqd',gm,d2dH) # group representation
            ghr=torch.einsum('ijkpqd,iwj->iwkpqd',gr,pH2m) # group representation
        if loss:
            with torch.no_grad():
                ma=mx.reshape(b*N,-1,h,w,d).sum([1,2,3,4])
                low,high=self.threshold[0]*h*w*d*c,self.threshold[1]*h*w*d*c
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
                    ga=gx.reshape(b*N,-1,h,w,d).sum([1,2,3,4])
                    low,high=self.threshold[0]*h*w*d*c,self.threshold[1]*h*w*d*c*N # higher high 
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


#----- Smaplers


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


class VoxSampler(nn.Module): # [x;z]->dz
    def __init__(
            self,
            c_in=1,
            c_m=1,
            dim=32,
            embed_dim=256,
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
            Niter=10,
            use_zca=False,
            use_attn=False, 
            use_self_attn=False, 
            n_heads=8, 
            d_head=16,
            context_dim=256, 
            share_mapper=False,
            use_ldm=False,
            ldm_out=8,
            ldm_ds=[1,2,1,1],
            mapper_small=False,
            # RL
            use_rl=False,
            critic_mid=128,
            **kwargs
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
        mapper=self.mapper.mapper if share_mapper else None
        self.use_rl=use_rl
        if use_rl: self.buffer=RolloutBuffer()
        self.use_ldm=use_ldm
        mul=4 if NH2>0 else 3
        in_channels=c_in+c_m*mul
        out_channels=c_m
        if use_ldm:
            self.ldm_out=ldm_out
            self.encoder=Encoder(c_in=c_in,dim=dim,dropout=dropout,use_attn=use_attn,n_heads=n_heads,d_head=d_head,ds=ldm_ds)
            self.decoder=Decoder(in_channels=ldm_out,c_out=out_channels,dim=dim,use_attn=use_attn,n_heads=n_heads,dropout=dropout,d_head=d_head,sf=ldm_ds[::-1])
            in_channels,out_channels=dim+ldm_out*mul,ldm_out
        self.net=VoxScoreNet(self.marginal_prob_std_fn,in_channels,out_channels,dropout=dropout,use_t=self.use_t,
            dim=dim,embed_dim=embed_dim,use_out_res=use_out_res,use_attn=use_attn,use_self_attn=use_self_attn,
            n_heads=n_heads,d_head=d_head,context_dim=context_dim,mapper=mapper,use_ac=use_rl,mid=critic_mid)
        self.eps=eps
        self.snr=snr
        self.method=method

    def get_input(self,x,m,N,context):
        b,c,h,w,d=x.shape
        c_m=self.ldm_out if self.use_ldm else self.c_m
        mcs=m.reshape(-1,1,N,c_m,h,w,d).repeat(1,N,1,1,1,1,1) # for each, we get its competetors representation, or only within the class
        mask=1-torch.eye(N).unsqueeze(0).unsqueeze(3).repeat(1,1,1,c_m*h*w*d).to(x.device)
        mcm=(mask*mcs.reshape(-1,N,N,c_m*h*w*d)).reshape(-1,N,c_m,h,w,d) # competetors
        mc=mcm.reshape(b,N,-1).sum(1).reshape(-1,c_m,h,w,d) # compatitors map
        pr,gr,phr,ghr=self.mapper(m.reshape(-1,N,c_m,h,w,d)) # d1 d2 clustering
        mg=gr.reshape(-1,c_m,h,w,d)
        context=context+pr.reshape(b,-1)
        if phr is not None: context=context+phr.reshape(b,-1)
        if ghr is not None: 
            mgh=ghr.reshape(-1,c_m,h,w,d)
            inp=torch.cat([x,m,mc,mg,mgh],dim=1)
        else: inp=torch.cat([x,m,mc,mg],dim=1)
        return inp, context, mcm 
    
    def scoring(self,x,m,N,context,t=None,hs=None):
        inp, context, mcm = self.get_input(x,m,N,context) # use current state
        hs=[h.detach() for h in hs] if hs is not None else None
        state=[x.detach(), m.detach(), inp.detach(), context.detach(), mcm.detach(), t, hs]
        return self.net(inp, context, t, mcm), state # current state

    @torch.no_grad()
    def get_state(self,x,m,N,context,t=None, hs=None):
        inp, context, mcm = self.get_input(x,m,N,context)
        return [x, m, inp, context, mcm,t,hs]

    def sample_action(self, m):
        mx=nn.Sigmoid()(m)
        dist = Bernoulli(mx) # For c_m>1, categorical
        action = dist.sample()
        entropy=dist.entropy()
        action_logprob = dist.log_prob(action)
        return action,action_logprob,entropy
      
    def actor(self,state, N):
        x, m, inp, context, mcm, t, hs = state
        m, state=self.scoring(x,m,N,context,t,hs)
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
        b,c,h,w,d=x.shape
        time_steps = torch.linspace(1., self.eps, 2).to(x.device)
        c_m=self.ldm_out if self.use_ldm else self.c_m
        m_init=torch.randn(b,c_m,h,w,d).to(x.device) * self.marginal_prob_std_fn(time_steps[0],x)#[:, None, None, None]
        m, state=self.scoring(x,m,N,context,hs=hs)
        if self.use_rl and self.training: 
            assert gt is not None
            with torch.no_grad():
                next_state=self.get_state(x,m,N,context,hs=hs)
                self.buffer.states.append(state)
                self.buffer.next_states.append(next_state)
                self.buffer.gts.append(gt)
                m_a=m
                if self.use_ldm: m_a=self.decoder(m,hs[0],hs[1],hs[2])
                m_in=torch.rand_like(m_a).to(x.device) * self.marginal_prob_std_fn(time_steps[0],x) if self.use_ldm else m_init
                self.buffer.m_in.append(m_in)
                self.buffer.ms.append(m_a)
                action,action_logprob,entropy=self.sample_action(m_a)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.is_terminals.append(1)
        return [m],None,None

    def sde(self,x,context,N,gt=None,hs=None):
        b,c,h,w,d=x.shape
        time_steps = torch.linspace(1., self.eps, self.K+2).to(x.device)
        c_m=self.ldm_out if self.use_ldm else self.c_m
        m=torch.randn(b,c_m,h,w,d).to(x.device) * self.marginal_prob_std_fn(time_steps[0],x)#[:, None, None, None]
        m_init=m
        m_in=None 
        step_size = (time_steps[0] - time_steps[1])*self.K/self.t
        ms,ss,ts=[],[],[]
        for i, time_step in enumerate(time_steps[1:-1]):      
            batch_time_step = torch.ones(b).to(x.device) * time_step
            ts.append(batch_time_step)
            grad,state=self.scoring(x, m, N, context, batch_time_step,hs)
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
                mean_m = m + (g**2)[:, None, None, None, None] * grad * step_size
                if self.method in ['EM','PC']:
                    m = mean_m + torch.sqrt(step_size) * g[:, None, None, None, None] * torch.randn_like(m)
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
                    if self.use_ldm: m_a=self.decoder(m,hs[0],hs[1],hs[2])
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
        loss = torch.mean(torch.sum((ss * std[:, None, None, None, None] + z)**2, dim=(1,2,3,4)))
        return loss


#----- TAE

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

def softjump(x,alpha=25,shift=0.2,smooth=False): 
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


class VoxTAE(nn.Module):
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
            jump_smooth=True,
            quota=8, # quota for each player
            beta=1.0, # sde overall
            gamma_cluster=1.0, # clustering overall
            cluster_start_epoch=20, 
            PE_mode='rand', # rand fix none
            loss_option='focal_1.0-dice_1.0', # e.g. focal_1.0-dice_1.0-bce_0.1-smoothl1_1.0:b 
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
            ppo_pc_rand=10,
            ppo_max_train_steps=1e6,
            ppo_update_every=5,
            reward_option='loss_1.0-prefer_1.0', # e.g. loss_1.0-preference_1.0, predict minus loss and human preference 
            prefer_option='ct_1.0-cp_1.0-sm_1.0-z_0.0', # continous, compactness, smoothness, base score
            lr_a=1e-3,
            lr_c=1e-3,
            **kwargs,
        ):
        super().__init__()
        self.sampler=sampler
        self.alpha_overlap=alpha_overlap
        self.alpha_l2=alpha_l2
        self.alpha_resources=alpha_resources
        self.focal_alpha=focal_alpha
        self.softjump=None if jump_alpha is None else ft.partial(softjump,alpha=jump_alpha,shift=jump_shift,smooth=jump_smooth)
        self.quota=quota
        self.beta=beta
        self.gamma_cluster=gamma_cluster
        self.c_m=c_m
        self.c_in=sampler.c_in
        self.N=N
        self.embed_dim=embed_dim
        self.act=nn.Sigmoid()
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
            self.ppo_use_lr_decay=ppo_use_lr_decay
            self.ppo_max_train_steps=ppo_max_train_steps
            self.ppo_update_every=ppo_update_every
            self.ppo_reward_norm=ppo_reward_norm
            self.ppo_inc_reward=ppo_inc_reward
            self.ppo_sparse_reward=ppo_sparse_reward
            self.ppo_human_tune=ppo_human_tune
            self.ppo_prefer_last=ppo_prefer_last
            self.ppo_pc_rand=ppo_pc_rand
            actor_params=get_params_exclude(self.sampler,self.sampler.net.critic_head)
            critic_params=get_params_exclude(self.sampler,self.sampler.net.actor_head)
            self.optimizer_actor = torch.optim.Adam(actor_params, lr=lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(critic_params, lr=lr_c, eps=1e-5)
            self.reward_option=reward_option
            reward_option=self.reward_option.split(':')[0].split('-')
            self.reward_dict={}
            for i in reward_option: self.reward_dict[i.split('_')[0]]=float(i.split('_')[1])
            self.prefer_dict=[float(i.split('_')[1]) for i in prefer_option.split(':')[0].split('-')]

    def forward(self,x,gt=None,ret_all=False): # x is image BCHW 
        x=x.to(device=x.device, dtype=torch.float)
        x=x[:,:self.c_in]
        b,c,h,w,d=x.shape
        if self.PE_mode=='none': PE=0
        else:
            if self.PE_mode=='fix':
                pos=torch.range(0,self.N-1).reshape(1,-1).repeat(b,1).reshape(-1,1)
            elif self.PE_mode=='rand':
                pos=torch.randint(0,256,[b*self.N,1]) # randomly assign player id
            PE=self.posemb(pos.long().to(x.device)).squeeze(1)
        X=x.unsqueeze(1).repeat(1,self.N,1,1,1,1).reshape(-1,c,h,w,d) # each batch copy x
        GT=gt.unsqueeze(1).repeat(1,self.N,1,1,1).reshape(-1,h,w,d) if gt is not None else None
        m,ss,ts,m_l=self.sampler(X,PE,self.N,GT)
        if ret_all: return m,ss,ts,m_l
        return m
    
    def relpred(self,m): return self.sampler.rel_pred(m)

    def loss(self,x,y,epoch=None): # step*batchsize
        x,y=x[:,:self.c_in],y[:,:self.c_in]
        gt=y.squeeze(1).float().to(x.device) 
        loss_option=self.loss_option.split(':')[0].split('-')
        loss_dict={}
        for i in loss_option: loss_dict[i.split('_')[0]]=float(i.split('_')[1])
        b,c,h,w,d=x.shape
        m,ss,ts,m_l=self(x,gt=gt,ret_all=True)
        mx=self.act(m)
        log={}
        loss=0

        #--- reconstruct ---
        mstack=mx.reshape(b,-1,1,h,w,d).sum([1]).squeeze(1)
        bmx=self.softjump(mx) if self.softjump is not None else mx
        bmstack=bmx.reshape(b,-1,1,h,w,d).sum([1]).squeeze(1)
        rx=mstack-nn.ReLU()(mstack-1); r=rx # all agreed parts to 1
        mxs=nn.ReLU()(mx.reshape(b,-1,h,w,d).sum([2,3,4])-self.quota).sum(1).mean() # exceed resources used for each
        loss_reconstruct=0
        for i in loss_dict:
            if i=='focal': loss_i=focal_loss(rx,gt,alpha=self.focal_alpha).mean()*loss_dict[i]
            elif i=='smoothl1': loss_i=nn.SmoothL1Loss()(rx,gt).mean()*loss_dict[i]
            elif i=='dice': loss_i=SoftDiceLoss()(rx,gt).mean()*loss_dict[i]
            elif i=='bce': loss_i=nn.BCELoss()(rx,gt).mean()*loss_dict[i]
            log.update({i:loss_i})
            loss_reconstruct+=loss_i
        loss_overlap=self.alpha_overlap*nn.ReLU()(bmstack-1).sum([1,2,3]).mean() # minimize overlap predicate-wise, avoid repeating
        loss_tae_l2=self.alpha_l2*(m**2).mean() if self.alpha_l2!=0 else 0
        loss_tae_resources=self.alpha_resources*mxs.mean() if self.alpha_resources!=0 else 0
        loss_tae=loss_reconstruct+loss_overlap+loss_tae_l2+loss_tae_resources
        loss+=loss_tae # 'loss_tae':loss_tae,
        log.update({'loss_reconstruct':loss_reconstruct,'loss_overlap':loss_overlap,
                    'loss_tae_resources':loss_tae_resources,'loss_tae_l2':loss_tae_l2})

        #--- clustering ---
        if self.gamma_cluster>0 and epoch is not None and epoch>=self.cluster_start_epoch:
            B,c_m,h_m,w_m,d_m=m_l.shape
            loss_cluster=self.sampler.mapper(m_l.reshape(-1,self.N,c_m,h_m,w_m,d_m),mx,loss=True)*self.gamma_cluster
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
        log.update({'loss':loss,'iou':iou})
        return loss,log,[mx,rx,gt]

    def ccs_score(self, m): 
        pc=to_pc(m,rand=self.ppo_pc_rand)
        if len(pc)==0: return 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        labels = np.array(pcd.cluster_dbscan(eps=1.8, min_points=1))
        NC = labels.max()+1 # continous
        mv,_=Counter(labels).most_common()[0]
        mpc=pc[np.where(labels==mv)[0]]
        pcm = o3d.geometry.PointCloud()
        pcm.points = o3d.utility.Vector3dVector(mpc)
        hull, _ = pcm.compute_convex_hull()
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcm, alpha=1.8)
        mesh_area=mesh.get_surface_area()
        if mesh_area==0: smoothness=0
        else: smoothness=hull.get_surface_area()/mesh_area
        hull_area=hull.get_volume()
        if hull_area==0: IOU=0
        else: IOU=len(mpc)/hull_area
        ct=1/NC+mesh.is_watertight()

        pd=self.prefer_dict
        Score=ct*pd[0]+IOU*pd[1]+smoothness*pd[2]+pd[3] # base to encourage explore
        return Score

    def preference(self,mx,logger=None,step=None,idx=None): # from heuristics or human feedback
        ms=mx.squeeze(1).detach().cpu().numpy()
        if self.ppo_human_tune:
            name='Step-'+str(step)+'-'+str(idx)
            im=plot_voxels(ms,title=name)
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
        B,c_m,h,w,d=m.shape
        if self.reward_dict['loss']!=0:
            if self.use_ldm:
                mcs=m.reshape(-1,1,self.N,self.c_m,h,w,d).repeat(1,self.N,1,1,1,1,1) # for each, we get its competetors representation, or only within the class
                mask=1-torch.eye(self.N).unsqueeze(0).unsqueeze(3).repeat(1,1,1,self.c_m*h*w*d).to(x.device)
                mcm=(mask*mcs.reshape(-1,self.N,self.N,self.c_m*h*w*d)).reshape(-1,self.N,self.c_m,h,w,d) # competetors
            reward=0
            # estimation of loss per player
            mx=self.act(m) # B c h w
            mc=mcm.reshape(B,self.N,-1).sum(1).reshape(-1,self.c_m,h,w,d) # compatitors map
            mcx=self.act(mc)
            bmx=self.softjump(mx) if self.softjump is not None else mx
            bmcx=self.softjump(mcx) if self.softjump is not None else mcx
            mstack=(mx+mcx).squeeze(1)
            bmstack=(bmx+bmcx).squeeze(1)
            rx=mstack-nn.ReLU()(mstack-1); r=rx # all agreed parts to 1
            mxs=nn.ReLU()(mx.sum([2,3,4])-self.quota).sum(1).reshape(B) # exceed resources used for each
            loss_reconstruct=0
            for i in self.loss_dict:
                if i=='focal': loss_i=focal_loss(rx,gt,alpha=self.focal_alpha).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='smoothl1': loss_i=nn.SmoothL1Loss(reduction='none')(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='dice': loss_i=SoftDiceLoss()(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='bce': loss_i=nn.BCELoss(reduction='none')(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                elif i=='mse': loss_i=nn.MSELoss(reduction='none')(rx,gt).reshape(B,-1).mean(1)*self.loss_dict[i]
                loss_reconstruct+=loss_i
            loss_overlap=self.alpha_overlap*nn.ReLU()(bmstack-1).sum([1,2,3]).reshape(B,-1).mean(1) # minimize overlap predicate-wise, avoid repeating
            loss_tae_l2=self.alpha_l2*(m**2).mean([1,2,3,4]).reshape(B,-1).mean(1) if self.alpha_l2!=0 else 0
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

    def update_ppo(self,x,total_steps,logger=None):
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
                gae = delta + self.ppo_gamma * self.ppo_lamda * gae * (1.0 - d)
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

                advantage=self.sampler.buffer.advantages[index].reshape(-1,1,1,1,1)
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


def get_params_exclude(net,module): # not safe, be careful
     module.requires_grad=False
     params=filter(lambda p: p.requires_grad, net.parameters())
     module.requires_grad=True
     return params

def plot_voxel(mat,th=0.15,title=None):
    ax = plt.figure().add_subplot(projection='3d')
    colors = np.empty(list(mat.shape)+[3])
    colors[:,:,:]=np.array([[[255, 255, 255]]])/255
    mat[mat<th]=0
    colors=np.concatenate([colors,np.expand_dims(mat,3)],axis=3)
    ax.voxels(mat, facecolors=colors, edgecolor=[0,0,0,0])
    ax.set_axis_off()
    ax.set_title(title)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

def plot_voxels(mats,th=0.15,title=None,N_cols=8,fsize=4):
    bs=len(mats)
    cols=N_cols
    rows=int(np.ceil(bs/N_cols))
    fig = plt.figure(figsize=(fsize*cols,fsize*rows))
    for i in range(len(mats)):
        mat=mats[i]
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        colors = np.empty(list(mat.shape)+[3])
        colors[:,:,:]=np.array([[[255, 255, 255]]])/255
        mat[mat<th]=0
        colors=np.concatenate([colors,np.expand_dims(mat,3)],axis=3)
        ax.voxels(mat, facecolors=colors, edgecolor=[0,0,0,0])
        ax.set_axis_off()
        ax.set_title(title+':'+str(i))
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

def compose_fig(model,y,rx,mx,N_cols=4,fsize=4,th1=0.5,th2=0.15):
    b,c,h,w,d=y.shape
    ind=random.choice(range(b))
    mats=torch.cat([y[ind].reshape(h,w,d).unsqueeze(0),
        rx[ind].unsqueeze(0),mx.reshape(b,-1,h,w,d)[ind]]).detach().cpu().numpy()
    cols=N_cols
    rows=int(np.ceil(model.N/N_cols))
    fig = plt.figure(figsize=(fsize*cols,fsize*rows))
    for i in range(len(mats)):
        mat=mats[i]
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        colors = np.empty(list(mat.shape)+[3])
        colors[:,:,:]=np.array([[[255, 255, 255]]])/255
        if i<2: mat[mat<th1]=0
        else: mat[mat<th2]=0
        colors=np.concatenate([colors,np.expand_dims(mat,3)],axis=3)
        ax.voxels(mat, facecolors=colors, edgecolor=[0,0,0,0])
        ax.set_axis_off()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

class VoxTAETrainer(pl.LightningModule):
    def __init__(self,tae,opt,sched=None,lr=1e-3,dataset='lineworld',draw_freq=1,use_rl=False):
        super().__init__()
        self.save_hyperparameters()
        self.model=tae
        self.dataset=dataset
        self.opt=opt # input a partial function, e.g. optim.AdamW
        self.sched=sched
        self.lr=lr
        self.forward=self.model.forward
        self.save_hyperparameters()
        self.valid_cache=[]
        self.draw_freq=draw_freq
        self.use_rl=use_rl

    def training_step(self,batch,batch_idx):
        y, x_ = batch
        if self.current_epoch<self.model.rl_start_epoch: 
            self.model.rl_on(False)
        else: self.model.rl_on(self.use_rl)
        self.model.update_ppo(x_,self.global_step+1,self)
        loss,log,[mx,rx,gt] = self.model.loss(x_,y,self.current_epoch)
        for i in log: self.log(i,log[i])
        return loss

    def validation_step(self, batch, batch_idx):
        y, x_ = batch
        loss,log,[mx,rx,gt] = self.model.loss(x_,y)
        self.log('val_loss',log['loss'])
        self.log('val_iou',log['iou'])
        self.valid_cache.append([y,rx,mx])

    def on_validation_epoch_end(self):
        if self.current_epoch%self.draw_freq==0:
            [y,rx,mx]=random.choice(self.valid_cache)
            im=compose_fig(self.model,y,rx,mx)
            self.logger.log_image(key="samples", images=[im])
        self.valid_cache.clear()  # free memory

    def configure_optimizers(self):
        if self.use_rl:
            params=get_params_exclude(self.model,self.model.sampler.net.critic_head)
        else: params=self.model.parameters()
        opt = self.opt(params, lr=self.lr)
        if self.sched is not None: sched = self.sched(opt)
        else: return opt
        return [opt], [sched]



if __name__=='__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    c_m=1
    c_in=1
    method='EM'
    embed_dim=256
    sigma=2
    K=6
    K=6
    t=100
    memlen=3000
    NP1=6
    NP2=5
    NH1=4
    NH2=3
    N=6
    use_zca=False
    use_zca=False
    sampler_dim=32
    mapper_mid=256
    threshold=(0.0,1.0)
    use_small=True
    jump_alpha=25
    loss_option='focal_1.0-dice_1.0-bce_1.0-smoothl1_1.0'
    use_ldm=True
    ldm_ds=[1,2,2,2]
    ldm_ds=[1,2,2,2]

    use_rl=False
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

    sampler=VoxSampler(c_in=c_in,c_m=c_m,dim=sampler_dim,method=method,embed_dim=embed_dim,sigma=sigma,use_zca=use_zca,use_rl=use_rl,ldm_ds=ldm_ds,
        K=K,t=t,NP1=NP1,NP2=NP2,NH1=NH1,NH2=NH2,memlen=memlen,mapper_mid=mapper_mid,threshold=threshold,critic_mid=critic_mid,use_ldm=use_ldm)    
    tae=VoxTAE(sampler,c_m=c_m,N=N,embed_dim=embed_dim,loss_option=loss_option,jump_alpha=jump_alpha,rl_start_epoch=rl_start_epoch,use_ldm=use_ldm,
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


    bs=32
    size=64
    x=torch.rand(bs,c_in,size,size,size).cuda() 
    y=torch.rand(bs,c_in,size,size,size).cuda() 
    b,c,h,w,d=x.shape

    optimizer=optim.Adam(tae.parameters())
    optimizer.zero_grad()
    # m=tae(x)
    # loss,log,[mx,rx,gt]=tae.loss(x,y,30000)
    # rx=tae.act(m.reshape(b,-1,c_m,h,w,d).sum(1)) # mask reconstruct, allow empty masks
    # print(m.mean(),m.std())

    # t0=time.time()
    # epochs=5
    # for i in range(epochs):
    #     print(i)
    #     tae.update_ppo(x,i+1)
    #     loss,log,[mx,rx,gt]=tae.loss(x,y,30000)
    #     loss.backward()
    #     optimizer.step()
    # print(time.time()-t0)

    # with profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         loss,log,[mx,rx,gt]=tae.loss(x,y,30000)
    #         loss.backward()
    #         optimizer.step()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    tae.eval()
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            tae(x)
            # loss,log,[mx,rx,gt]=tae.loss(x,y,30000)
            # loss.backward()
            # optimizer.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # print(mx.shape,rx.shape)
    # print(log)

    # compose_img(tae,y,rx,mx)

    