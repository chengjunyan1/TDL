# Adopted from https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=zOsoqPdXHuL5
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools as ft
from attention import SamplerCrossAttention,SamplerSelfAttention



class Encoder(nn.Module):
  def __init__(self, c_in, dim=32,dropout=0,use_attn=False, n_heads=8, d_head=16, ds=[1,2,1,1]):
    super().__init__()
    Down=ft.partial(nn.Conv2d, bias=False, padding=1)
    # embed_dim=embed_dim*2 # for concat
    self.use_attn=use_attn
    if use_attn: Attn=ft.partial(SamplerSelfAttention,n_heads=n_heads,d_head=d_head,dropout=dropout)
    # Encoding layers where the resolution decreases
    self.conv1 = Down(c_in, dim, 3, stride=ds[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=dim)
    if use_attn: self.attn1=Attn(in_channels=dim)
    self.conv2 = Down(dim, dim, 3, stride=ds[1])
    self.gnorm2 = nn.GroupNorm(dim, num_channels=dim)
    if use_attn: self.attn2=Attn(in_channels=dim)
    self.conv3 = Down(dim, dim, 3, stride=ds[2])
    self.gnorm3 = nn.GroupNorm(dim, num_channels=dim)
    if use_attn: self.attn3=Attn(in_channels=dim)
    self.conv4 = Down(dim, dim, 3, stride=ds[3])
    self.act = nn.SiLU()

  def forward(self, x):  
    h1 = self.conv1(x)    
    ## Incorporate information from t
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    if self.use_attn: h1=self.attn1(h1)
    h2 = self.conv2(h1)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    if self.use_attn: h2=self.attn2(h2)
    h3 = self.conv3(h2)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    if self.use_attn: h3=self.attn3(h3)
    z = self.conv4(h3)

    return z,h1,h2,h3


class Decoder(nn.Module):
  def __init__(self,in_channels,c_out,dim,use_attn,n_heads,dropout,d_head,sf=[1,1,2,1]):
    super().__init__()
    Up=Upsample#nn.ConvTranspose2d
    self.use_attn=use_attn
    if use_attn: Attn=ft.partial(SamplerSelfAttention,n_heads=n_heads,d_head=d_head, dropout=dropout)
    # Decoding layers where the resolution increases
    self.tconv4 = Up(in_channels, dim, 3, bias=False,scale_factor=sf[0])
    self.tgnorm4 = nn.GroupNorm(dim, num_channels=dim)
    if use_attn: self.attn5=Attn(in_channels=dim)
    self.tconv3 = Up(dim*2, dim, 3, bias=False,scale_factor=sf[1])#, output_padding=1)    
    self.tgnorm3 = nn.GroupNorm(dim, num_channels=dim)
    if use_attn: self.attn6=Attn(in_channels=dim)
    self.tconv2 = Up(dim*2, dim, 3, bias=False,scale_factor=sf[2])#, output_padding=1)    
    self.tgnorm2 = nn.GroupNorm(dim, num_channels=dim)
    if use_attn: self.attn7=Attn(in_channels=dim)
    self.tconv1 = Up(dim*2, c_out, 3,scale_factor=sf[3])
    self.act = nn.SiLU()

  def forward(self,z,h1,h2,h3):
    # Decoding path
    h = self.tconv4(z)
    ## Skip connection from the encoding path
    h = self.tgnorm4(h)
    h = self.act(h)
    if self.use_attn: h=self.attn5(h)
    h = self.tconv3(padcat(h, h3))
    h = self.tgnorm3(h)
    h = self.act(h)
    if self.use_attn: h=self.attn6(h)
    h = self.tconv2(padcat(h, h2))
    h = self.tgnorm2(h)
    h = self.act(h)
    if self.use_attn: h=self.attn7(h)
    h = self.tconv1(padcat(h, h1))
    return h


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.silu=nn.SiLU()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


def padcat(x1,x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2]) 
    return torch.cat([x1, x2], dim=1)
  

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, out_channels, kernel_size=3, padding=1, bias=True,scale_factor=2):
        super().__init__()
        self.channels = channels
        self.scale_factor=scale_factor
        self.conv = nn.Conv2d(self.channels, out_channels, kernel_size, bias=bias,padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.scale_factor>1: x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x
    


class CriticHead(nn.Module):
  def __init__(self,c_out,mid,dim,embed_dim,use_attn,use_self_attn,n_heads,mapper,dropout,d_head,context_dim):
      super().__init__()
      self.use_attn=use_attn
      if use_attn: Attn=ft.partial(SamplerCrossAttention, use_self_attn=use_self_attn,n_heads=n_heads,
                d_head=d_head, context_dim=context_dim, mapper=mapper,dropout=dropout, c_m=c_out)
      channels=[dim, dim*2, dim*4, dim*8]
      self.cup3 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.SiLU(),
        Dense(channels[2], mid)
      ) 
      self.cup2 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.SiLU(),
        Dense(channels[1], mid)
      ) 
      self.cup1 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.SiLU(),
        Dense(channels[0], mid)
      ) 
      Conv=ft.partial(nn.Conv2d, bias=True, padding=1)
      self.cconv1 = Conv(channels[3], mid, 3, stride=1)
      self.cdense1 = Dense(embed_dim, mid)
      self.cgnorm1 = nn.GroupNorm(32, num_channels=mid)
      if use_attn: self.cattn1=Attn(in_channels=mid)
      self.cconv2 = Conv(mid, mid, 3, stride=1)
      self.cdense2 = Dense(embed_dim, mid)
      self.cgnorm2 = nn.GroupNorm(32, num_channels=mid)
      if use_attn: self.cattn2=Attn(in_channels=mid)
      self.cconv3 = Conv(mid, mid, 3, stride=1)
      self.cdense3 = Dense(embed_dim, mid)
      self.cgnorm3 = nn.GroupNorm(32, num_channels=mid)
      if use_attn: self.cattn3=Attn(in_channels=mid)
      self.cout=nn.Sequential(
          nn.AdaptiveAvgPool2d(1),
          nn.Flatten(),
          nn.SiLU(),
          Dense(mid, 1),
          nn.Flatten(),
      )
      self.cact = nn.SiLU()

  def forward(self,h1,h2,h3,h4,embed,mcm):
    c = self.cconv1(h4)    
    c += self.cdense1(embed)+self.cup3(h3)
    c = self.cact(self.cgnorm1(c))
    if self.use_attn: c=self.cattn1(c,mcm)
    c = self.cconv2(c)
    c += self.cdense2(embed)+self.cup2(h2)
    c = self.cact(self.cgnorm2(c))
    if self.use_attn: c=self.cattn2(c,mcm)
    c = self.cconv3(c)
    c += self.cdense3(embed)+self.cup1(h1)
    c = self.cact(self.cgnorm3(c))
    if self.use_attn: c=self.cattn3(c,mcm)
    cout=self.cout(c)
    return cout.squeeze(1)


class ActorHead(nn.Module):
  def __init__(self,c_out,marginal_prob_std,use_t,dim,embed_dim,use_attn,use_self_attn,n_heads,mapper,dropout,d_head,context_dim):
    super().__init__()
    Up=Upsample#nn.ConvTranspose2d
    self.use_attn=use_attn
    self.use_t=use_t
    if use_attn: Attn=ft.partial(SamplerCrossAttention, use_self_attn=use_self_attn,n_heads=n_heads,
              d_head=d_head, context_dim=context_dim, mapper=mapper,dropout=dropout, c_m=c_out)
    channels=[dim, dim*2, dim*4, dim*8]
    # Decoding layers where the resolution increases
    self.tconv4 = Up(channels[3], channels[2], 3, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(dim, num_channels=channels[2])
    if use_attn: self.attn5=Attn(in_channels=channels[2])
    self.tconv3 = Up(channels[2] + channels[2], channels[1], 3, bias=False)#, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(dim, num_channels=channels[1])
    if use_attn: self.attn6=Attn(in_channels=channels[1])
    self.tconv2 = Up(channels[1] + channels[1], channels[0], 3, bias=False)#, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(dim, num_channels=channels[0])
    if use_attn: self.attn7=Attn(in_channels=channels[0])
    self.tconv1 = Up(channels[0] + channels[0], c_out, 3,scale_factor=1)
    
    # The swish activation function
    self.aact = nn.SiLU()
    self.marginal_prob_std = marginal_prob_std
    
  def forward(self,h1,h2,h3,h4,embed,mcm,t):
    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.aact(h)
    if self.use_attn: h=self.attn5(h,mcm)
    h = self.tconv3(padcat(h, h3))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.aact(h)
    if self.use_attn: h=self.attn6(h,mcm)
    h = self.tconv2(padcat(h, h2))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.aact(h)
    if self.use_attn: h=self.attn7(h,mcm)
    h = self.tconv1(padcat(h, h1))

    # Normalize output
    h = h / self.marginal_prob_std(t,h)[:, None, None, None] if self.use_t else h
    return h


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, c_in, c_out, dim=32, embed_dim=256, use_t=True, dropout=0,
              use_attn=False, use_self_attn=False, n_heads=8, d_head=16,context_dim=256, mapper=None, 
              use_ac=False, mid=128, use_ldm=False, **kwargs):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    Down=ft.partial(nn.Conv2d, bias=False, padding=1)
    channels=[dim, dim*2, dim*4, dim*8]
    # Gaussian random feature embedding layer for time
    if use_t: self.embed = nn.Sequential(
        GaussianFourierProjection(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim))
    # embed_dim=embed_dim*2 # for concat
    self.use_t=use_t
    self.use_attn=use_attn
    if use_attn: Attn=ft.partial(SamplerCrossAttention, use_self_attn=use_self_attn,n_heads=n_heads,
              d_head=d_head, context_dim=context_dim, mapper=mapper,dropout=dropout, c_m=c_out)
    # Encoding layers where the resolution decreases
    self.conv1 = Down(c_in, channels[0], 3, stride=1)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(dim if use_ldm else 4, num_channels=channels[0])
    if use_attn: self.attn1=Attn(in_channels=channels[0])
    self.conv2 = Down(channels[0], channels[1], 3, stride=2)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(dim, num_channels=channels[1])
    if use_attn: self.attn2=Attn(in_channels=channels[1])
    self.conv3 = Down(channels[1], channels[2], 3, stride=2)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(dim, num_channels=channels[2])
    if use_attn: self.attn3=Attn(in_channels=channels[2])
    self.conv4 = Down(channels[2], channels[3], 3, stride=2)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(dim, num_channels=channels[3])   
    if use_attn: self.attn4=Attn(in_channels=channels[3]) 

    self.use_ac=use_ac
    self.actor_head=ActorHead(c_out,marginal_prob_std,use_t,dim,embed_dim,use_attn,use_self_attn,n_heads,mapper,dropout,d_head,context_dim)
    if use_ac: 
      self.critic_head=CriticHead(c_out,mid,dim,embed_dim,use_attn,use_self_attn,n_heads,mapper,dropout,d_head,context_dim)
      self.actor=ft.partial(self.forward,critic=False)
      self.critic=ft.partial(self.forward,critic=True)

    # The swish activation function
    self.act = nn.SiLU()
  
  def forward(self, x, embed, t=None, mcm=None, critic=False): 
    # Obtain the Gaussian random feature embedding for t   
    if self.use_t: 
      embed = embed+self.act(self.embed(t)) 
      # embed = torch.cat([self.act(self.embed(t)),embed],-1)
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    if self.use_attn: h1=self.attn1(h1,mcm)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    if self.use_attn: h2=self.attn2(h2,mcm)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    if self.use_attn: h3=self.attn3(h3,mcm)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    if self.use_attn: h4=self.attn4(h4,mcm)

    if critic:
      assert self.use_ac==True
      return self.critic_head(h1,h2,h3,h4,embed,mcm)
    else: return self.actor_head(h1,h2,h3,h4,embed,mcm,t)


# device="cuda"

def marginal_prob_std(t, x, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t).to(x.device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
  

def diffusion_coeff(t, x, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t).to(x.device)


if __name__=='__main__':
  sigma =  25.0#@param {'type':'number'}
  marginal_prob_std_fn = ft.partial(marginal_prob_std, sigma=sigma)

  c_in=4
  c_out=1
  context_dim=64
  d_head=8
  n_head=8
  use_ac=False
  mid=64

  s=ScoreNet(marginal_prob_std_fn,c_in=c_in,c_out=c_out, context_dim=context_dim,
          use_attn=True,use_self_attn=True,d_head=d_head,n_heads=n_head,
          use_ac=use_ac,mid=mid).cuda()
  total = sum([param.nelement() for param in s.parameters()])
  print("Number of parameter: %.2fM" % (total/1e6))
  size=28
  b=8
  x=torch.rand(b,c_in,size,size).cuda()
  eps=1e-5

  c_m=1
  N=8
  h=w=size
  X=x.unsqueeze(1).repeat(1,N,1,1,1).reshape(-1,c_in,h,w) # each batch copy x
  m=torch.rand(b,N,c_m,h,w).cuda()
  mcs=m.reshape(-1,1,N,c_m,h,w).repeat(1,N,1,1,1,1) # for each, we get its competetors representation, or only within the class
  mask=1-torch.eye(N).unsqueeze(0).unsqueeze(3).repeat(1,1,1,c_m*h*w).to(x.device)
  mcm=(mask*mcs.reshape(-1,N,N,c_m*h*w)).reshape(b*N,N,c_m,h,w) # competetors
        
  x=X
  context=torch.rand(x.shape[0],256).to(x)
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std_fn(random_t,x)
  perturbed_x = x + z * std[:, None, None, None]
  dx = s(perturbed_x, context, t=random_t,mcm=mcm, critic=use_ac)

  print(dx.shape)


