# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:10:40 2022

@author: lidon
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats

#key=jax.random.PRNGKey(42)

class Sampler:
  # p is the pdf of the target distribution, expected to be up to a normalizing constant
  # q is the pdf of the proposal distribution, expected to be up to a normalizing constant
  # q should be of the form q(x,y), which will produce q(y|x)
  # q_sample is a function to produce a sample generated from proposal q, input shall be q_sample(x)
  # q_sample are set to be None by default, because user may choose using Langevin kernel or some
  # other non-reversible kernels
  def __init__(self,p,q=None,q_sample=None):
    self.p=p
    self.q=q
    self.q_sample=q_sample
  
  # Metropolis step (Metropolis kernel)
  # x <- current value 
  # y <- new value proposed by q(y|x) or q(x,y) written in another way
  def metropolis(self,x,y):
    alpha=min(0,jnp.log(self.p(y))+jnp.log(self.q(y,x))-jnp.log(self.p(x))-jnp.log(self.q(x,y)))
    u=jnp.log(np.random.uniform(0,1,1)[0])
    
    if u<= alpha:
      return y
    else:
      return x

  def metropolis(self,x):
    y=self.q_sample(x)
    y=self.metropolis(x,y)
    return y

  def langevin(self,x):
    eps=0.05
    def log_p(x):
      return jnp.log(self.p(x))
    grad_log_p=jax.grad(log_p)
    y=x+eps*grad_log_p(x)+jnp.sqrt(2*eps)*np.random.normal(0.,1.,len(x))
    return y



  # draw samples from p by Metropolis step
  # chain_num: number of chains running simultaneously
  # pos: starting position
  # kernel: specify which kernel you want to use 
  # Currently Metropolis kernel + langevin kernel (non reversible MC) are available
  # samples: total samples to generate from each chain, should be an integer
  def sampling(self,pos,kernel,samples):
    output=[]
    output.append(kernel(pos))
    for i in range(1,samples):
      x=output[i-1]
      output.append(kernel(x))
    output=jnp.array(output)
    return output


