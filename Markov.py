# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:10:40 2022

@author: lidon
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
import time
import scipy

key=jax.random.PRNGKey(42)

class Sampler:
  # p is the pdf of the target distribution, expected to be up to a normalizing constant
  # q is the pdf of the proposal distribution, expected to be up to a normalizing constant
  # q should be of the form q(x,y), which will produce q(y|x)
  # q_sample is a function to produce a sample generated from proposal q, input shall be q_sample(x)
  # q and q_sample are set to be None by default, because user may choose using Langevin kernel or some
  # other non-reversible kernels
  # When defining q_sample, if you want to run multiple chains at the same time, must be of the form q_sample(x,key)!
  def __init__(self,p,q=None,q_sample=None):
    self.p=p
    self.q=q
    self.q_sample=q_sample
    self.key=jax.random.PRNGKey(42)
    #self.log_p=jnp.log(self.p)

  # vectorize pdf evaluation
  # evaluate the distribution functions in a paralllel manner
  # these functions are called while you want to run multiple chains

  # evaluate p(x), where x is a nxd matrix now
  # p(x) is evaluated line by line
  def para_p(self,x):
    return jax.vmap(self.p)(x)
  
  def para_q(self,x,y):
    return jax.vmap(self.q)(x,y)

  # vectorize sampling from proposal
  # generate proposal when running multiple chains
  def para_q_sample(self,x):
    keys=jax.random.split(key,x.shape[0])
    return jax.vmap(self.q_sample)(x,keys)



  # Metropolis step (Metropolis kernel)
  # x <- current value 
  # y <- new value proposed by q(y|x) or q(x,y) written in another way
  def metropolis_step(self,x,y):
    alpha=min(0,jnp.log(self.p(y))+jnp.log(self.q(y,x))-jnp.log(self.p(x))-jnp.log(self.q(x,y)))
    u=jnp.log(np.random.uniform(0,1,1)[0])
    
    if u<= alpha:
      return y
    else:
      return x

  # Metropolis kernel
  def metropolis_kernel(self,x):
    y=self.q_sample(x)
    y=self.metropolis_step(x,y)
    return y
  
  # vectorize metropolis step
  # Now x is a matrix of size nxd
  # apply metropolis step/kernel to each row
  def para_metropolis_step(self,x,y):
    alpha=jnp.log(self.para_p(y))+jnp.log(self.para_q(y,x))-jnp.log(self.para_p(x))-jnp.log(self.para_q(x,y))
    alpha=jnp.where(alpha>0,alpha,0)
    u=jnp.log(np.random.uniform(0,1,len(alpha)))


    index=jnp.where(u<=alpha)[0]
    
    x=x.at[index].set(y)

    return x
  
  # define vectorized MH kernel
  def para_metropolis_kernel(self,x):
    y=self.para_q_sample(x)
    y=self.para_metropolis_step(x,y)

    return y
    


  # acquire the log likelihood, which will be used in langevin kernel
  def log_p(self,x):
    return jnp.log(self.p(x))

  # Non-reversible sampling based on langevin diffusion
  # Langevin kernel, used to differ from M-H kernel
  def langevin_kernel(self,x):
    # step size
    eps=0.1
    grad_log_p=jax.grad(self.log_p)
    y=x+eps*grad_log_p(x)+jnp.sqrt(2*eps)*np.random.normal(0.,1.,len(x))
    return y



  # draw samples from p by MCMC
  # chain_num: number of chains running simultaneously (will be added later)
  # pos: starting position
  # kernel: specify which kernel you want to use 
  # Currently Metropolis kernel + langevin kernel (non reversible MC) are available
  # samples: total samples to generate from each chain, should be an integer
  def sampling(self,pos,kernel,samples):
    pos=jnp.array(pos)
    output=[]
    output.append(kernel(pos))
    for i in range(1,samples):
      # test runtime
      # start time
      start=time.time()
      x=output[i-1]
      output.append(kernel(x))
      # end time
      end=time.time()
      print(end-start)
    output=jnp.array(output)
    return output

