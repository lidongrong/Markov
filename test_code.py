# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:12:56 2022

@author: lidon
"""
# test for normal distribution
def p(x):
  return 0.5*jnp.exp(-0.5*jnp.dot(x,x))+0.5*jnp.exp(-0.5*jnp.dot(x-jnp.array([5.,5.]),x-jnp.array([5.,5.])))


def q(x,y):
  return jnp.exp(-0.5*jnp.dot(y-x,y-x))

# For multiple chains, must specify the key to avoid generating same random numbers!
def q_sample(x,key):
  #np.random.seed()
  y=jax.random.normal(key,x.shape)

  y=x+y
  return y

# Start Sampling
start_point=np.random.normal(0,1,50)
start_point=start_point.reshape((25,2))

start_point=jnp.array(start_point)

model=Sampler(p,q,q_sample)
#model.para_q_sample(start_point)
s=model.sampling(start_point,model.para_metropolis_kernel,20)
