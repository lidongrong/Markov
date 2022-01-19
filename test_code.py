# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:12:56 2022

@author: lidon
"""

# test for normal distribution
def p(x):
  return 0.5*jnp.exp(-0.5*jnp.dot(x,x))+0.5*jnp.exp(-0.5*jnp.dot(x-jnp.array([3.,3.]),x-jnp.array([3.,3.])))


def q(x,y):
  return jnp.exp(-0.5*jnp.dot(y-x,y-x))

def q_sample(x):
  I=np.eye(len(x))
  y=stats.multivariate_normal.rvs(x,np.eye(len(x)),1)
  return y

start_point=np.array([-1.,-1.])

model=Sampler(p,q,q_sample)

s=model.sampling(start_point,model.langevin,12000)

import matplotlib.pyplot as plt

plt.hist(s[3000:,0],bins=50)