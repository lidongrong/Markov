# test for mixed normal distribution
def p(x):
  return 0.5*jnp.exp(-0.5*jnp.dot(x,x))+0.5*jnp.exp(-0.5*jnp.dot(x-jnp.array([5.,5.]),x-jnp.array([5.,5.])))


def q(x,y):
  return jnp.exp(-0.5*jnp.dot(y-x,y-x))

# For multiple chains, must specify the key to avoid generating same random numbers!
def q_sample(x,key=jax.random.PRNGKey(int(time.time()))):
  y=jax.random.normal(key,x.shape)
  y=x+y
  return y


# Run 20 Markov Chains in parallel
start_point=np.random.normal(0,1,10)
start_point=start_point.reshape((5,2))

start_point=jnp.array(start_point)

# Define the model
model=Sampler(p,q,q_sample)
#model.para_q_sample(start_point)
# run iterations, use the kernel specified
s=model.sampling(start_point,model.para_langevin_kernel,5000)
