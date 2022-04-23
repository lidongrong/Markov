**2022.1.16** Start project

**2022.4.18** Support running parallel multiple chains.

Can consider: by using high-order functions, avoid defining the key explicitly in q_sample

**2022.4.22** Support running langevin algorithm (non-reversible MCMC) in parallel

Next: add different proposal functions, provide diagnostics of the chain, explore how to use jit() to accelerate the chain.

**2022.4.23** Perhaps I shall implement this package via numpy, which is more convenient for parallelization & easy to deploy
