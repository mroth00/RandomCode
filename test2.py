import pymc as pm
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt


# notice the unequal sample sizes -- no problem in Bayesian analysis.
N_A = 3003
N_B = 2641

nA = 390
nB = 297

diff1=N_A-nA
diff2=N_B-nB

observations_A = np.asarray(nA*[1]+diff1*[0])
observations_B = np.asarray(nB*[1]+diff2*[0])

# Set up the pymc model. Again assume Uniform priors for p_A and p_B.
p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)


# Define the deterministic delta function. This is our unknown of interest.
@pm.deterministic
def delta(p_A=p_A, p_B=p_B):
    return p_A - p_B

# Set of observations, in this case we have two observation datasets.
obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)

# To be explained in chapter 3.
mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
mcmc.sample(25000, 1000)

p_A_samples = mcmc.trace("p_A")[:]
p_B_samples = mcmc.trace("p_B")[:]
delta_samples = mcmc.trace("delta")[:]


# Count the number of samples less than 0, i.e. the area under the curve
# before 0, represent the probability that site A is worse than site B.


print("\n===================================================")
print("In test A you had", N_A, "Visits with", nA, "conversions\n" )
print("In test B you had", N_B, "Visits with", nB, "conversions\n" )
print("Converstion rate A: ",nA/N_A,)
print("\nConverstion rate A: ",nB/N_B,)
print("Rate A - Rate B = ", nA/N_A - nB/N_B)
print("\n===================================================")

print("\n===================================================")
print( "\nProbability site A is WORSE than site B: %.3f" % \
    (delta_samples < 0).mean())

print("Probability site A is BETTER than site B: %.3f" % \
    (delta_samples > 0).mean())
print("\n===================================================")

print("\n===================================================")
print( "\nProbability site A is 1%% WORSE than site B: %.3f" % \
    (delta_samples < -.01).mean())

print("Probability site A is 1%% BETTER than site B: %.3f" % \
    (delta_samples > .01).mean())
print("\n===================================================")

print("\n===================================================")
print( "\nProbability site A is 2%% WORSE than site B: %.3f" % \
    (delta_samples < -.02).mean())

print("Probability site A is 2%% BETTER than site B: %.3f" % \
    (delta_samples > .02).mean())
print("\n===================================================")
# histogram of posteriors


plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(0, 0, 60, linestyle="--")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right")
plt.show()

