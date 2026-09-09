# Bibliography

The scientific references behind walnutbutter's learning rule, in the order
they bear on the code. Each entry says what it contributes here.

## The estimator

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms
   for connectionist reinforcement learning. *Machine Learning*, 8(3-4),
   229-256. https://doi.org/10.1007/BF00992696
   The REINFORCE family: a scalar reward, a baseline subtracted to give an
   advantage, and a weight update proportional to advantage x (how the
   unit's random exploration deviated). `reinforce()` is this rule.

2. Fiete, I. R., & Seung, H. S. (2006). Gradient learning in spiking neural
   networks by dynamic perturbation of conductances. *Physical Review
   Letters*, 97(4), 048104. https://doi.org/10.1103/PhysRevLett.97.048104
   Node perturbation for spiking neurons: perturb each neuron's input,
   correlate the perturbation with the global reward, and you have an
   unbiased estimate of the reward gradient without any backward pass. The
   `--late ignore` rule is this estimator proper; the exploration noise
   added to potentials each epoch is the perturbation.

## Local eligibility, global signal: three-factor rules

3. Izhikevich, E. M. (2007). Solving the distal reward problem through
   linkage of STDP and dopamine signaling. *Cerebral Cortex*, 17(10),
   2443-2452. https://doi.org/10.1093/cercor/bhl152
   Each synapse keeps a decaying eligibility trace written by pre/post spike
   timing; a later dopamine signal turns the trace into a weight change. The
   biological answer to "how does a synapse know it was on the trace that
   succeeded": it doesn't, the coincidence stands in for causality.

4. Legenstein, R., Pecevski, D., & Maass, W. (2008). A learning theory for
   reward-modulated spike-timing-dependent plasticity with application to
   biofeedback. *PLoS Computational Biology*, 4(10), e1000180.
   https://doi.org/10.1371/journal.pcbi.1000180
   What reward-modulated STDP can and cannot learn, and why the reward must
   correlate with the local eligibility for learning to happen. Relevant to
   the decoded critic's flat reward, and to the sign of the `--late depress`
   rule.

5. Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
   plasticity, and theory of three-factor learning rules. *Frontiers in
   Neural Circuits*, 9, 85. https://doi.org/10.3389/fncir.2015.00085
   The review that names the family: presynaptic activity x postsynaptic
   activity (or perturbation) x a global third factor. Places node
   perturbation and reward-modulated Hebbian/STDP rules side by side, which
   is the choice `--late` exposes.

## Not yet mentioned in our conversations, but the roots of the above

- Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation
  of synaptic efficacy by coincidence of postsynaptic APs and EPSPs.
  *Science*, 275(5297), 213-215. The first report that the order of pre and
  post spikes sets the sign of the change.
- Bi, G.-Q., & Poo, M.-M. (1998). Synaptic modifications in cultured
  hippocampal neurons: dependence on spike timing, synaptic strength, and
  postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
  The STDP window itself: potentiation for pre-before-post, depression for
  post-before-pre, over tens of milliseconds.
- Werfel, J., Xie, X., & Seung, H. S. (2003). Learning curves for stochastic
  gradient descent in linear feedforward networks. *NIPS 16*. Compares
  weight perturbation, node perturbation and backpropagation: why node
  perturbation's variance grows with the number of neurons, which is what
  the 14-column runs felt.
