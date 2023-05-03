Theory
======
In the following section specific parts of `km3buu` are described which
exceed the functionaliy of GiBUU and are important for the usage.

Kinematic variables
-------------------
Some additional variables which characterise the interaction are
calculated inside `km3buu`.

.. list-table:: Kinematic Variables
   :widths: 50 50
   :header-rows: 1

   * - Variable
     - Content
   * - Bjorken scaling variable :math:`B_x`
     - :math:`Q^2 / 2p\cdot q` where :math:`p` is the momentum given by GiBUU
   * - Energy fraction passed to hadr. system / :math:`B_y`
     - :math:`p\cdot q / p\cdot k`
   * - Momentum passed to the hadronic system / :math:`Q^2`
     - :math:`(k_{\nu}-k_{lepton})^2`

Weights
-------
In order to retrieve correct results and provide correct KM3NeT weights (w2)
the treatment of the GiBUU weights is an important step. A brief description
of the GiBUU weights and how to calculate actual cross sections is given on the
`GiBUU Homepage <https://gibuu.hepforge.org/trac/wiki/perWeight>`__ and
a more detailed description of the calculation can be found in the `PhD Thesis
of Tina Leitner <https://inspirehep.net/literature/849921>`__ in Chapter 8.3.
As it is mentioned in the description of the output flux file in the
`documentation <https://gibuu.hepforge.org/Documentation/code/init/neutrino/initNeutrino_f90.html#robo1685>`__ this is not taken somehow into account inside the weights.
Following the description the GiBUU event weight can be converted to a binned
cross section via

.. math::
    \frac{d\sigma}{E} = \frac{\sum_{i\in I_\text{bin}} w_i}{\Delta E}\cdot\frac{1}{E\Phi},

where :math:`\Phi` is the simulated flux.
As the weights are given for each run individually the weight also has to be divided
by the number of runs.

Free Particle Cuts
------------------
The secondary (pertubative) particles are not filter with respect to their energy
and bound to the nuclear potential. This violates the energy conservation of the
system and may lead to unphysical result. Therefore a particle mask is
provided by the `GiBUUOutput` object in order to check the
energy-momentum relation of the contained nucleons, d.h. :math:`pdgid {\epsilon} [2112,2212]`),
for the rest mass. If this is :math:`m_p<m_0` the particle is rejected in the given mask.
