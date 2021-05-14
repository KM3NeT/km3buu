#!/usr/bin/python3

from km3buu.jobcard import generate_neutrino_jobcard
from km3buu.jobcard import write_jobcard
from km3buu.jobcard import XSECTIONMODE_LOOKUP, PROCESS_LOOKUP, FLAVOR_LOOKUP

events = 1000
energy_min = 0.1
energy_max = 50
interaction = "cc"
flavor = "electron"
target_z = 8
target_a = 16
fluxfile = None
jc = generate_neutrino_jobcard(events, interaction, flavor, (energy_min,energy_max), (target_z,target_a),fluxfile=fluxfile)
write_jobcard(jc,"./job.job")
