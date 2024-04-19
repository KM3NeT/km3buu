Unreleased changes
------------------
* Fix nan direction of nucleon momentum for hydrogen

v1.8.0
----------------------------
* Fix flux fit function
* Add fitted flux norm and GiBUU flux bin width to header

v1.7.1
----------------------------
* Enabled photon propagation in GiBUU in std. configuration
* Fix right handed probability calculation for NC

v1.7.0
----------------------------
* Add additinal info about target nuclei in mc_trks field and fix particle states
* Update to GiBUU2023patch3
* Add field for right handed helicity probability for outgoing lepton in GiBUUOutput

v1.6.1
----------------------------
* Update process id mapping (i.e. 2p2h QE and Delta)

v1.6.0
----------------------------
* Add tau decays to raw km3net-dataformat output
* Fixed visible energy for muons
* Refine free particle cuts (tolerance for restmass of nucleons)
* Increase particle vector for PYTHIA decays

v1.5.0
----------------------------
* Use static flux model (no interpolation as previously)

v1.4.2
----------------------------
* Remove the fix as it was applied now in the GiBUU source

v1.4.1
----------------------------
* Fix anti-nu problem in GiBUU2023-Patch2

v1.4.0
----------------------------
* Update to GiBUU2023-Patch2
* Possibility to skip km3net dataformat writeout in `km3buu` runner script
* Write out nucleus information to km3net dataformat track list
* Fix setuptools (python) dependency

v1.3.0
----------------------------
* Make proposal an optional dependency
* Add invariant target mass (W2) to calculated features in GiBUUOutput
* Compatibility for particle module v0.23.0
* Update xsec settings in default jobcard (e.g. transition width to DIS and Bosted Christy for single pi)

v1.2.0
----------------------------
* Update to GiBUU2023-Patch1

v1.1.6
----------------------------
* refine jobcard

v1.1.5
----------------------------
* Add target to KM3NeT dataformat file header

v1.1.4
----------------------------
* Disable smoothening for flux interpolation (of data points from GiBUU)

v1.1.3
----------------------------
* Improved curve for flux index

v1.1.2
----------------------------
* Change input flux convention (for input value gamma from E^gamma to E^-gamma)

v1.1.1
----------------------------
* Limit minimal number of ensembles to 100 in estimate helper function
* Fix setup files

v1.1.0
----------------------------
* GiBUU2023 update
* Fix spherical volume by adding the distibute_events method

v1.0.0
----------------------------
* Implementation status from rc2 applied
* New logo and updated details in readme

v1.0.0 - Release Candidate 2
----------------------------
* Limit the number of resamples in order to speed sampling
* Use unix time interval for distributing events and fix time fields
* Add functions to export crosssections for SWIM software directly

v1.0.0 - Release Candidate 1
----------------------------
* Dockerfile with full GiBUU2021r4 installation (including RootTuple writeout)
* GiBUU output parser object
* KM3NeT file write out
* Runnerscript for std. configuarion (for use with Docker)
* Fixed header and sec. lepton write out
* Neutrino jobcard generator with FSI timesteps and particle decay option
* GiBUU to KM3NeT w2 weight conversion
* Add decay option to the runner script
* Add muon propagation and upgrade tau propagation/decay being km3net geometry based
* Add singularity build and deploy option to KM3NeT FTP to CI
* Add free particle cuts to check if nucleons are bound to nuclear potential
* Resturcture application of the target density in order to care for different materials, i.e. water/rock
* Add function to estimate #ensembles & #runs by desired number of events
