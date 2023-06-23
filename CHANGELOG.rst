Unreleased changes
------------------
* Limit the number of resamples in order to speed sampling

* Use unix time interval for distributing events
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
