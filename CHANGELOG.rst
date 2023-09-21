Unreleased changes
------------------

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
