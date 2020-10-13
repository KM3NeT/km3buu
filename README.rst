KM3BUU
======

.. image:: https://git.km3net.de/simulation/km3buu/badges/master/pipeline.svg
    :target: https://git.km3net.de/simulation/km3buu/pipelines

.. image:: https://git.km3net.de/simulation/km3buu/badges/master/coverage.svg



The KM3BUU project is an integrated environment for the GiBUU studies
within the KM3NeT experiment.

Installation
------------

The project is based on images using ``singularity``, for which version
3 or higher (e.g. `v3.4 <https://sylabs.io/guides/3.4/user-guide/>`__)
is required. This is done due to the intention to provide a comparable
installation on all systems and thus make the results easily
reproducible. The main project control is based on ``make``. In order to
apply installation commands presented within this section, clone this
repository and change to the project directory:

::

   git clone https://git.km3net.de/simulation/km3buu
   cd km3buu

Local Machine
~~~~~~~~~~~~~

By “Local Machine” a computer where are root (administrative) privileges
are available is meant. These root privileges are required to build the
singularity image by yourself. In order to start the build, run the
following ``make`` command:

::

   make build

Compute Cluster
~~~~~~~~~~~~~~~

In order to make this project also usable in a non-root environment, the
Image is also provided via the KM3NeT Docker-Server. Within KM3NeT
computing infrastructure this is the case for the lyon compute cluster,
thus this case is customised for this environment.

In order to build the singularity image based on the remote image, run
the following ``make`` command:

::

   make buildremote

Structure & Usage
-----------------

The used GiBUU jobcards are located in a sub-folder within the jobcards
folder. Each sub-folder represents a set of jobcards, which can be
processed by:

::

   make run CARDSET=examples

This command runs all jobcards within the ``jobcards/examples`` folder
and writes the output it to the folder ``output``. The folder structure
is applied from the ``jobcards``\ folder.
