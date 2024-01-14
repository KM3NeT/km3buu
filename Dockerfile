FROM rootproject/root:6.26.02-ubuntu22.04
MAINTAINER Johannes Schumann <jschumann@km3net.de>

RUN  apt-get -qq update && \
     apt-get -qq install python3.10 python3.10-distutils python3-pip libbz2-dev git 

RUN   cd /opt && \
      mkdir GiBUU && cd GiBUU && \
      wget --content-disposition https://gibuu.hepforge.org/downloads?f=archive/r2023_02/release2023.tar.gz&& \
      tar -xzvf release2023.tar.gz && \
      wget --content-disposition https://gibuu.hepforge.org/downloads?f=archive/r2023_02/buuinput2023.tar.gz && \
      tar -xzvf buuinput2023.tar.gz && \
      wget --content-disposition https://gibuu.hepforge.org/downloads?f=archive/r2023_02/libraries2023_RootTuple.tar.gz && \
      tar -xzvf libraries2023_RootTuple.tar.gz && \
      rm -rf ./*.tar.gz && \
      sed -i '6 a set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)' ./libraries2023/RootTuple/RootTuple-master/CMakeLists.txt && \
      cd release2023 && make -j buildRootTuple_POS && \
      make -j FORT=gfortran MODE=lto ARGS="-march=x86-64-v3" withROOT=1 

ADD . /km3buu

RUN cd /km3buu && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools && \
    pip3 install setuptools-scm && \
    pip3 install pytest-runner && \
    pip3 install conan && \
    pip3 install -e . && \
    pip3 install -e ".[dev]" && \
    pip3 install -e ".[extras]"

RUN init4buu --proposal=/proposal

RUN cd /km3buu/externals/km3net-dataformat/ && \
    make
ENV KM3NET_LIB=/km3buu/externals/km3net-dataformat/lib    
ENV CONTAINER_GIBUU_EXEC=/opt/GiBUU/release2023/objects/GiBUU.x 
ENV CONTAINER_GIBUU_INPUT=/opt/GiBUU/buuinput 
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV KM3BUU_CONFIG="/root/.km3buu"
