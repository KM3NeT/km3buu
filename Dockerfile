FROM rootproject/root:6.26.02-ubuntu22.04

RUN  apt-get -qq update && \
     apt-get -qq install python3.10 python3.10-distutils python3-pip libbz2-dev git 

RUN   cd /opt && \
      mkdir GiBUU && cd GiBUU && \
      wget --content-disposition https://gibuu.hepforge.org/downloads?f=release2021.tar.gz && \
      tar -xzvf release2021.tar.gz && \
      wget --content-disposition https://gibuu.hepforge.org/downloads?f=buuinput2021.tar.gz && \
      tar -xzvf buuinput2021.tar.gz && \
      wget --content-disposition https://gibuu.hepforge.org/downloads?f=libraries2021_RootTuple.tar.gz && \
      tar -xzvf libraries2021_RootTuple.tar.gz && \
      rm -rf ./*.tar.gz && \ 
      sed -i '6 a set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)' ./libraries2021/RootTuple/RootTuple-master/CMakeLists.txt && \ 
      cd release2021 && make -j buildRootTuple_POS && \
      make -j FORT=gfortran MODE=lto ARGS="-march=native" withROOT=1

ADD . /km3buu

RUN cd /km3buu && \
    pip3 install --upgrade pip && \
    pip3 install setuptools-scm && \
    pip3 install pytest-runner && \
    pip3 install conan && \
    pip3 install -e . && \
    pip3 install -e ".[dev]" && \
    pip3 install -e ".[extras]"

RUN cd /km3buu/externals/km3net-dataformat/ && \
    make
ENV KM3NET_LIB=/km3buu/externals/km3net-dataformat/lib    
ENV CONTAINER_GIBUU_EXEC=/opt/GiBUU/release2021/objects/GiBUU.x 
ENV CONTAINER_GIBUU_INPUT=/opt/GiBUU/buuinput 
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
