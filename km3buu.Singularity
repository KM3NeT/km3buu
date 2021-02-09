Bootstrap: docker
From: debian:stretch

%post
    apt-get update -qq
    apt-get install -qq -y neovim
    apt-get install -qq -y gfortran make libbz2-dev
    apt-get install -qq -y wget g++ git
    apt-get install -qq -y python3-dev python3-pip python3-tk python3-lxml python3-six

    cd /opt && \
    mkdir -p cmake-3.18 && wget -qO- "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C cmake-3.18

    cd /opt && \
    wget https://root.cern.ch/download/root_v6.20.04.source.tar.gz && \
    tar xvzf root_v6.20.04.source.tar.gz && \
    export PATH=/opt/cmake-3.18/bin:$PATH && \
    cd root-6.20.04 && \
    mkdir -p obj; cd obj && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -Dpyroot=OFF -Dpyroot_experimental=OFF -Dx11=OFF -Dxft=OFF ..  && \
    make -j4; make install

    cd /opt && \
    wget http://www.hepforge.org/archive/roottuple/RootTuple-1.0.0.tar.gz && \
    tar xvzf RootTuple-1.0.0.tar.gz && \
    cd RootTuple-1.0.0 && \
    sed -i 's/SHARED/STATIC/g' ./src/CMakeLists.txt && \
    sed -i '$d' ./CMakeLists.txt && \
    mkdir build; cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. &&\
    make; make install

    wget https://gibuu.hepforge.org/downloads?f=buuinput2021.tar.gz && \
    tar xvzf downloads?f=buuinput2021.tar.gz && \
    wget https://gibuu.hepforge.org/downloads?f=release2021.tar.gz && \
    tar xvzf downloads?f=release2021.tar.gz && \
    ls -ahl && \
    cd release2021 && \
    cp /opt/RootTuple-1.0.0/build/src/libRootTuple.a ./objects/LIB/lib/libRootTuple.100.a && \
    make -j withROOT=1 && \
    rm -rf /opt/*.tar.gz

%environment
    export CONTAINER_GIBUU_EXEC=/opt/release2021/objects/GiBUU.x && \
    export LD_LIBRARY_PATH=/usr/local/lib



