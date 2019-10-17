Bootstrap: docker
From: debian:stretch

%post
    apt-get update -qq 
    apt-get install -qq -y neovim
    apt-get install -qq -y gfortran make libbz2-dev
    apt-get install -qq -y wget
    cd /opt && \
    wget https://gibuu.hepforge.org/downloads?f=buuinput2019.tar.gz && \
    tar xvzf downloads?f=buuinput2019.tar.gz && \
    wget https://gibuu.hepforge.org/downloads?f=release2019.tar.gz && \
    tar xvzf downloads?f=release2019.tar.gz && \
    ls -ahl && \
    cd release2019 && \
    make -j withROOT=0
    rm -rf /opt/*.tar.gz 

%environment
    export CONTAINER_GIBUU_EXEC=/opt/release2019/objects/GiBUU.x



