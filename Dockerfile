FROM docker.km3net.de/base/centos7-bundle:2021.03.01

RUN  yum install -y -e 0 devtoolset-10

RUN  source /opt/rh/devtoolset-10/enable && \
     cd /opt && \
     wget http://www.hepforge.org/archive/roottuple/RootTuple-1.0.0.tar.gz && \
     tar -xzvf RootTuple-1.0.0.tar.gz && \
     cd RootTuple-1.0.0 && \
     sed -i 's/SHARED/STATIC/g' ./src/CMakeLists.txt && \
     sed -i '$d' ./CMakeLists.txt && \
     mkdir build; cd build && \
     cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
     make; make install

RUN  source /opt/rh/devtoolset-10/enable && \
    cd /opt && \
    wget https://gibuu.hepforge.org/downloads?f=buuinput2021.tar.gz && \
    tar xvzf downloads?f=buuinput2021.tar.gz && \
    wget https://gibuu.hepforge.org/downloads?f=release2021.tar.gz && \
    tar xvzf downloads?f=release2021.tar.gz && \
    ls -ahl && \
    cd release2021 && \
    sed -i '/type(particle)/s/dimension(10)/dimension(30)/' code/collisions/oneBodyReactions/AddDecay.f90 && \
    cp /opt/RootTuple-1.0.0/build/src/libRootTuple.a ./objects/LIB/lib/libRootTuple.100.a && \
    make -j withROOT=1 && \
    rm -rf /opt/*.tar.gz

ADD . /km3buu

RUN cd /km3buu && \
    pip3 install setuptools-scm && \
    pip3 install pytest-runner && \
    pip3 install -e .

RUN cd /km3buu/externals/km3net-dataformat/ && \
    make
ENV KM3NET_LIB=/km3buu/externals/km3net-dataformat/lib    
ENV CONTAINER_GIBUU_EXEC=/opt/release2021/objects/GiBUU.x 
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
