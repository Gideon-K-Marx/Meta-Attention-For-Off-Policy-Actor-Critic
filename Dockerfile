FROM openpai/standard:python_3.6-pytorch_1.4.0-gpu

RUN echo "export DISPLAY=:0"  >> /etc/profile

RUN export DEBIAN_FRONTEND=noninteractive; \
    export DEBCONF_NONINTERACTIVE_SEEN=true; \
    apt-get update \
    && apt-get install -y libglu1-mesa-dev \
                          freeglut3-dev \
                          mesa-common-dev \
                          libxmu-dev \
                          libxi-dev \
                          xvfb \
                          ffmpeg \
                          xorg-dev \
                          libsdl2-dev \
                          swig \
                          cmake \
                          zlib1g-dev \
                          python-opengl \
                          python3-tk \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

RUN $PIP_INSTALL gym==0.15.4
RUN $PIP_INSTALL gym[atari]==0.15.4
RUN $PIP_INSTALL gym[box2D]==0.15.4
RUN $PIP_INSTALL python-dateutil

# import
RUN wget http://www.atarimania.com/roms/Roms.rar \
    && unrar e Roms.rar \
    && unzip ROMS.zip \
    && python -m atari_py.import_roms ./ROMS \
    && rm -rf *ROMS* Roms.rar

# roboschool
RUN $PIP_INSTALL roboschool==1.0.48 \
    && apt-get install libgl1-mesa-dev
