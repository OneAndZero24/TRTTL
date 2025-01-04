FROM  nvcr.io/nvidia/tensorrt:23.05-py3@sha256:7c7a603c9b958c9feaed887fee4bded7eb9963494c13dd9ad51eb2388205216c
WORKDIR /trttl
COPY . /trttl

RUN mkdir /doxygen
WORKDIR /doxygen
RUN wget https://github.com/doxygen/doxygen/releases/download/Release_1_13_1/doxygen-1.13.1.linux.bin.tar.gz
RUN tar -xf doxygen-1.13.1.linux.bin.tar.gz

RUN apt-get update
RUN apt-get install -y graphviz

WORKDIR /trttl
RUN mkdir build
WORKDIR /trttl/build
RUN cmake ..
RUN make