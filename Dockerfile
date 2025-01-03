FROM  nvcr.io/nvidia/tensorrt:24.09-py3@sha256:832dd68095a66693aafe6d854c9c8a8e7046965ea4df40ecfa571ee0fefe59e1
WORKDIR /trttl
COPY . /trttl
RUN mkdir build
RUN cd build
RUN cmake .
RUN make