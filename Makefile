CC=g++
NVCC?=nvcc
CDEFINES=
SOURCES=Dispatcher.cpp Mode.cpp precomp.cpp profanity.cpp SpeedSample.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=profanity2.x64
EXECUTABLE_CUDA=profanity2_cuda

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS=-framework OpenCL
	CFLAGS=-c -std=c++11 -Wall -mmmx -O2
else
	LDFLAGS=-s -lOpenCL -mcmodel=large
	CFLAGS=-c -std=c++11 -Wall -mmmx -O2 -mcmodel=large 
endif

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(CDEFINES) $< -o $@

clean:
	rm -rf *.o

# CUDA build (optional)
CUDA_SOURCES=profanity_cuda.cpp Mode.cpp precomp.cpp
CUDA_OBJECTS=$(CUDA_SOURCES:.cpp=.cuo)

$(EXECUTABLE_CUDA): $(CUDA_OBJECTS) cuda_kernels.cu
	$(NVCC) -O3 -std=c++14 -Xcompiler -fPIC -o $@ $(CUDA_OBJECTS) cuda_kernels.cu

%.cuo: %.cpp
	$(NVCC) -O3 -std=c++14 -c $< -o $@
