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
CUDA_HOST_SOURCES=Mode.cpp precomp.cpp SpeedSample.cpp
CUDA_DEVICE_SOURCES=profanity_cuda.cu cuda_kernels.cu
CUDA_OBJECTS_HOST=$(CUDA_HOST_SOURCES:.cpp=.o)
CUDA_OBJECTS_DEVICE=$(CUDA_DEVICE_SOURCES:.cu=.o)

$(EXECUTABLE_CUDA): $(CUDA_OBJECTS_HOST) $(CUDA_OBJECTS_DEVICE)
	$(NVCC) -O3 -std=c++14 -o $@ $(CUDA_OBJECTS_HOST) $(CUDA_OBJECTS_DEVICE)

%.o: %.cpp
	$(NVCC) -O3 -std=c++14 -c $< -o $@

%.o: %.cu
	$(NVCC) -O3 -std=c++14 -c $< -o $@
