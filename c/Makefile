CXX = g++
CC = gcc
AR = ar

C_CXX_FLAGS := -pedantic -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas
C_CXX_DEBUG_FLAGS := $(C_CXX_FLAGS)

C_CXX_FLAGS += -O3 -march=native
C_CXX_DEBUG_FLAGS += -g -O0

C_FLAGS := -std=c99

CXX_FLAGS =

HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../robin-map/include/

# pass additional cflag options via make OPTS=


.PHONY: build
build:
	$(CXX) $(OPTS) $(C_CXX_FLAGS) $(CXX_FLAGS) -o lib.o -c lib.cpp $(HEADER_INCLUDES) -shared -fPIC
	$(CXX) $(OPTS) $(C_CXX_FLAGS) $(CXX_FLAGS) -o libusearch.so lib.o $(HEADER_INCLUDES) -shared -fPIC
	$(AR) -rcv libusearch.a lib.o

debug:
	$(CXX) $(OPTS) $(C_CXX_DEBUG_FLAGS) $(CXX_FLAGS) -o libusearch.so -c lib.cpp $(HEADER_INCLUDES) -shared -fPIC

.PHONY: static
static:
	$(CXX) $(OPTS) $(C_CXX_FLAGS) $(CXX_FLAGS) -o libusearch.a lib.cpp $(HEADER_INCLUDES) -shared -static -fPIC

.PHONY: test
test:
	$(CC) $(OPTS) -D_GNU_SOURCE $(C_CXX_DEBUG_FLAGS) $(CX_FLAGS) test.c external_retreival_testutils.c -L. -lusearch -Wl,-rpath,. -o test

.PHONY: clean
clean:
	rm -f *.o *.so *.a test
