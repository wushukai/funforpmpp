CC = nvcc
CFLAGS = -O3 -arch=sm_75
TARGET = main

all: $(TARGET)

$(TARGET): main.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
