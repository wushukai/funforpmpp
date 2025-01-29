CC = nvcc
CFLAGS = -O3
TARGET = main

all: $(TARGET)

$(TARGET): main.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
