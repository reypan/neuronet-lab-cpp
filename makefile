CC = g++
SOURCES = neuro.cpp
EXECUTABLE = neuro

all: debug

debug:
	$(CC) $(SOURCES) -g -o $(EXECUTABLE)

release:
	$(CC) $(SOURCES) -o $(EXECUTABLE)
