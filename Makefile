CXX=g++
PY=python3
LIBS=-I./libs/eigen/Eigen -I$(INCDIR)
FLAGS=-std=c++11

DATADIR=data
INCDIR=./inc
SRCDIR=src
OBJDIR=obj
IMGDIR=img
PYDIR=script

_OBJ=reader.o solver.o main.o shifter.o writer.o
OBJ =$(patsubst %,$(OBJDIR)/%,$(_OBJ))

_HEADER=reader.h solver.h shifter.h writer.h
HEADER = $(patsubst %,$(INCDIR)/%,$(_HEADER))

default: solver

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADER)
	$(CXX) -c -o $@ $< $(FLAGS) $(LIBS)

solver: $(OBJ)
	$(CXX) -o $@ $^ $(FLAGS) $(LIBS)

$(DATADIR)/%.txt.out: $(DATADIR)/%.txt solver
	./solver $< 

%.png: $(DATADIR)/%.txt.out
	$(PY) ./script/draw_images.py $<
	mv $@ $(IMGDIR)

.PRECIOUS: $(DATADIR)/%.txt.out

.PHONY: clean

clean:
	rm -rf *.o solver