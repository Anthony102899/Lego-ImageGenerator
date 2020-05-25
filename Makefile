CXX=g++
PY=python3
INC=-I./libs/eigen/Eigen -I$(INCDIR) -I${GUROBI_HOME}/include -I./libs
CFLAGS=-std=c++14 -g -Wall -O2
LIBS=-L/opt/gurobi902/linux64/lib -lgurobi_c++ -lgurobi90

DATADIR=data
INCDIR=./include
SRCDIR=src
OBJDIR=obj
IMGDIR=img
LOGDIR=log
PYDIR=script

_OBJ=reader.o solver.o shifter.o writer.o gurobi_solver.o coordinator.o
OBJ =$(patsubst %,$(OBJDIR)/%,$(_OBJ))

_HEADER=reader.h solver.h shifter.h writer.h gurobi_solver.h coordinator.h
HEADER = $(patsubst %,$(INCDIR)/%,$(_HEADER))

default: solver


$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADER)
	mkdir -p $(OBJDIR)
	$(CXX) -c -o $@ $< $(CFLAGS) $(INC)

solver: $(SRCDIR)/main.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(INC) $(LIBS)

gurobi_solver: $(SRCDIR)/gurobi_main.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(INC) $(LIBS)

$(DATADIR)/%.txt.out: $(DATADIR)/%.txt solver
	./solver $< 

%.png: $(DATADIR)/%.txt.out
	$(PY) ./script/draw_images.py $<
	mv $@ $(IMGDIR)

%.log: $(DATADIR)/%.txt gurobi_solver
	./gurobi_solver $< > $@

.PRECIOUS: $(DATADIR)/%.txt.out

.PHONY: clean archive all

clean:
	rm -rf obj solver