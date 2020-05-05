CXX=g++
PY=python3
LIBS=-I./libs/eigen/Eigen
FLAGS=-std=c++11

HEADER=reader.h solver.h shifter.h writer.h

DATADIR=data
IMGDIR=img
OBJ=reader.o solver.o main.o shifter.o writer.o

default: solver

%.o: %.cpp $(HEADER)
	$(CXX) -c -o $@ $< $(FLAGS) $(LIBS)

solver: $(OBJ)
	$(CXX) -o $@ $^ $(FLAGS) $(LIBS)

$(DATADIR)/%.txt.out: $(DATADIR)/%.txt solver
	./solver $< 

%.png: $(DATADIR)/%.txt.out
	$(PY) draw_images.py $<
	mv $@ $(IMGDIR)

.PRECIOUS: $(DATADIR)/%.txt.out

clean:
	rm -rf *.o solver