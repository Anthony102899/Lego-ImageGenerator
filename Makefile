CXX=g++
PY=python3
MAKE=make

DATADIR=data
IMGDIR=img
PYDIR=script
OUTPUTDIR=data/output
MODELDIR=data/model


_CPP_EXECUTABLES=solver gurobi_solver constraint_matrix_extractor
CPP_EXECUTABLES=$(patsubst %,cpp/%,$(_CPP_EXECUTABLES))

all: $(CPP_EXECUTABLES)

cpp/solver: 
	cd cpp && $(MAKE) solver

cpp/gurobi_solver:
	cd cpp && $(MAKE) gurobi_solver

cpp/constraint_matrix_extractor:
	cd cpp && $(MAKE) constraint_matrix_extractor 

$(OUTPUTDIR)/%.txt.out: $(MODELDIR)/%.txt cpp/solver
	./cpp/solver $< 

%.png: $(OUTPUTDIR)/%.txt.out
	mkdir -p $(OUTPUTDIR)
	$(PY) ./script/draw_images.py $<

%.log: $(MODELDIR)/%.txt cpp/gurobi_solver
	./cpp/gurobi_solver $< > $@

.PRECIOUS: $(OUTPUTDIR)/%.txt.out

clean:
	cd cpp && $(MAKE) clean

.PHONY:
	clean