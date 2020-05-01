g++ reader.cpp -I./libs/eigen/Eigen reader.h -c -std=c++11 && ^
g++ solver.cpp -I./libs/eigen/Eigen -c -g -std=c++11 && ^
g++ reader.o solver.o -o main.exe
