rm -rf build
mkdir build
cd build

cmake ..
make -j


echo "----- SEQ -----"
./seq


echo "----- OMP -----"
export OMP_NUM_THREADS=1 && ./omp
export OMP_NUM_THREADS=4 && ./omp
export OMP_NUM_THREADS=8 && ./omp
export OMP_NUM_THREADS=12 && ./omp


echo "----- MPI -----"
export OMP_NUM_THREADS=1
mpirun -np 1 ./mpi
mpirun -np 2 ./mpi
mpirun -np 4 ./mpi


echo "----- MPI + OMP -----"
export OMP_NUM_THREADS=3 && mpirun -np 4 ./mpi
export OMP_NUM_THREADS=6 && mpirun -np 2 ./mpi
export OMP_NUM_THREADS=12 && mpirun -np 1 ./mpi