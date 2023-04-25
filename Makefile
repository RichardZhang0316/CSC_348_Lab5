all: MatrixMultNaive.cu MatrixMultTiled.cu MatrixMultAtomic.cu MatrixMultMutiStreaming.cu MatrixMultMutiStream+Atomic.cu
	nvcc -o MatrixMultNaive MatrixMultNaive.cu
	nvcc -o MatrixMultTiled MatrixMultTiled.cu
	nvcc -o MatrixMultAtomic MatrixMultAtomic.cu
	nvcc -o MatrixMultMutiStreaming MatrixMultMutiStreaming.cu
	nvcc -o MatrixMultMutiStream+Atomic MatrixMultMutiStream+Atomic.cu

clean:
	rm -f MatrixMultNaive MatrixMultTiled MatrixMultAtomic MatrixMultMutiStreaming MatrixMultMutiStream+Atomic
