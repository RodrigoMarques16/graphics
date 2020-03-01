NVCC_DBG       = -g -G
#NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_52,code=sm_52

all:
	nvcc main.cu -ccbin="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include\thrust"

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg