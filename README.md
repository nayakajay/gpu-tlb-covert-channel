# (Mis)managed: A Novel TLB-based Covert Channel on GPUs
This repository gives the implementation of proof-of-concept code used to evaluate a TLB-based covert channel on NVIDIA GPUs.
We also provide the files used to reverse-engineer the TLB hierarchy of the GPUs.
We evaluated this work on an NVIDIA GTX 1080Ti

This work appeared in the proceedings of the 16th ACM Asia Conference on Computer and Communications Security. For more details, please refer https://doi.org/10.1145/3433210.3453077

### trojan.cu
The file used to send (Trojan) content. It provides a multi-threadblock implementation for the sender.

### spy.cu
The file used to receive (SPY) content. It provides a multi-threadblock implementation for the receiver.

### sync_utils.h
This file provides configurations and builds access patterns which is used by ```spy.cu``` and ```trojan.cu``` for building the covert channel.

Usage:
```
nvcc -Xptxas -dscm=wt -Xptxas -dlcm=cg -arch=sm_61 spy.cu -o spy
nvcc -Xptxas -dscm=wt -Xptxas -dlcm=cg -arch=sm_61 trojan.cu -o trojan
# Enable MPS
nvidia-cuda-mps-control -d
./spy > spy.txt &
./trojan > trojan.txt
# Disable MPS
echo quit | nvidia-cuda-mps-control
```

### aggregate.cu
This file enables the reverse-engineering of GPU TLBs using Unified Memory feature in CUDA. It traverses a range virtual address regiob at a stride. Both provided as an argument.

Usage:
```
nvcc -Xptxas -dscm=wt -Xptxas -dlcm=cg -arch=sm_61 aggregate.cu -o agg
./agg start_size_mb end_size_mb stride_size_kb
```

### tlb.cu
This file performs similar to ```aggregate.cu```, but provides fine-grained visibility of each access performed in the strided access pattern. This file was used in reverse-engineering the hash-function used for indexing into the TLB.

Usage:
```
nvcc -Xptxas -dscm=wt -Xptxas -dlcm=cg -arch=sm_61 tlb.cu -o tlb
./tlb start_size_mb end_size_mb stride_size_kb
```
