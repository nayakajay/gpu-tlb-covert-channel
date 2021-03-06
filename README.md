# (Mis)managed: A Novel TLB-based Covert Channel on GPUs
This repository gives the implementation of proof-of-concept code used to evaluate a TLB-based covert channel on NVIDIA GPUs.
We also provide the files used to reverse-engineer the TLB hierarchy of the GPUs.
We evaluated this work on an NVIDIA GTX 1080Ti.

This work appeared in the proceedings of the 16th ACM Asia Conference on Computer and Communications Security. For more details, please refer https://doi.org/10.1145/3433210.3453077

### trojan.cu
The file used to send (Trojan) content. It provides a multi-threadblock implementation for the sender.

### spy.cu
The file used to receive (SPY) content. It provides a multi-threadblock implementation for the receiver.

### sync_utils.h
This file provides configurations (empirically decided) and builds access patterns (using indexing function of TLBs) which are used by ```spy.cu``` and ```trojan.cu``` for using the TLB-based covert channel.

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
Alternatively, can use the ```runner``` script present in the root directory of the repository.

### outputs directory
This directory lists example outputs we observed on using the covert channel

### scripts directory
This directory contains some utility scripts to decode the messages transmitted by the trojan.
For example, ```python scripts/decode.py spy.txt``` will decode the message to binary form.

### aggregate.cu
This file was used to reverse-engineer the GPU's TLB hierarchy using Unified Memory feature in CUDA. It traverses a range of virtual address region at a stride. All parameters are provided as arguments.

Usage:
```
nvcc -Xptxas -dscm=wt -Xptxas -dlcm=cg -arch=sm_61 aggregate.cu -o agg
./agg start_size_mb end_size_mb stride_size_kb
```

### tlb.cu
This file performs accesses similar to ```aggregate.cu```, but provides fine-grained visibility of each access in the strided access pattern. This file was used to reverse-engineer the hash-function used for indexing into the TLB.

Usage:
```
nvcc -Xptxas -dscm=wt -Xptxas -dlcm=cg -arch=sm_61 tlb.cu -o tlb
./tlb start_size_mb end_size_mb stride_size_kb
```
