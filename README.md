# OpenMP_Gravity
OpenMP project on NVIDIA GPU, testing neural network inference for the Gravity compiler. Dataset and Neural Network parameters are stored in the **data** folder.
## Usage
Modify the `-gpu` flag in **make.def** to match the target device's compute capability. Check NVIDIA documentation for more information. Then compile:
```
make test
./activate_function N
```
Where `N` is the batch size.
