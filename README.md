## Use cmake

Please set the following environment variables using -D option of cmake:

 - Torch_DIR: path to torch(with subpath `/share/cmake/Torch`)
 - LAMMPS_BINARY_ROOT: path to compiled LAMMPS and its library(with subpath `/lib/cmake/LAMMPS`)
 - LAMMPS_SOURCE_DIR: path to LAMMPS source code(with subpath `/cmake/Modules/LAMMPSUtils.cmake`) 

And if you use CUDA version of torch, make sure your environment has CUDA environment variables.

You can also set cmake option "-DCMAKE_INSTALL_PREFIX=<path>" to set install target.

### Build Lammps

we need build lammps with shared libs. You can build lammps with cmake as follows:

```
cd lammps/build
cmake -C ../cmake/presets/most.cmake -C ../cmake/presets/oneapi.cmake -D CMAKE_INSTALL_PREFIX=../binary -D BUILD_SHARED_LIBS=yes ../cmake/
```

### Dependencies

- cuda nvcc
- complier
- mpi
- mkl

## make and install

One can make the plugin like:
```bash 
cmake .. -C /root/lammps/cmake/presets/oneapi.cmake -DTorch_DIR=/root/libtorch/libtorch -DLAMMPS_BINARY_ROOT=/root/lammps/binary -DLAMMPS_SOURCE_DIR=/root/lammps -DCMAKE_INSTALL_PREFIX=../binary
```

and install as:
```
make install -j <cpu_num>
```

## run lammps settings

You can run lammps using plugin but not setting environment variables as follows for test version before it is setting in your environment:
```bash
 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<...>/libtorch/lib:<...>/lammpsPluginTest/binary/lib/ LAMMPS_PLUGIN_PATH=<...>/binary/lib/torchmolnet_lmp lmp_mpi -in in.lammps > log
```

- LD_LIBRARY_PATH needs /libtorch/lib of path of `libtorch.so`.
- LD_LIBRARY_PATH needs /lammpsPluginTest/binary/lib of path of `libtorchmolnet_*.so`.
- LAMMPS_PLUGIN_PATH needs the path of `*plugin.so`

## run lammps

just run lammps as follows if you have set the environment variables `LAMMPS_PLUGIN_PATH` correctly:
```bash
lmp_mpi -in in.lammps > log
```

In which the in.lammps consist pair settings as follows:
```
pair_style	torchmolnet model_traced_torch_1.12.0.pt cuda 6.0
pair_coeff	* *
```
Where `spookynet_model_traced_torch_1.12.0.pt` is the model file by torchscript `torch.jit.trace`, `cuda` is the device type, `6.0` is the cutoff value for the model, and `* *` means all atoms are used for the model.(It is the default and only option.)

Please notice the pbc neighbor list by lammps is not used in the plugin. So the cutoff by lammps only used for lammps examination. The cutoff by model is set here and can be different within your model training, set on your own risk.

If you want to use cpu version, you can set `cpu` instead of `cuda` in pair_style. If you want debug mode, you can add `debug` after the cutoff value.
