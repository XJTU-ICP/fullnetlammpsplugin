## Use cmake

Please set the following environment variables using -D option of cmake:

 - Torch_DIR: path to torch(with subpath `/share/cmake/Torch`)
 - LAMMPS_BINARY_ROOT: path to compiled LAMMPS and its library(with subpath `/lib/cmake/LAMMPS`)
 - LAMMPS_SOURCE_DIR: path to LAMMPS source code(with subpath `/cmake/Modules/LAMMPSUtils.cmake`) 

And if you use CUDA version of torch, make sure your environment has CUDA environment variables.

You can also set cmake option "-DCMAKE_INSTALL_PREFIX=<path>" to set install target.

## make and install

One can make the plugin like:
```bash 
cmake .. -C /root/lammps/cmake/presets/oneapi.cmake -DTorch_DIR=/root/libtorch/libtorch -DLAMMPS_BINARY_ROOT=/root/lammps/binary -DLAMMPS_SOURCE_DIR=/root/lammps -DCMAKE_INSTALL_PREFIX=../binary
```

and install as:
```
make install
```

## run lammps settings

You can run lammps using plugin but not setting environment variables as follows:
```bash
 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<...>/libtorch/lib:<...>/lammpsPluginTest/binary/lib/ LAMMPS_PLUGIN_PATH=<...>/binary/lib/torchmolnet_lmp lmp_mpi -in in.lammps > log.lammps
```

- LD_LIBRARY_PATH needs /libtorch/lib of path of `libtorch.so`.
- LD_LIBRARY_PATH needs /lammpsPluginTest/binary/lib of path of `libtorchmolnet_*.so`.
- LAMMPS_PLUGIN_PATH needs the path of `*plugin.so`
