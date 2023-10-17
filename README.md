## Use cmake

Please set the following environment variables using -D option of cmake:

 - Torch_DIR: path to torch(with subpath `/share/cmake/Torch`)
 - LAMMPS_BINARY_ROOT: path to compiled LAMMPS and its library(with subpath `/lib/cmake/LAMMPS`)
 - LAMMPS_SOURCE_DIR: path to LAMMPS source code(with subpath `/cmake/Modules/LAMMPSUtils.cmake`) 

**All path should be absolute path.**

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
- mpi (optional)
- mkl (optional)

We recommend using conda or mamba to install the nvcc/cuda environment, you may need to install using `mamba install cuda=11.8 cuda-nvcc=11.8 cudatoolkit=11.8 -c nvidia`, and make sure using the same version for cudart with libtorch. And also for your pytorch version (11.8).

And compile lammps with:

(ref to https://github.com/deepmd-kit-recipes/lammps-feedstock/blob/master/recipe/build.sh)

```shell
ARGS="-D PKG_ASPHERE=ON -DPKG_BODY=ON -D PKG_CLASS2=ON -D PKG_COLLOID=ON -D PKG_COMPRESS=OFF -D PKG_CORESHELL=ON -D PKG_DIPOLE=ON -D PKG_EXTRA-COMPUTE=ON -D PKG_EXTRA-DUMP=ON -D PKG_EXTRA-FIX=ON -D PKG_EXTRA-MOLECULE=ON -D PKG_EXTRA-PAIR=ON -D PKG_GRANULAR=ON -D PKG_KSPACE=ON -D PKG_MANYBODY=ON -D PKG_MC=ON -D PKG_MEAM=ON -D PKG_MISC=ON -D PKG_MOLECULE=ON -D PKG_PERI=ON -D PKG_REPLICA=ON -D PKG_RIGID=ON -D PKG_SHOCK=ON -D PKG_SNAP=ON -D PKG_SRD=ON -D PKG_OPT=ON -D PKG_KIM=OFF -D PKG_GPU=OFF -D PKG_KOKKOS=OFF -D PKG_MPIIO=OFF -D PKG_MSCG=OFF -D PKG_LATTE=OFF -D PKG_PHONON=ON -D PKG_REAXFF=ON -D WITH_GZIP=ON -D PKG_COLVARS=ON -D PKG_PLUMED=yes -D PKG_FEP=ON -D PLUMED_MODE=runtime -D PKG_QTB=ON -D PKG_PLUGIN=ON -D PKG_H5MD=ON"
cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_LIB=ON -D BUILD_SHARED_LIBS=ON -D LAMMPS_INSTALL_RPATH=ON -DCMAKE_INSTALL_LIBDIR=lib $ARGS -D FFT=FFTW3 -D CMAKE_INSTALL_PREFIX=【install path】 ../cmake
make #-j${NUM_CPUS}
make install
```
Sometimes you may not always need fftw3, so you can delete -D FFT=FFTW3. Or install it following https://docs.lammps.org/Build_settings.html#fft-library.

To be briefly, we always need the BUILD_LIB=ON, PKG_PLUGIN=ON, see https://docs.lammps.org/Build_package.html for details from lammps.

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
