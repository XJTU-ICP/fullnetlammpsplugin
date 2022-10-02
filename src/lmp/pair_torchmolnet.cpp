#include <iostream>
#include <string.h>
#include <iomanip>
#include <limits>
#include <filesystem>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#include "output.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "modify.h"
#include "fix.h"
#include "citeme.h"

#include "pair_torchmolnet.h"

using namespace LAMMPS_NS;

static const char cite_torch_mol_net_package[] =
    "TorchMolNet package:\n\n"
    "@misc{Han_TorchMolNet,\n"
    "  author = {Han, Yanbo},\n"
    "  url = {https://github.com/saltball/lammpsPluginTest},\n"
    "  year = 2022,\n"
    "}\n\n";

PairTorchMolNet::PairTorchMolNet(LAMMPS *lmp)
    : Pair(lmp)
{
  if (lmp->citeme)
    lmp->citeme->add(cite_torch_mol_net_package);
  if (strcmp(update->unit_style, "metal") != 0)
  {
    error->all(FLERR, "Pair torchmolnet requires metal unit, please set it by \"units metal\", see https://docs.lammps.org/units.html for details.\n");
  }
  cutoff_ = 5.;
  print_summary();
}

void PairTorchMolNet::settings(int narg, char **arg)
{
  numb_computes_ = 0;
  if (std::filesystem::exists("debug"))
  {
    std::filesystem::remove_all("debug");
  }
  std::filesystem::create_directory("debug");

  if (narg > 2)
  {
    error->all(FLERR, "Illegal pair_style command");
  }
  if (narg == 1)
  {
    std::string model_path = arg[0];
    std::cout << model_path << std::endl;
    torchmolnet_.init(model_path, "cuda", true);
  }
  else
  {
    std::string model_path = arg[0];
    std::cout << model_path << std::endl;
    torchmolnet_.init(model_path, arg[1], true);
  }
  numb_types_ = torchmolnet_.get_z_max();
}

void PairTorchMolNet::print_summary(const std::string pre) const
{
  if (comm->me == 0)
  {
    std::cout << pre << "TorchMolNet LAMMPS Plugin:" << std::endl;
    std::cout << pre << "  Float precision: " << STR_FLOAT_PREC << std::endl;
    std::cout << pre << "  TorchMolNet root: " << STR_TORCHMOLNET_ROOT << std::endl;
    std::cout << pre << "  Torch include dirs: " << STR_Torch_INCLUDE_DIRS << std::endl;
    std::cout << pre << "  Torch libraries: " << STR_Torch_LIBRARY << std::endl;
    std::cout << pre << "TorchMolNet Package: " << std::endl;
    torchmolnet_.print_summary(pre);
  }
}

PairTorchMolNet::~PairTorchMolNet() {}

void PairTorchMolNet::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, xj, yj, zj, rij;
  int *ilist, *jlist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  numb_computes_++;
  debug_xyz_file_.open("debug/" + std::to_string(numb_computes_) + ".output", std::ofstream::out);

  std::vector<double> dcoord(nall * 3, 0.0);
  // get coord
  for (int ii = 0; ii < nall; ++ii)
  {
    for (int dd = 0; dd < 3; ++dd)
    {
      dcoord[ii * 3 + dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }

  // get type
  int newton_pair = force->newton_pair;
  std::vector<int> dtype(nall);
  for (int ii = 0; ii < nall; ++ii)
  {
    dtype[ii] = type[ii];
  }

  // get box
  std::vector<double> dbox(9, 0);
  dbox[0] = domain->h[0]; // xx
  dbox[4] = domain->h[1]; // yy
  dbox[8] = domain->h[2]; // zz
  dbox[7] = domain->h[3]; // zy
  dbox[6] = domain->h[4]; // zx
  dbox[3] = domain->h[5]; // yx

  // predict values.
  double denergy = 0.0;
  std::vector<double> dforces(nall * 3, 0.0);
  std::vector<double> deatoms(nall, 0.0);

  numb_computes_++;
  debug_xyz_file_.open("debug/" + std::to_string(numb_computes_) + ".debuglog", std::ofstream::out);
  // std::cout << "Computing..." << std::endl;
  // std::cout << "inum:" << inum << std::endl;

  double atom_wise_denergy = 0.0;
  double sum_of_atom_wise_deatoms = 0.0;
  std::vector<double> vector_of_atom_wise_deatoms(nlocal, 0.0);
  std::vector<int> atom_wise_dtype(nall, -1);
  std::vector<double> atom_wise_dcoord(nall * 3, 0.0);
  std::vector<double> atom_wise_dforces(nall * 3, 0.0);
  std::vector<double> atom_wise_deatoms(nall, 0.0);

  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    debug_xyz_file_ << jnum << std::endl;

    // std::cout << "i:" << i << " jnum:" << jnum << std::endl;
    // debug_xyz_file_ << "nlocal + nghost:" << nall << " "
    //                 << "nlocal:" << nlocal << std::endl;
    // debug_xyz_file_ << itype << " " << xtmp << " " << ytmp << " " << ztmp << "#"
    //                 << " " << i + 1 << " " << std::endl;
    for(int dim=0;dim<3;dim++){
      atom_wise_dcoord[dim]=x[i][dim];
    }
    atom_wise_dtype[0]=itype;
    for (jj = 0; jj < jnum; jj++)
    {
      j = jlist[jj];
      j &= NEIGHMASK;
      xj = x[j][0];
      yj = x[j][1];
      zj = x[j][2];
      // rij = sqrt((xtmp - xj) * (xtmp - xj) + (ytmp - yj) * (ytmp - yj) + (ztmp - zj) * (ztmp - zj));
      jtype = type[j];
      atom_wise_dcoord[(jj+1)*3+0]=xj;
      atom_wise_dcoord[(jj+1)*3+1]=yj;
      atom_wise_dcoord[(jj+1)*3+2]=zj;
      atom_wise_dtype[jj+1]=jtype;

      // std::cout << "coordination of" << i << "(" << itype << "):(" << xtmp << "," << ytmp << "," << ztmp << "),";
      // std::cout << "coordination of" << j << "(" << jtype << "):(" << xj << "," << yj << "," << zj << ")" << std::endl;
      // std::cout << "distance of i,j: " << rij << std::endl;

      // debug_xyz_file_ << jtype << " " << xj << " " << yj << " " << zj << "#"
      //                 << " " << j + 1 << " " << rij << std::endl;
    }
    // atom-wise calculation
    torchmolnet_.predict(atom_wise_denergy, atom_wise_dforces, atom_wise_dcoord, atom_wise_dtype, dbox, jnum, atom_wise_deatoms);

    // get atom-wise energy
    sum_of_atom_wise_deatoms += atom_wise_deatoms[0];
    vector_of_atom_wise_deatoms[ii] = atom_wise_deatoms[0];

    // get atom-wise force
    for (int dd = 0; dd < 3; ++dd)
    {
      f[ii][dd] = atom_wise_dforces[3 * ii + dd];
      dforces[3 * ii + dd] = atom_wise_dforces[dd]; // only the first atom
    }
  }
  // full calculation
  torchmolnet_.predict(denergy, dforces, dcoord, dtype, dbox, nall, deatoms);
  debug_xyz_file_ << "sum of atom-wise calculation energy: \n"
                  << sum_of_atom_wise_deatoms;
  debug_xyz_file_ << "total calculation energy: \n"
                  << denergy;
  debug_xyz_file_ << "total calculation atom-wise energy \t total calculation energy of each atom: \n";

  for (int ii = 0; ii < nlocal; ii++)
  {
    debug_xyz_file_ << vector_of_atom_wise_deatoms[ii] << "\t" << deatoms[ii];
  }

  debug_xyz_file_ << "total calculation force: \n"
                  << dforces;
  // return to lammps
  if (eflag_global)
    eng_vdwl += denergy;
  if (eflag_atom)
  {
    for (int ii = 0; ii < nall; ii++)
      eatom[ii] += deatoms[ii];
  }

  if (vflag_fdotr)
    virial_fdotr_compute();
  debug_xyz_file_.close();
  std::cout << "Computing end." << std::endl;
}

void PairTorchMolNet::coeff(int narg, char **arg)
{
  if (!allocated)
  {
    allocate();
  }

  int n = atom->ntypes;
  int ilo, ihi, jlo, jhi;
  ilo = 0;
  jlo = 0;
  ihi = n;
  jhi = n;
  if (narg == 2)
  {
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
    if (ilo != 1 || jlo != 1 || ihi != n || jhi != n)
    {
      error->all(FLERR, "This pair type requires that the scale should be set to all atom types, i.e. pair_coeff * *.");
    }
  }
  for (int i = ilo; i <= ihi; i++)
  {
    for (int j = MAX(jlo, i); j <= jhi; j++)
    {
      setflag[i][j] = 1;
      scale[i][j] = 1.0;
      if (i > numb_types_ || j > numb_types_)
      {
        char warning_msg[1024];
        sprintf(warning_msg, "Interaction between types %d and %d is set in your input file, but will be ignored.\n This model has only %d types, it only computes the multibody interaction of types: 1-%d.", i, j, numb_types_, numb_types_);
        error->warning(FLERR, warning_msg);
      }
    }
  }
}
void PairTorchMolNet::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(scale, n + 1, n + 1, "pair:scale");

  for (int i = 1; i <= n; i++)
  {
    for (int j = i; j <= n; j++)
    {
      setflag[i][j] = 0;
      scale[i][j] = 0;
    }
  }
  for (int i = 1; i <= numb_types_; ++i)
  {
    if (i > n)
      continue;
    for (int j = i; j <= numb_types_; ++j)
    {
      if (j > n)
        continue;
      setflag[i][j] = 1;
      scale[i][j] = 1;
    }
  }
}
double PairTorchMolNet::init_one(int i, int j)
{
  if (i > numb_types_ || j > numb_types_)
  {
    char warning_msg[1024];
    sprintf(warning_msg, "Interaction between types %d and %d is set in your input file, but will be ignored.\n This model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types_, numb_types_);
    error->warning(FLERR, warning_msg);
  }

  if (setflag[i][j] == 0)
    scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];

  return cutoff_;
}