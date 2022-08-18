#include <iostream>
#include <string.h>
#include <iomanip>
#include <limits>
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
  cutoff = 5.;
  print_summary();
}

void PairTorchMolNet::settings(int narg, char **arg)
{
  if (narg > 2)
  {
    error->all(FLERR, "Illegal pair_style command");
  }
  if (narg == 1)
  {
    std::string model_path = arg[0];
    std::cout << model_path << std::endl;
    torchmolnet.init(model_path, "cuda");
  }
  else
  {
    std::string model_path = arg[0];
    std::cout << model_path << std::endl;
    torchmolnet.init(model_path, arg[1]);
  }
  numb_types_ = torchmolnet.get_z_max();
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
    torchmolnet.print_summary(pre);
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

  torchmolnet.predict(denergy, dforces, dcoord, dtype, dbox, nghost);

  // get force
  for (int ii = 0; ii < nall; ++ii)
  {
    for (int dd = 0; dd < 3; ++dd)
    {
      f[ii][dd] = dforces[3 * ii + dd];
    }
  }

  // return to lammps
  if (eflag)
  {
    eng_vdwl = denergy;
  }

  if (vflag_fdotr)
    virial_fdotr_compute();
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

  return cutoff;
}