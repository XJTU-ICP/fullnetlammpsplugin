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
#ifdef USE_TTM
#include "fix_ttm_mod.h"
#endif

#include "pair_torchmolnet.h"

using namespace LAMMPS_NS;

static const char cite_torch_mol_net_package[] =
    "TorchMolNet package:\n\n"
    "@misc{Han_TorchMolNet,\n"
    "  author = {Han, Yanbo},\n"
    "  url = {https://github.com/saltball/lammpsPluginTest},\n"
    "  year = 2022,\n"
    "}\n\n";

PairTorchMolNet::PairTorchMolNet(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme)
    lmp->citeme->add(cite_torch_mol_net_package);
  if (strcmp(update->unit_style, "metal") != 0)
  {
    error->all(FLERR, "Pair torchmolnet requires metal unit, please set it by \"units metal\", see https://docs.lammps.org/units.html for details.\n");
  }
  print_summary();
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

void PairTorchMolNet::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp;
  int *ilist, *jlist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++)
    {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      std::cout << "i=" << i << " j=" << j << std::endl;
      std::cout << "itype=" << itype << " jtype=" << jtype << std::endl;
    }
  }