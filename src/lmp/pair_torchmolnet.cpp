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
  restartinfo = 0;
  manybody_flag = 1;
  if (lmp->citeme)
    lmp->citeme->add(cite_torch_mol_net_package);
  if (strcmp(update->unit_style, "metal") != 0)
  {
    error->all(FLERR, "Pair torchmolnet requires metal unit, please set it by \"units metal\", see https://docs.lammps.org/units.html for details.\n");
  }
  print_summary();
}

void PairTorchMolNet::settings(int narg, char **arg)
{
  numb_computes_ = 0;
  // if (std::filesystem::exists("debug"))
  // {
  //   std::filesystem::remove_all("debug");
  // }
  // std::filesystem::create_directory("debug");
#pragma omp master
  {
    if (narg > 4)
    {
      error->all(FLERR, "Illegal pair_style command");
    }
    if (narg == 1)
    {
      std::string model_path = arg[0];
      std::cout << "Load model from " << model_path << std::endl;
      std::cout << "Device set to default value cuda." << std::endl;
      torchmolnet_.init(model_path, "cuda");
    }
    else if (narg == 2)
    {
      std::string model_path = arg[0];
      std::cout << "Load model from " << model_path << std::endl;
      std::cout << "Device set input value " << arg[1] << std::endl;
      torchmolnet_.init(model_path, arg[1]);
    }
    else if (narg == 3)
    {
      std::string model_path = arg[0];
      std::cout << "Load model from " << model_path << std::endl;
      std::cout << "Device set input value " << arg[1] << std::endl;
      std::cout << "Cutoff set input value " << arg[2] << std::endl;
      torchmolnet_.init(model_path, arg[1], std::stod(arg[2]));
    }
    else
    {
      std::string model_path = arg[0];
      std::cout << "Load model from " << model_path << std::endl;
      std::cout << "Device set input value " << arg[1] << std::endl;
      std::cout << "Cutoff set input value " << arg[2] << std::endl;
      std::string debug_option = arg[3];
      if (debug_option == "debug")
      {
        torchmolnet_.init(model_path, arg[1], std::stod(arg[2]), true);
      }
      else
      {
        torchmolnet_.init(model_path, arg[1], std::stod(arg[2]));
      }
    }
    cutoff_ = torchmolnet_.get_cutoff();
    std::cout << "Cutoff set to " << std::setprecision(16) << cutoff_ << std::endl;
    numb_types_ = torchmolnet_.get_z_max();
  }
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
// #pragma omp master
  {
    int i, j, ii, jj, inum, jnum, itype, jtype;
    double xtmp, ytmp, ztmp, xj, yj, zj, rij;
    int *ilist, *jlist, *numneigh, **firstneigh;
    int itag, jtag;

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
    tagint *tag = atom->tag;

    numb_computes_++;

    std::vector<double> dcoord(nlocal * 3, 0.0);


    // get type
    int newton_pair = force->newton_pair;
    std::vector<int> dtype(nlocal);

    // get box
    std::vector<double> dbox(9,0.0);
    dbox[0] = domain->boxhi[0] - domain->boxlo[0];
    
    dbox[3] = domain->xy;
    dbox[4] = domain->boxhi[1] - domain->boxlo[1];

    dbox[6] = domain->xz;
    dbox[7] = domain->yz;
    dbox[8] = domain->boxhi[2] - domain->boxlo[2];


    // predict values.
    double denergy = 0.0;
    std::vector<double> dforces(nlocal * 3, 0.0);
    std::vector<double> deatoms(nlocal, 0.0);
    std::vector<double> dstress(6, 0.0);
    torch::Tensor cell_inv = torch::tensor(dbox, torch::kDouble).view({3,3}).inverse().transpose(0,1);

    // calculate the number of neighbors under the cutoff
    int nneibors = std::accumulate(numneigh, numneigh + nlocal, 0);
    std::vector<long> idx_i(nneibors, -99);
    std::vector<long> idx_j(nneibors, -99);
    std::vector<double> cell_shifts(nneibors * 3, -99);
    int neigh_flag = 0;
    torch::Tensor cell_shift_tmp;
    std::vector<int> tag2i(inum, -99);
    std::vector<int> tag2all(nall, -99);
    #pragma omp for
    for (ii = 0; ii < inum; ii++){
      i = ilist[ii];
      itag = tag[i];
      dcoord[(itag-1)*3+0] = x[i][0];
      dcoord[(itag-1)*3+1] = x[i][1];
      dcoord[(itag-1)*3+2] = x[i][2];
      dtype[itag-1] = type[i];
    }
    #pragma omp for
    for (ii = 0; ii < inum; ii++){
      i = ilist[ii];
      itag = tag[i];
      tag2i[itag-1] = ii;
    }
    
    #pragma omp master
    for (ii = 0; ii < inum; ii++)
    {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];
      itag = tag[i];
      for(jj = 0; jj < jnum; jj++)
      {
        j = jlist[jj];
        j &= NEIGHMASK;
        jtype = type[j];
        jtag = tag[j];
        xj = x[j][0];
        yj = x[j][1];
        zj = x[j][2];
        rij = pow(xtmp - xj, 2) + pow(ytmp - yj, 2) + pow(ztmp - zj, 2);

          idx_i[neigh_flag] = itag - 1;
          idx_j[neigh_flag] = jtag - 1;
          cell_shift_tmp = cell_inv.matmul(torch::tensor({xj - dcoord[(jtag-1)*3+0], yj - dcoord[(jtag-1)*3+1], zj - dcoord[(jtag-1)*3+2]}, torch::kDouble));
          for (int dd = 0; dd < 3; ++dd)
          {
            cell_shifts[neigh_flag * 3 + dd] = std::round(cell_shift_tmp[dd].item<double>());
          }
            // std::cout << "itag=" << itag << " jtag=" << jtag << std::endl;
            // std::cout << "coord:" << dcoord[(jtag-1)*3+0] << " " << dcoord[(jtag-1)*3+1] << " " << dcoord[(jtag-1)*3+2] << std::endl;
            // std::cout << "xj=" << xj << " yj=" << yj << " zj=" << zj << std::endl;
            // std::cout << "cell_shift_tmp=" << cell_shift_tmp << std::endl;
            // std::cout << "cell_shifts=" << cell_shifts[neigh_flag * 3 + 0] << " " << cell_shifts[neigh_flag * 3 + 1] << " " << cell_shifts[neigh_flag * 3 + 2] << std::endl;
        neigh_flag++;
      }
    }



    // full calculation
    torchmolnet_.predict(
      denergy, 
      dforces, 
      dstress,
      deatoms,
      dcoord, 
      dbox,
      dtype, 
      idx_i,
      idx_j,
      cell_shifts
    );

    // get force
    #pragma omp for
    for (itag = 0; itag < inum; itag++)
    {
      i = tag2i[itag];
      f[i][0] = dforces[itag*3+0];
      f[i][1] = dforces[itag*3+1];
      f[i][2] = dforces[itag*3+2];
      // std::cout<<"atom i=" << i << " force set to" << f[i][0] << " " << f[i][1] << " " << f[i][2] << std::endl;
    }
    // get force for ghost atoms
    #pragma omp for
    for (i = inum; i < inum + nghost; i++)
    {
      itag = tag[i];
      f[i][0] = dforces[(itag-1)*3+0];
      f[i][1] = dforces[(itag-1)*3+1];
      f[i][2] = dforces[(itag-1)*3+2];
      // std::cout<<"ghost atom i=" << i << " force set to" << f[i][0] << " " << f[i][1] << " " << f[i][2] << std::endl;
    }
    // return to lammps
    if (eflag_global)
      eng_vdwl += denergy;
    if (eflag_atom)
    {
      #pragma omp for
      for (itag = 0; itag < inum; itag++)
      {
        i = tag2i[itag];
        eatom[i] += deatoms[itag];
      }
    }

    // // always compute virial
    virial_fdotr_compute();
    // std::cout<<"Virial:"<<std::endl;
    // for (ii=0; ii < 6; ii++){
    //   std::cout<<virial[ii]<<" ";
    // }
    // std::cout<<std::endl;
    // #pragma omp for
    // for (int i = 0; i < nlocal; i++) {
    //   virial[0] += f[i][0]*x[i][0];
    //   virial[1] += f[i][1]*x[i][1];
    //   virial[2] += f[i][2]*x[i][2];
    //   virial[3] += f[i][1]*x[i][0];
    //   virial[4] += f[i][2]*x[i][0];
    //   virial[5] += f[i][2]*x[i][1];
    // }
    // std::vector<double> virial_compare(6,0.0);
    // for (int i =0; i < nlocal; i++){
    //   for (int j = 0; j < 3; j++){
    //     virial_compare[j] += f[i][j] * x[i][j];
    //   }
    // }
    // std::cout<<"Virial compare:"<<std::endl;
    // for (ii=0; ii < 6; ii++){
    //   std::cout<<virial_compare[ii]<<" ";
    // }
    // std::cout<<std::endl;
  }
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

void PairTorchMolNet::init_style()
{
#if LAMMPS_VERSION_NUMBER >= 20220324
  neighbor->add_request(this, NeighConst::REQ_FULL);
#else
#error only support Lammps >=20220324
#endif
}