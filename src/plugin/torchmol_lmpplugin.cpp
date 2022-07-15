/**
* See https://docs.lammps.org/Developer_plugins.html
*/
#include "lammpsplugin.h"
#include "version.h"

using namespace LAMMPS_NS;

static Pair *pair_torchmolnet(LAMMPS *lmp)
{
  return new PairTorchMolNet(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "torchmolnet";
  plugin.info = "torchmolnet pair style";
  plugin.author = "Han Yanbo";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &pair_torchmolnet;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
