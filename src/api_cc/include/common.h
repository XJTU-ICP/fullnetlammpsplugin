#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "version.h"

#include <torch/script.h> // One-stop header for libtorch.

namespace torchmolnet
{
    void select_by_type(std::vector<int> &fwd_map,
                        std::vector<int> &bkw_map,
                        int &nghost_real,
                        const std::vector<double> &dcoord_,
                        const std::vector<int> &datype_,
                        const int &nghost,
                        const std::vector<int> &sel_type_);
    void select_real_atoms(std::vector<int> &fwd_map,
                           std::vector<int> &bkw_map,
                           int &nghost_real,
                           const std::vector<double> &dcoord_,
                           const std::vector<int> &datype_,
                           const int &nghost,
                           const int &ntypes);
    struct InputNlist
    {
        /// Number of core region atoms
        int inum;
        /// Array stores the core region atom's index
        int *ilist;
        /// Array stores the core region atom's neighbor atom number
        int *numneigh;
        /// Array stores the core region atom's neighbor index
        int **firstneigh;
        InputNlist()
            : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL){};
        InputNlist(
            int inum_,
            int *ilist_,
            int *numneigh_,
            int **firstneigh_)
            : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_){};
        ~InputNlist(){};
    };
} // namespace torchmolnet