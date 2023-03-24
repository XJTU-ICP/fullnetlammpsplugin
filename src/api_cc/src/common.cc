#include "common.h"
namespace torchmolnet
{
    void select_by_type(std::vector<int> &fwd_map,
                        std::vector<int> &bkw_map,
                        int &nghost_real,
                        const std::vector<double> &dcoord_,
                        const std::vector<int> &datype_,
                        const int &nghost,
                        const std::vector<int> &sel_type_)
    {
        std::vector<int> sel_type(sel_type_);
        sort(sel_type.begin(), sel_type.end());
        int nall = dcoord_.size() / 3;
        int nloc = nall - nghost;
        int nloc_real = 0;
        nghost_real = 0;
        fwd_map.resize(nall);
        bkw_map.clear();
        bkw_map.reserve(nall);
        int cc = 0;
        for (int ii = 0; ii < nall; ++ii)
        {
            // exclude virtual sites
            // select the type with id < ntypes
            if (binary_search(sel_type.begin(), sel_type.end(), datype_[ii]))
            {
                bkw_map.push_back(ii);
                if (ii < nloc)
                {
                    nloc_real += 1;
                }
                else
                {
                    nghost_real += 1;
                }
                fwd_map[ii] = cc;
                cc++;
            }
            else
            {
                fwd_map[ii] = -1;
            }
        }
        assert((nloc_real + nghost_real) == bkw_map.size());
    }
    void select_real_atoms(std::vector<int> &fwd_map,
                           std::vector<int> &bkw_map,
                           int &nghost_real,
                           const std::vector<double> &dcoord_,
                           const std::vector<int> &datype_,
                           const int &nghost,
                           const int &ntypes)
    {
        std::vector<int> sel_type;
        for (int ii = 0; ii < ntypes; ++ii)
        {
            sel_type.push_back(ii);
        }
        select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, sel_type);
    }
    int get_neighbors(
        torch::Tensor positions,
        torch::Tensor cell,
        float cutoff,
        torch::Tensor &idx_i,
        torch::Tensor &idx_j,
        torch::Tensor &cell_shifts)
    {
        // calculate the neighbors of atom and return pair and shift vectors.
        // positions: (N, 3)
        // box: (3)
        // cutoff: float
        // idx_i: (M,)
        // idx_j: (M,)
        // cell_shifts: (M, 3)

        torch::Tensor dis_vec = positions.view({-1, 1, 3}) - positions.view({1, -1, 3});
        torch::Tensor dis = torch::norm(dis_vec - cell.view({1, 1, 3}) * torch::round(dis_vec / cell.view({1, 1, 3})), 2, 2) + torch::eye(positions.size(0), torch::kFloat) * cutoff * 2;
        torch::Tensor neighbors = torch::nonzero(dis < cutoff);
        idx_i = neighbors.select(1, 0);
        idx_j = neighbors.select(1, 1);
        cell_shifts = torch::where(
                          dis_vec > cell / 2, 1, torch::where(dis_vec < -cell / 2, -1, 0))
                          .index({neighbors.select(1, 0), neighbors.select(1, 1)});
        return 0;
    }
} // namespace torchmolnet