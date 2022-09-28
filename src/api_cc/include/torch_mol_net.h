#pragma once

#include "common.h"
#include <fstream>

namespace torchmolnet
{
    /**
     * @brief TorchMolNet is a wrapper class for the TorchMolNet library.
     */
    class TorchMolNet
    {
    public:
        /**
         * @brief Construct a new Torch Mol Net object
         */
        TorchMolNet();
        /**
         * @brief Construct a new Torch Mol Net object
         *
         * @param[in] model_path Path to the model file.
         * @param[in] device The device to use.
         * @param[in] option_debug Whether to print debug information.
         */
        TorchMolNet(const std::string &model_path, const std::string &device = "cuda", bool option_debug = false);

        /**
         * @brief Destroy the Torch Mol Net object
         *
         */
        ~TorchMolNet();

        /**
         * @brief Load the model from the given path.
         *
         * @param[in] model_path Path to the model file.
         * @param[in] device The device name to use.
         * @param[in] option_debug Whether to print debug information.
         */
        void init(const std::string &model_path, const std::string &device = "cuda", bool option_debug = false);

        /**
         * @brief Predict the energy and forces for the given molecule.
         *
         * @param[in] dcoord The coordinates of the molecule.
         * @param[in] datype The atom types of the molecule.
         * @param[in] dbox The cell of the molecule. (not used)
         * @param[out] denergy The energy of the molecule.
         * @param[out] dforces The forces of each atom in the molecule.
         */
        void predict(double &denergy,
                     std::vector<double> &dforces,
                     const std::vector<double> &dcoord,
                     const std::vector<int> &datype,
                     const std::vector<double> &dbox,
                     const int nghost,
                     std::vector<double> &deatoms);

        /**
         * @brief Pring the model summary.
         *
         * @param[in] pre Prefix for the summary.
         */
        void print_summary(const std::string pre = ">>>") const;

        /**
         * @brief Get model's z_max.
         *
         * @return Model's z_max.
         */
        int get_z_max() const;

    private:
        /**
         * @brief The model path to use.
         */
        std::string m_model_path_;
        /**
         * @brief The torch model by jit.
         */
        torch::jit::Module m_model_;
        /**
         * @brief The device to use.
         */
        torch::Device m_device_;
        /**
         * @brief Whether the model is loaded.
         */
        bool inited_;

        /**
         * @brief Whether running with debug information for models.
         */
        bool option_debug_;

        /**
         * @brief The debug file to use when setting debug_option=True.
         */
        std::fstream file_debug_;
    };

} // namespace torchmolnet
