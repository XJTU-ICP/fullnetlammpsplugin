#pragma once

#include "common.h"

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
         *
         */
        TorchMolNet();
        /**
         * @brief Construct a new Torch Mol Net object
         *
         * @param[in] model_path Path to the model file.
         * @param[in] device The device to use.
         */
        TorchMolNet(const std::string &model_path, const std::string &device = "cuda");

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
         */
        void init(const std::string &model_path, const std::string &device = "cuda");

        /*function only for development*/
        void predict(double &denergy,
                     std::vector<double> &dforces,
                     const std::vector<double> &dcoord_,
                     const std::vector<int> &datype_,
                     const std::vector<double> &dbox,
                     const int nghost);

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
        bool inited;
    };

} // namespace torchmolnet
