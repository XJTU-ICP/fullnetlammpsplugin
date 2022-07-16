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
         * @param[in] model_path Path to the model file.
         * @param[in] device The device to use.
         */
        TorchMolNet(const std::string &model_path, const std::string &device);

        /**
         * @brief Destroy the Torch Mol Net object
         *
         */
        ~TorchMolNet();

        /**
         * @brief Predict the molecule properties.
         *
         * @param inputs The input molecule.
         * @param outputs The output molecule.
         */
        void predict(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> *outputs);

        /**
         * @brief Pring the model summary.
         *
         */
        void print_summary() const;

    private:
        std::string m_model_path;
        torch::jit::Module m_module_;
        torch::Device m_device_;
    };

} // namespace torchmolnet
