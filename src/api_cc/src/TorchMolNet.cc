#include "TorchMolNet.h"

namespace torchmolnet
{
    TorchMolNet::TorchMolNet(const std::string &model_path, const std::string &device)
    {
        // Load the model from a file.
        m_model_path_ = model_path;
        try
        {
            m_model_ = torch::jit::load(model_path);
        }
        catch (const c10::Error &e)
        {
            throw std::runtime_error(e.what());
        }
        m_device_ = torch::Device(&device);
        m_model_.to(m_device_);
    }

    TorchMolNet::~TorchMolNet()
    {
    }

    void TorchMolNet::predict(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs)
    {
        // Create a tensor from the input vector.
        std::cout << "inputs size: " << inputs.size() << std::endl;
        std::cout << "inputs is:" << std::endl;
        for (int i = 0; i < inputs.size(); i++)
        {
            std::cout << inputs[i] << std::endl;
        }

        return output_vector;
    }

    void TorchMolNet::print_summary() const
    {
        std::cout << "TorchMolNet summary:" << std::endl;
        std::cout << "  Model path:       " << m_model_path << std::endl;
        std::cout << "  Device:           " << m_device << std::endl;
        std::cout << "Module build summary:     " << std::endl;
        std::cout << "  Installed to:     " << global_install_prefix << std::endl;
        std::cout << "  Git summary:      " << global_git_summ << std::endl;
        std::cout << "  Git hash:         " << global_git_hash << std::endl;
        std::cout << "  Git date:         " << global_git_date << std::endl;
        std::cout << "  Git branch:       " << global_git_branch << std::endl;
        std::cout << "  Model version:    " << global_model_version << std::endl;
        std::cout << "  Float precision:  " << global_float_prec << std::endl;
        std::cout << std::endl;
    }

} // namespace torchmolnet