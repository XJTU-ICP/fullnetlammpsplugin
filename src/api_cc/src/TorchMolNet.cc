#include "TorchMolNet.h"

namespace torchmolnet
{
    TorchMolNet::TorchMolNet(const std::string &model_path, const std::string &device) : m_device_(torch::Device(torch::kCPU))
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
        if (device == "cuda")
        {
            m_device_ = torch::Device(torch::kCUDA);
            m_model_.to(m_device_);
        }
        else if (device == "cpu")
        {
            m_device_ = torch::Device(torch::kCPU);
            m_model_.to(m_device_);
        }
        else
        {
            throw std::runtime_error("Unknown device: " + device);
        }
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

        return;
    }

    void TorchMolNet::print_summary(const std::string pre) const
    {
        std::cout << pre << "TorchMolNet summary:" << std::endl;
        std::cout << pre << "  Model path:       " << m_model_path_ << std::endl;
        std::cout << pre << "  Device:           " << m_device_ << std::endl;
        std::cout << pre << "Module build summary:     " << std::endl;
        std::cout << pre << "  Installed to:     " << global_install_prefix << std::endl;
        std::cout << pre << "  Git summary:      " << global_git_summ << std::endl;
        std::cout << pre << "  Git hash:         " << global_git_hash << std::endl;
        std::cout << pre << "  Git date:         " << global_git_date << std::endl;
        std::cout << pre << "  Git branch:       " << global_git_branch << std::endl;
        std::cout << pre << "  Model version:    " << global_model_version << std::endl;
        std::cout << pre << "  Float precision:  " << global_float_prec << std::endl;
        std::cout << std::endl;
    }

} // namespace torchmolnet