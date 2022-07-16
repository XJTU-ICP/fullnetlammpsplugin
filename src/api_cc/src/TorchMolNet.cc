#include "TorchMolNet.h"

namespace torchmolnet
{
    TorchMolNet::TorchMolNet(const std::string &model_path, const std::string &device)
    {
        // Load the model from a file.
        try
        {
            m_module_ = torch::jit::load(model_path);
        }
        catch (const c10::Error &e)
        {
            throw std::runtime_error(e.what());
        }
        m_device_ = torch::Device(device);
        m_module_->to(m_device_);
    }

    TorchMolNet::~TorchMolNet()
    {
    }

    std::vector<float> TorchMolNet::predict(const std::vector<float> &features)
    {
        // Create a tensor from the input vector.
        torch::Tensor input = torch::from_blob(features.data(), {1, features.size()});
        input = input.to(m_device_);

        // Run the model.
        torch::Tensor output = m_model->forward(input).toTensor();

        // Convert the output to a vector.
        std::vector<float> output_vector(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

        return output_vector;
    }

    void TorchMolNet::print_summary()
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