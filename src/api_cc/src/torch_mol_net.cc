#include "torch_mol_net.h"

namespace torchmolnet
{
    TorchMolNet::TorchMolNet() : m_device_(torch::Device(torch::kCUDA)), inited_(false), option_debug_(false)
    {
    }
    TorchMolNet::TorchMolNet(const std::string &model_path, const std::string &device, const bool option_debug) : m_device_(torch::Device(torch::kCUDA))
    {
        init(model_path, device, option_debug);
    }

    TorchMolNet::~TorchMolNet()
    {
    }

    void TorchMolNet::init(const std::string &model_path, const std::string &device, const bool option_debug)
    {
        // Load the model from a file.
        m_model_path_ = model_path;
        option_debug_ = option_debug;

        if (option_debug_)
            file_debug_.open("debuginfo.txt", std::ofstream::out);

        try
        {
            m_model_ = torch::jit::load(model_path);
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Error loading the model from file: " << model_path << std::endl;
            throw std::runtime_error(e.what());
        }
        try
        {
            if (device == "cuda")
            {
                std::cout << "Using device as gpu: " << device << std::endl;
                m_device_ = torch::Device(torch::kCUDA);
                m_model_.to(m_device_);
                inited_ = true;
            }
            else if (device == "cpu")
            {
                std::cout << "Using device as cpu: " << device << std::endl;

                m_device_ = torch::Device(torch::kCPU);
                m_model_.to(m_device_);
                inited_ = true;
            }
            else
            {
                std::cout << "Using device as unknown: " << device << std::endl;

                throw std::runtime_error("Unknown device: " + device);
            }
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Error initializing the model on device: " << device << std::endl;
            throw std::runtime_error(e.what());
        }
    }

    void TorchMolNet::predict(double &denergy,
                              std::vector<double> &dforces,
                              const std::vector<double> &dcoord,
                              const std::vector<int> &datype,
                              const std::vector<double> &dbox,
                              const int nghost,
                              std::vector<double> &deatoms)
    {
        // Create a tensor from the input vector.
        // torch::Tensor tensor_dcoord = torch::tensor(dcoord_, torch::kDouble).view({(int)dcoord_.size() / 3, 3});
        // torch::Tensor tensor_datype = torch::tensor(datype_, torch::kLong);
        // torch::Tensor tensor_dbox = torch::tensor(dbox, torch::kDouble).view({3, 3});
        // torch::Tensor tensor_nghost = torch::tensor(nghost, torch::kLong).view({1});
        long system_size = (int)dcoord.size() / 3;

        // TODO: api for charge and mag_moment.
        double total_charge[] = {0.0};
        double mag_moment[] = {0.0};
        // TODO: more efficient way to generate idx.
        torch::Tensor tensor_idx = torch::arange(0, system_size, torch::kLong);
        torch::Tensor tensor_idx_i_tmp = tensor_idx.view({system_size, 1}).expand({system_size, system_size}).reshape(system_size * system_size);
        torch::Tensor tensor_idx_j_tmp = tensor_idx.view({1, system_size}).expand({system_size, system_size}).reshape(system_size * system_size);
        // exclude self-interactions
        torch::Tensor tensor_idx_i = tensor_idx_i_tmp.index({tensor_idx_i_tmp != tensor_idx_j_tmp}).to(m_device_);
        torch::Tensor tensor_idx_j = tensor_idx_j_tmp.index({tensor_idx_i_tmp != tensor_idx_j_tmp}).to(m_device_);

        torch::jit::IValue total_charge_tensor = torch::from_blob(total_charge, {1}, torch::kDouble).to(m_device_);
        torch::jit::IValue mag_moment_tensor = torch::from_blob(mag_moment, {1}, torch::kDouble).to(m_device_);
        torch::jit::IValue coords_tensor = torch::tensor(dcoord, torch::TensorOptions().dtype(torch::kDouble).requires_grad(true)).view({(int)dcoord.size() / 3, 3}).to(m_device_);
        torch::jit::IValue atomic_number_tensor = torch::tensor(datype, torch::kLong).to(m_device_);

        std::vector<torch::jit::IValue> inputs;

        inputs.push_back(atomic_number_tensor);
        inputs.push_back(total_charge_tensor);
        inputs.push_back(mag_moment_tensor);
        inputs.push_back(coords_tensor);
        inputs.push_back(tensor_idx_i);
        inputs.push_back(tensor_idx_j);

        // write info to fp
        if (option_debug_)
        {
            file_debug_ << "###----INPUT----###" << std::endl;
            file_debug_ << "atomic_number_tensor: \n"
                        << atomic_number_tensor.toTensor().reshape({1, -1}) << std::endl;
            file_debug_ << "total_charge_tensor: \n"
                        << total_charge_tensor.toTensor() << std::endl;
            file_debug_ << "mag_moment_tensor: \n"
                        << mag_moment_tensor.toTensor() << std::endl;
            file_debug_ << "coords_tensor: \n"
                        << coords_tensor.toTensor().reshape({-1, 3}) << std::endl;
            file_debug_ << "###----INPUT----###" << std::endl;
        }

        try
        {
            auto out = m_model_.forward(inputs).toTuple();
            denergy = out->elements()[0].toTensor().item<double>();
            // transfer force matrix to cpu and vector
            torch::Tensor force_tensor = out->elements()[1].toTensor().to(torch::kCPU).to(torch::kDouble).flatten();
            torch::Tensor energy_wise_tensor = out->elements()[4].toTensor().to(torch::kCPU).to(torch::kDouble).flatten();

            dforces = std::vector<double>(force_tensor.data_ptr<double>(), force_tensor.data_ptr<double>() + force_tensor.numel());
            deatoms = std::vector<double>(energy_wise_tensor.data_ptr<double>(), energy_wise_tensor.data_ptr<double>() + energy_wise_tensor.numel());
            if (option_debug_)
            {
                file_debug_ << "###----OUTPUT----###" << std::endl;
                file_debug_ << "energy: \n"
                            << denergy << std::endl;
                // print force matrix
                file_debug_ << "force:\n"
                            << std::endl;
                for (int i = 0; i < dforces.size() / 3; i++)
                {
                    file_debug_ << " " << dforces[i * 3] << " " << dforces[i * 3 + 1] << " " << dforces[i * 3 + 2] << std::endl;
                }
                file_debug_ << "###----OUTPUT----###" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "error calculating.\n";
            std::cout << e.what();
            return;
        }
    }

    void TorchMolNet::print_summary(const std::string pre) const
    {
        std::cout << pre << "TorchMolNet summary:" << std::endl;
        std::cout << pre << "  Model path:       " << m_model_path_ << std::endl;
        std::cout << pre << "  Default torch device:           " << m_device_ << std::endl;
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

    int TorchMolNet::get_z_max() const
    {
        if (inited_)
        {
            // TODO: get the max atomic number from the model.
            return 87;
        }
        else
        {
            throw std::runtime_error("TorchMolNet not inited");
            return -1;
        }
    }

} // namespace torchmolnet