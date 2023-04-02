#include "torch_mol_net.h"

namespace torchmolnet
{
    TorchMolNet::TorchMolNet() : m_device_(torch::Device(torch::kCUDA)), inited_(false), option_debug_(false)
    {
    }
    TorchMolNet::TorchMolNet(const std::string &model_path, const std::string &device, double cutoff, const bool option_debug) : m_device_(torch::Device(torch::kCUDA))
    {
        init(model_path, device, cutoff, option_debug);
    }

    TorchMolNet::~TorchMolNet()
    {
    }

    void TorchMolNet::init(const std::string &model_path, const std::string &device, double cutoff, const bool option_debug)
    {
        // Load the model from a file.
        m_model_path_ = model_path;
        option_debug_ = option_debug;
        cutoff_ = cutoff;

        if (option_debug_)
        {
            file_debug_.open("debuginfo.txt", std::ofstream::out);
            file_debug_.close();
        }

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
                std::cout << "Using device gpu." << std::endl;
                m_device_ = torch::Device(torch::kCUDA);
                m_model_.to(m_device_);
                inited_ = true;
            }
            else if (device == "cpu")
            {
                std::cout << "Using device cpu." << std::endl;

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
                              const long nneighbor,
                              std::vector<double> &deatoms)
    {
        // Create a tensor from the input vector.
        // torch::Tensor tensor_dcoord = torch::tensor(dcoord_, torch::kDouble).view({(int)dcoord_.size() / 3, 3});
        // torch::Tensor tensor_datype = torch::tensor(datype_, torch::kLong);
        // torch::Tensor tensor_dbox = torch::tensor(dbox, torch::kDouble).view({3, 3});
        // torch::Tensor tensor_nghost = torch::tensor(nghost, torch::kLong).view({1});
        long system_size = nneighbor + 1;

        // TODO: api for charge and mag_moment.
        double total_charge[] = {0.0};
        double mag_moment[] = {0.0};

        torch::jit::IValue total_charge_tensor = torch::from_blob(total_charge, {1}, torch::kDouble).to(m_device_);
        torch::jit::IValue mag_moment_tensor = torch::from_blob(mag_moment, {1}, torch::kDouble).to(m_device_);

        // std::vector<double> coords_inner(system_size * 3, 0.0);
        // std::vector<double> atomic_number_inner(system_size, -1.0);
        // for (long i = 0; i < system_size; i++)
        // {
        //     coords_inner[i * 3 + 0] = dcoord[i * 3 + 0];
        //     coords_inner[i * 3 + 1] = dcoord[i * 3 + 1];
        //     coords_inner[i * 3 + 2] = dcoord[i * 3 + 2];
        //     atomic_number_inner[i] = datype[i];
        // }

        torch::jit::IValue coords_tensor = torch::tensor(dcoord, torch::TensorOptions().dtype(torch::kDouble).requires_grad(true)).view({(int)system_size, 3}).to(m_device_);
        torch::jit::IValue atomic_number_tensor = torch::tensor(datype, torch::kLong).to(m_device_);
        torch::jit::IValue dbox_tensor = torch::tensor(dbox, torch::TensorOptions().dtype(torch::kDouble).device(m_device_)).diag().view({1, 3, 3});
        // TODO: more efficient way to generate idx.
        torch::Tensor tensor_idx_i;
        torch::Tensor tensor_idx_j;
        torch::Tensor tensor_cell_shifts;
        get_neighbors(
            coords_tensor.toTensor(),
            torch::tensor(dbox, torch::kDouble).view({3}).to(m_device_),
            cutoff_,
            tensor_idx_i,
            tensor_idx_j,
            tensor_cell_shifts);

        std::vector<torch::jit::IValue> inputs;

        inputs.push_back(atomic_number_tensor);
        inputs.push_back(total_charge_tensor);
        inputs.push_back(mag_moment_tensor);
        inputs.push_back(coords_tensor);
        inputs.push_back(tensor_idx_i);
        inputs.push_back(tensor_idx_j);
        inputs.push_back(dbox_tensor);
        inputs.push_back(tensor_cell_shifts);

        // write info to fp
        if (option_debug_)
        {
            file_debug_.open("debuginfo.txt", std::ofstream::app);
            file_debug_ << "###----INPUT----###" << std::endl;
            file_debug_ << "atomic_number_tensor: \n"
                        << atomic_number_tensor.toTensor().reshape({1, -1}) << std::endl;
            file_debug_ << "total_charge_tensor: \n"
                        << total_charge_tensor.toTensor() << std::endl;
            file_debug_ << "mag_moment_tensor: \n"
                        << mag_moment_tensor.toTensor() << std::endl;
            file_debug_ << "coords_tensor: \n"
                        << coords_tensor.toTensor().reshape({-1, 3}) << std::endl;
            file_debug_ << "dbox_tensor: \n"
                        << dbox_tensor.toTensor() << std::endl;
            file_debug_ << "###----INPUT----###" << std::endl;
            file_debug_.close();
        }

        try
        {
            if (option_debug_)
            {
                std::cout << "Running on model..." << std::endl;
            }
            auto out = m_model_.forward(inputs).toTuple();
            if (option_debug_)
            {
                std::cout << "Model prediction done." << std::endl;
            }
            denergy = out->elements()[0].toTensor().item<double>();
            // transfer force matrix to cpu and vector
            torch::Tensor force_tensor = out->elements()[1].toTensor().to(torch::kCPU).to(torch::kDouble).flatten();
            torch::Tensor energy_wise_tensor = out->elements()[4].toTensor().to(torch::kCPU).to(torch::kDouble).flatten();

            dforces = std::vector<double>(force_tensor.data_ptr<double>(), force_tensor.data_ptr<double>() + force_tensor.numel());
            deatoms = std::vector<double>(energy_wise_tensor.data_ptr<double>(), energy_wise_tensor.data_ptr<double>() + energy_wise_tensor.numel());
            if (option_debug_)
            {
                file_debug_.open("debuginfo.txt", std::ofstream::app);
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
                file_debug_.close();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "error calculating.\n";
            std::cout << e.what();
            exit(1);
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

    int TorchMolNet::get_cutoff() const
    {
        if (inited_)
        {
            return cutoff_;
        }
        else
        {
            throw std::runtime_error("TorchMolNet not inited");
            return -1;
        }
    }

} // namespace torchmolnet