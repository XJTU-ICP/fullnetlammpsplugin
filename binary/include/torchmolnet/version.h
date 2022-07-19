#pragma once

#include <string>
// using namespace std;

#ifdef NOT_HIGH_PREC
const std::string global_float_prec = "float";
#else
const std::string global_float_prec = "double";
#endif

const std::string global_install_prefix = "/root/lammpsPluginTest/binary";
const std::string global_git_summ = "";
const std::string global_git_hash = "9e0c0c7";
const std::string global_git_date = "2022-07-19 00:15:04 +0800";
const std::string global_git_branch = "main";
const std::string global_model_version = "";
