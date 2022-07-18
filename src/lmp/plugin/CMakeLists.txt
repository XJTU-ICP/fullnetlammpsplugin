if(DEFINED LAMMPS_HEADER_DIR OR DEFINED LAMMPS_VERSION)
  message(STATUS "enable LAMMPS plugin mode")
  add_library(lammpsPluginTest INTERFACE)
  set(LAMMPS_HEADER_DIR ${LAMMPS_HEADER_DIR})
  message(STATUS "LAMMPS_HEADER_DIR is ${LAMMPS_HEADER_DIR}")

  target_include_directories(lammpsPluginTest INTERFACE ${LAMMPS_HEADER_DIR})
  find_package(LAMMPS)
  message(STATUS "LAMMPS version: ${LAMMPS_VERSION}")
  message(STATUS "LAMMPS_HEADER_DIR is ${LAMMPS_HEADER_DIR}")
  target_link_libraries(lammpsPluginTest INTERFACE LAMMPS::lammps)

  find_package(MPI)
  if(MPI_FOUND)
    set(LAMMPS_MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
    message(STATUS "LAMMPS_MPI_INCLUDE_DIRS is ${LAMMPS_MPI_INCLUDE_DIRS}")
    message(STATUS "MPI_CXX_LIBRARIES
    is ${MPI_CXX_LIBRARIES}")
    message(STATUS "MPI_LIBRARIES
    is ${MPI_LIBRARIES}")
    target_link_libraries(lammpsPluginTest INTERFACE MPI::MPI_CXX)
  else()
    # Use LAMMPS serial mpi.h header
    target_include_directories(lammpsPluginTest
                               INTERFACE "${LAMMPS_SOURCE_DIR}/src/STUBS")
  endif()

  # get_lammps_version
  # https://github.com/lammps/lammps/blob/c2a12f97c5f665852fb38fdd4922f7dd2e77a0a1/cmake/Modules/LAMMPSUtils.cmake#L27-L46
  include(${LAMMPS_SOURCE_DIR}/cmake/Modules/LAMMPSUtils.cmake)
  get_lammps_version(${LAMMPS_SOURCE_DIR}/src/version.h LAMMPS_VERSION_NUMBER)
  set(LAMMPS_VERSION_NUMBER
      ${LAMMPS_VERSION_NUMBER}
      PARENT_SCOPE)

  configure_file("../pair_torchmolnet.h.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/pair_torchmolnet.h" @ONLY)

  file(GLOB LMP_SRC torchmol_lmpplugin.cpp ../*.cpp)

  function(_add_lmp_plugin_variant variant_name prec_def)
    set(libname "torchmolnet_lmp${variant_name}")

    add_library(${libname} MODULE ${LMP_SRC})

    # link: libtorchmolnet
    target_link_libraries(
      ${libname} PUBLIC lammpsPluginTest ${LIB_TORCHMOLNET_CC}${variant_name})
    message("CMAKE_CURRENT_BINARY_DIR" ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(
      ${libname}
      PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
              ${LAMMPS_SOURCE_DIR}/src)

    set_target_properties(
      ${libname} PROPERTIES INSTALL_RPATH "$ORIGIN;${TORCH_LIBRARIES}"
                            LINK_FLAGS "-rdynamic")

    target_compile_definitions(
      ${libname}
      PUBLIC ${prec_def}
      PRIVATE "LMPPLUGIN" # fix header path
      PRIVATE "LAMMPS_VERSION_NUMBER=${LAMMPS_VERSION_NUMBER}")
    install(TARGETS ${libname} DESTINATION lib/)

    if(${LAMMPS_VERSION_NUMBER} GREATER_EQUAL 20220324)
      set(PLUGINNAME "torchmolplugin_lmp.so")
      install(
        CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E make_directory \
		${CMAKE_INSTALL_PREFIX}/lib/${libname}/   \
        )")
      install(
        CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
		../${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${CMAKE_SHARED_LIBRARY_SUFFIX} \
        ${CMAKE_INSTALL_PREFIX}/lib/${libname}/${PLUGINNAME}   \
        )")
    endif()

  endfunction()
  _add_lmp_plugin_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
  _add_lmp_plugin_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")

else()
  message(STATUS "disable LAMMPS plugin mode")
endif()
