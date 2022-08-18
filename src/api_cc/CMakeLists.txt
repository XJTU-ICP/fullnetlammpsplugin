# libmd

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)


function(_add_libapicc_variant variant_name prec_def)
set (libname "${LIB_TORCHMOLNET_CC}${variant_name}")

add_library(${libname} SHARED ${LIB_SRC})

# link: libtorchmolnet libtorchmolnet_op
target_link_libraries (${libname} PUBLIC ${TORCH_LIBRARIES})
target_include_directories(${libname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(
  ${libname} 
  PROPERTIES 
  COMPILE_DEFINITIONS ${prec_def}
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/torchmolnet
)

endfunction()

_add_libapicc_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_libapicc_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")