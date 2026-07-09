add_ops_compile_options(
        OP_NAME TurboQuantSparseFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnn PRIVATE
        turbo_quant_sparse_flash_attention_def.cpp
)

target_sources(optiling PRIVATE
        turbo_quant_sparse_flash_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        turbo_quant_sparse_flash_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        turbo_quant_sparse_flash_attention_infershape.cpp
)

set(TURBOQUANT_CANN_PKG_INC
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
)

target_include_directories(op_host_aclnn PRIVATE
        ${TURBOQUANT_CANN_PKG_INC}
)
target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TURBOQUANT_CANN_PKG_INC}
)
target_link_libraries(optiling PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/libops_base.so)
target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TURBOQUANT_CANN_PKG_INC}
)

if (NOT BUILD_OPEN_PROJECT)
    target_include_directories(opmaster_ct PRIVATE
            ${TURBOQUANT_CANN_PKG_INC}
    )
endif ()
