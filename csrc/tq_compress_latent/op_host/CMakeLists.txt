add_ops_compile_options(
        OP_NAME TqCompressLatent
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnn PRIVATE
        tq_compress_latent_def.cpp
)

target_sources(optiling PRIVATE
        tq_compress_latent_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            tq_compress_latent_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        tq_compress_latent_infershape.cpp
)

set(TQ_COMPRESS_CANN_PKG_INC
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
)

target_include_directories(op_host_aclnn PRIVATE
        ${TQ_COMPRESS_CANN_PKG_INC}
)
target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TQ_COMPRESS_CANN_PKG_INC}
)
target_link_libraries(optiling PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/libops_base.so)
target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TQ_COMPRESS_CANN_PKG_INC}
)

if (NOT BUILD_OPEN_PROJECT)
    target_include_directories(opmaster_ct PRIVATE
            ${TQ_COMPRESS_CANN_PKG_INC}
    )
endif ()
