cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_C_COMPILER "tecocc")
set(CMAKE_CXX_COMPILER "tecocc")

project(teco_ops)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Werror)

option(WITH_TORCH "Enable PyTorch support" OFF)
if(WITH_TORCH)
    find_package(Python3 REQUIRED COMPONENTS Development)
    find_package(Torch REQUIRED)
    add_definitions(-DWITH_TORCH)
    include_directories(${TORCH_INCLUDE_DIRS})
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${TORCH_SDAA_HOME}/include)
    get_filename_component(TORCH_LIB_DIR ${TORCH_LIBRARY} DIRECTORY)
    message(STATUS "WITH_TORCH is enabled")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "TORCH_LIB_DIR: ${TORCH_LIB_DIR}")
else()
    message(STATUS "WITH_TORCH is disabled")
endif()

option(WITH_INFERENCE_PLUGIN "Enable Teco-inference Plugin support" OFF)
if(WITH_INFERENCE_PLUGIN)
    add_definitions(-DWITH_INFERENCE_PLUGIN)
    file(GLOB_RECURSE SRC_PLUGIN "${OPS_DIR}/plugin/*/*.cc")
    set(FIND_SDAART "/opt/tecoai/lib64")
    set(FIND_PluginUtil "$ENV{TECO_INFER_PLUGIN_UTIL_PATH}")
    set(PLUGIN_LIBS ${FIND_SDAART}/libsdaart.so ${FIND_PluginUtil}/libTecoInferPluginUtil.so)
    set(PLUGIN_INCLUDES
        -I/opt/tecoai/include
        -I/opt/tecoai/include/dyn/include
        -I/opt/tecoai/include/dyn/src
        -I/opt/tecoai/include/teco_infer_dyn
        -I${OPS_DIR}
    )
    set(PLUGIN_COMPILE_FLAGS -O3 -std=c++17 -fPIC -DWITH_INFERENCE_PLUGIN)
    set(PLUGIN_OBJ_DIR ${CMAKE_BINARY_DIR}/plugin_objs)
    set(PLUGIN_OBJECTS "")
    add_custom_target(plugin_mkdir ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${PLUGIN_OBJ_DIR})
    foreach(src ${SRC_PLUGIN})
        get_filename_component(src_name ${src} NAME)
        list(APPEND PLUGIN_OBJECTS ${PLUGIN_OBJ_DIR}/${src_name}.o)
        add_custom_command(OUTPUT ${PLUGIN_OBJ_DIR}/${src_name}.o
            COMMAND g++ -c ${src} -o ${PLUGIN_OBJ_DIR}/${src_name}.o ${PLUGIN_COMPILE_FLAGS} ${PLUGIN_INCLUDES}
            DEPENDS ${src}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endforeach()
    add_custom_target(plugin_compile ALL DEPENDS plugin_mkdir ${PLUGIN_OBJECTS})
    message(STATUS "WITH_INFERENCE_PLUGIN is enabled")
else()
    message(STATUS "WITH_INFERENCE_PLUGIN is disabled")
endif()

set(TECOOPS_OPS_PATH ${OPS_DIR}/interface/ops)

file(GLOB_RECURSE SRC_INTERFACE_COMMON "${OPS_DIR}/interface/common/*.cpp")
file(GLOB_RECURSE SRC_INTERFACE_OPS "${OPS_DIR}/interface/ops/*.cpp")
file(GLOB_RECURSE SRC_UAL_COMMON "${OPS_DIR}/ual/com/*.cpp")
file(GLOB_RECURSE SRC_UAL_OPS "${OPS_DIR}/ual/ops/*/*.cpp")
file(GLOB_RECURSE SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/*/*.scpp")

# [c37 ops-tax TAX-1] The device-code section order inside .sdaa_fatbin equals the ORDER OF THE
# OBJECT FILES ON THE OFFICIAL LINK LINE (tecocc $<TARGET_OBJECTS:ops_objs> -flto ... --sdaa-link),
# which equals the target source-list order.  Measured byte-level on build root
# /root/tmpwork/c37/lead/ops-base @7ea7e84: permuting only the object order on the link command
# moves weight_stream_gemv's .aitext VMA 0x26418 (base) -> 0x25998 / 0x0cd68 / 0x2f0d4.
# This is the link-stage lever that c35 F1 ("link-time levers are all dead") did NOT test: F1
# swept section-attribute flags (--sort-section / --section-start / -falign-functions), not the
# input object order.  Evidence: /root/tmpwork/c37/ops-tax/RESULT.md sec.1 + arms/layout-*.txt.
# Pin weight_stream_gemv's device code immediately BEFORE rotary_embedding so that its .aitext
# VMA becomes 0x25998 -- the 128-byte-line neighbour of the only measured "fast" address 0x259dc
# (c33 window: lib-W-FAST2.so AHT 0.0041612 vs lib-A-asis.so 0x26418 AHT 0.0071927).
# Collateral is one section: rotary_embedding moves 0x25998 -> 0x267fc; every other section keeps
# its base VMA (proved by relink: arms/layout-before_rotary.txt).
list(FIND SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/weight_stream_gemv/weight_stream_gemv.scpp" _TAX1_WSG)
list(FIND SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/rotary_embedding/rotary_embedding_fp16.scpp" _TAX1_ROT)
if(_TAX1_WSG GREATER -1 AND _TAX1_ROT GREATER -1 AND _TAX1_WSG GREATER _TAX1_ROT)
    list(REMOVE_AT SRC_UAL_KERNEL ${_TAX1_WSG})
    list(INSERT SRC_UAL_KERNEL ${_TAX1_ROT} "${OPS_DIR}/ual/kernel/weight_stream_gemv/weight_stream_gemv.scpp")
endif()

# [c38 R10] Device-section order guard (build-time only; no hot-path effect).  Targets, measured in the
# cycle-38 windows: weight_stream_gemv -> 0x25a08 (wsg case_0 -31.0%), rotary_embedding -> 0x28858
# (rotary c0/c1/c2 -57.8%/-38.5%/-56.4%), zzz_flash_attention_d128 -> 0x2a878 (flash case_0 -2.41%).
# Same shape as the cycle-37 TAX-1 block: explicit FIND/REMOVE_AT/INSERT, idempotent, existence-guarded.
macro(_c38_r10_move_before _obj _anchor)
    list(FIND SRC_UAL_KERNEL "${_obj}" _oi)
    list(FIND SRC_UAL_KERNEL "${_anchor}" _ai)
    if(_oi GREATER -1 AND _ai GREATER -1 AND NOT _oi EQUAL _ai)
        list(REMOVE_AT SRC_UAL_KERNEL ${_oi})
        if(_oi LESS _ai)
            math(EXPR _ai "${_ai} - 1")
        endif()
        list(INSERT SRC_UAL_KERNEL ${_ai} "${_obj}")
    endif()
endmacro()
set(_R10K "${OPS_DIR}/ual/kernel")
# 1) A3 relation: weight_stream_gemv immediately before rms_norm_compat (this is what makes rotary fast)
_c38_r10_move_before("${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp" "${_R10K}/rms_norm_compat/rms_norm_compat.scpp")
# 2) flash_d128 before the msdeform span TU   3) rac-dma before flash   4) bwd-gather before flash
_c38_r10_move_before("${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp" "${_R10K}/z_ms_deform_attn_forward_span/z_ms_deform_attn_forward_span.scpp")
_c38_r10_move_before("${_R10K}/z_reshape_and_cache_dma/reshape_and_cache_dma.scpp" "${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp")
_c38_r10_move_before("${_R10K}/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp" "${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp")
# 5) bwd-gather before weight_stream_gemv => wsg 0x25a08 and rotary 0x28858
_c38_r10_move_before("${_R10K}/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp" "${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp")

# [c39 R17] joint order arm: SWAP weight_stream_gemv and the bwd-gather TU.
# Rationale (device-free search, 13 candidate orders, see _c39j17/joint-search-arms.json):
#   wsg fast addresses are all BELOW its 60a40d7 address (0x25bb4) and the gather TU is its immediate
#   predecessor, so lowering wsg necessarily displaces the gather -> "wsg fast AND gather at base" is
#   unsatisfiable.  The swap is the minimum-collateral member of that frontier:
#     wsg 0x25bb4 -> 0x23dd4 (measured fast class, -33%), gather 0x23dd4 -> 0x24d48,
#     while flash_d128 (0x2ab34), rms_norm_fp16 (0x212a0), memset (0x151fc), msdeform_fwd (0x1aa08)
#     keep their 60a40d7 addresses (unlike K3, which also moved flash_d128 by -2868).
_c38_r10_move_before("${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp" "${_R10K}/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp")

list(APPEND SRC_ALL
    ${SRC_INTERFACE_COMMON}
    ${SRC_INTERFACE_OPS}
    ${SRC_UAL_COMMON}
    ${SRC_UAL_OPS}
    ${SRC_UAL_KERNEL})

set(complie_options -O3 -msimd -mslave -fPIC -flto -x sdaa -std=c++17 --sdaa-device-lib=${OPS_DIR}/../thirdparty/teco-hal/lib/teco_hal.bc)
add_definitions(-DUSE_TECO)

add_library(ops_objs OBJECT ${SRC_ALL})
set_source_files_properties(${SRC_UAL_KERNEL} PROPERTIES LANGUAGE CXX)
set_target_properties(ops_objs PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(ops_objs PRIVATE ${complie_options})
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../common")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../thirdparty/teco-hal/include")

if(WITH_TORCH)
    set(TORCH_LINK_FLAGS "-L${TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10")
else()
    set(TORCH_LINK_FLAGS "")
endif()

if(WITH_INFERENCE_PLUGIN)
    set(TECO_OPS_PLUGIN_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops_plugin.so)
    set(TECO_INFER_PLUGIN_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libTecoInferPlugin.so)

    add_custom_target(ops_lib ALL
            COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND tecocc $<TARGET_OBJECTS:ops_objs>
            -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
            -o ${TECO_OPS_PLUGIN_LIB}
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(ops_lib ops_objs)

    add_custom_target(plugin_lib ALL
            COMMAND gcc ${PLUGIN_OBJECTS}
            -L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lteco_ops_plugin
            ${PLUGIN_LIBS}
            -shared -o ${TECO_INFER_PLUGIN_LIB}
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(plugin_lib ops_lib plugin_compile)
else()
    add_custom_target(ops_lib ALL
            COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND tecocc $<TARGET_OBJECTS:ops_objs> -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
            ${TORCH_LINK_FLAGS}
            -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops.so
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(ops_lib ops_objs)
endif()

set(CMAKE_CXX_LINK_FLAGS "-L$ENV{SDAA_HOME}/lib64  -lsdaart")
