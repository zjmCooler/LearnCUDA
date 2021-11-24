add_executable(1_hello_world 1_hello_world.cu)
set_target_properties(1_hello_world PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(2_device_info 2_device_info.cu)
set_target_properties(2_device_info PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(3_sum_arrays 3_sum_arrays.cu)
set_target_properties(3_sum_arrays PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(3_sum_arrays
                            PRIVATE
                            ${PROJECT_SOURCE_DIR}/include)