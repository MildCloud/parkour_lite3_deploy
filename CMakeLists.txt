cmake_minimum_required(VERSION 3.10)

project(rl_controller)

set(CMAKE_BUILD_TYPE Release)
add_definitions(-w) # warning ignore
add_compile_options(-fPIC)
set(BUILD_PLATFORM "x86" CACHE STRING "select build cpu type")
option(BUILD_SIM off)
if (BUILD_PLATFORM STREQUAL arm)
  message("this is arm platform")
  set(BUILD_SIM off)
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++") 
else()
  message("this is x86 platform")
endif()
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

get_filename_component(WORKSPACE_DIR ./ ABSOLUTE)
set(THIRD_PARTY ${WORKSPACE_DIR}/third_party)


include_directories(
  ${THIRD_PARTY}/eigen/
  ${THIRD_PARTY}/gamepad/include
  ${THIRD_PARTY}/x30_motion_sdk/include
  ${THIRD_PARTY}/Lite3_MotionSDK/include
  types
  interface/robot
  interface/user_command
  state_machine
  run_policy
  ${WORKSPACE_DIR}/depth_sensor
  ${TORCH_INCLUDE_DIRS}
)

add_subdirectory(interface)
# add_subdirectory(run_policy)

if (BUILD_SIM)
    add_definitions(-DUSE_RAISIM)
    set(raisim_DIR "~/raisim_workspace/raisim_build/lib/cmake/raisim")
    find_package(raisim CONFIG REQUIRED)
    # add_library(simulation STATIC ${SIMULATOIN_SRC})
    # target_link_libraries(simulation PUBLIC raisim::raisim)
endif()

set(Torch_DIR "${THIRD_PARTY}/libtorch/${BUILD_PLATFORM}/share/cmake/Torch")
find_package(Torch REQUIRED)
message("Torch_DIR:  ${Torch_DIR}")

# include_directories(
#     ./
#     ${TORCH_INCLUDE_DIRS}
# )
file(GLOB_RECURSE RUN_POLICY_SRC "./run_policy/*.cpp")
file(GLOB_RECURSE SENSOR_SRC "depth_sensor/*.cpp")


# add_library(policy_runner STATIC ${RUN_POLICY_SRC})
# target_link_libraries(policy_runner PUBLIC -lpthread -lm -lrt -ldl -lstdc++ torch)

file(GLOB_RECURSE STATE_MACHINE_SRC "state_machine/*.c*")

add_executable(main main.cpp ${STATE_MACHINE_SRC} ${RUN_POLICY_SRC} ${SENSOR_SRC})
target_link_libraries(main interface -lpthread -lm -lrt -ldl -lstdc++ torch)

if (BUILD_SIM)
    message("build simulation")
    target_link_libraries(main raisim::raisim)
endif()

# file(GLOB_RECURSE SRC "src/*.cpp")

# add_library(test STATIC ${JUEYING_SDK_SRC})
# target_link_libraries(jueying_sdk PUBLIC -lpthread -lm -lrt -ldl -lstdc++ )
# cmake .. -DBUILD_SIM=ON -DCMAKE_PREFIX_PATH=/home/tianshu/raisimLib-master/raisim/linux