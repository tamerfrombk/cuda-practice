cmake_minimum_required(VERSION 4.0)
project(cuda-practice LANGUAGES NONE)

find_program(NVCC12_EXECUTABLE nvcc12 HINTS "$ENV{HOME}/.local/bin" REQUIRED)

set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${BIN_DIR})

file(GLOB CU_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cu")

foreach(cu_src ${CU_SOURCES})
    get_filename_component(name ${cu_src} NAME_WE)
    set(exe ${BIN_DIR}/${name})

    add_custom_command(
        OUTPUT ${exe}
        COMMAND ${NVCC12_EXECUTABLE} ${cu_src} -o ${exe}
        DEPENDS ${cu_src}
        COMMENT "Building ${name} with nvcc12"
        VERBATIM
    )

    add_custom_target(run-${name}
        COMMAND ${exe}
        DEPENDS ${exe}
        COMMENT "Running ${name}"
        VERBATIM
    )
endforeach()
