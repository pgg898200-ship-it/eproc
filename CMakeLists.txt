cmake_minimum_required(VERSION 3.16)

project(eproc
    VERSION 0.1.0
    DESCRIPTION "Sub-microsecond anytime-valid e-processes, multi-rate conformal calibration, Byzantine-safe fusion, and causal subsampling"
    HOMEPAGE_URL "https://github.com/pgg898200-ship-it/eproc"
    LANGUAGES C
)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(EPROC_IS_TOP_LEVEL ON)
else()
    set(EPROC_IS_TOP_LEVEL OFF)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_SHARED_LIBS "Build eproc as a shared library instead of static" OFF)
option(EPROC_BUILD_TESTS "Build the C11 core test suite" ${EPROC_IS_TOP_LEVEL})
option(EPROC_INSTALL "Generate install targets for eproc" ${EPROC_IS_TOP_LEVEL})
option(EPROC_NATIVE_ARCH "Enable -march=native for static AVX2/FMA codegen" OFF)
option(EPROC_FORCE_SCALAR "Disable SIMD kernels and force pure scalar execution" OFF)

set(EPROC_CORE_SOURCES
    csrc/eproc_eprocess.c
    csrc/eproc_conformal.c
    csrc/eproc_fusion.c
    csrc/eproc_subsample.c
)

add_library(eproc ${EPROC_CORE_SOURCES})
add_library(eproc::eproc ALIAS eproc)

target_compile_features(eproc PUBLIC c_std_11)
set_target_properties(eproc PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(eproc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(eproc PRIVATE -ffp-contract=off -Wall -Wextra)
    if(EPROC_NATIVE_ARCH)
        target_compile_options(eproc PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(eproc PRIVATE /fp:precise /W3)
endif()

if(EPROC_FORCE_SCALAR)
    target_compile_definitions(eproc PRIVATE EPROC_FORCE_SCALAR)
endif()

set(EPROC_MATH_LIBS "")
if(NOT WIN32)
    find_library(EPROC_MATH_LIBRARY m)
    if(EPROC_MATH_LIBRARY)
        set(EPROC_MATH_LIBS m)
        target_link_libraries(eproc PUBLIC m)
    endif()
endif()

# Single-header stb-style INTERFACE target (define EPROC_IMPLEMENTATION in one TU)
add_library(eproc_single_header INTERFACE)
add_library(eproc::single_header ALIAS eproc_single_header)

target_compile_features(eproc_single_header INTERFACE c_std_11)
target_include_directories(eproc_single_header
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dist>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(EPROC_MATH_LIBS)
    target_link_libraries(eproc_single_header INTERFACE ${EPROC_MATH_LIBS})
endif()

if(EPROC_BUILD_TESTS)
    enable_testing()
    find_package(Threads REQUIRED)

    add_executable(test_c_core csrc/test_c_core.c)
    target_link_libraries(test_c_core PRIVATE eproc::eproc Threads::Threads)
    set_target_properties(test_c_core PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(test_c_core PRIVATE -ffp-contract=off -Wall -Wextra)
    elseif(MSVC)
        target_compile_options(test_c_core PRIVATE /fp:precise /W3)
    endif()
    if(EPROC_FORCE_SCALAR)
        target_compile_definitions(test_c_core PRIVATE EPROC_FORCE_SCALAR)
    endif()

    add_test(NAME eproc_c_core COMMAND test_c_core)
endif()

if(EPROC_INSTALL)
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(EPROC_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(EPROC_PC_LIBDIR "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif()

    if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(EPROC_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
    else()
        set(EPROC_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif()

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/eproc.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/eproc.pc"
        @ONLY
    )

    install(TARGETS eproc eproc_single_header
        EXPORT eprocTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/csrc/eproc.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dist/eproc_single.h")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/dist/eproc_single.h"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
    endif()

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/eproc.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    )

    install(EXPORT eprocTargets
        FILE eprocTargets.cmake
        NAMESPACE eproc::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/eproc"
    )

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/eprocConfig.cmake"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/eprocTargets.cmake\")\n"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/eprocConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/eprocConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/eprocConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/eproc"
    )
endif()
