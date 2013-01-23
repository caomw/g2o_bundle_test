# project
cmake_minimum_required (VERSION 2.6)
project (g2o_bundle_test)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

#FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
FIND_PACKAGE(Cholmod REQUIRED)

IF(G2O_FOUND)
	MESSAGE(STATUS "G2O solver found")
ENDIF()

# directories
set (SRC_DIR src)

if(APPLE)
# This defines specific definition for Apple MacOSX platform
	add_definitions(-DAPPLE=1)
endif()

if(APPLE)
#Define your specific definition for Apple MacOSX platform as below examples.
#SET(LIBLEVMAR_2_6_DIR /Users/hyon/Codes/levmar-2.6)
#SET(LIBFUNDEST_1_0_DIR /Users/hyon/Codes/fundest-1.0)
else()
#Define your specific definition for other platform as below examples.
#SET(LIBLEVMAR_2_6_DIR /home/hyon/Codes/levmar-2.6)
#SET(LIBFUNDEST_1_0_DIR /home/hyon/Codes/fundest-1.0)
endif()

SET(LIB_DBOW2 /usr/local/include/DBoW2)


# 3rd party libraries
if(APPLE)
SET(LIB_NAMES amd colamd f2c lapack blas g2o_types_sba Sophus)
else()
SET(LIB_NAMES lapack blas f2c g2o_types_sba Sophus)
endif()

FOREACH(lib ${LIB_NAMES})
  FIND_LIBRARY(LIB_${lib} ${lib})
  IF (LIB_${lib})
    MESSAGE(STATUS "found library '${lib}': ${LIB_${lib}}")
  ELSE (LIB_${lib})
    MESSAGE(FATAL_ERROR "cannot find library '${lib}' - this will not work ...")
  ENDIF (LIB_${lib})
  LIST(APPEND LIBS ${LIB_${lib}})
ENDFOREACH(lib)

# include directory
include_directories(${SRC_DIR} ${EIGEN3_INCLUDE_DIR} ${G2O_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB SRC_FILES "${SRC_DIR}/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_VERBOSE_MAKEFILE yes)

# demo program
add_executable(main "${SRC_DIR}/main.cpp")
target_link_libraries (main ${LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${CHOLMOD_LIBRARY})
add_executable(ba_demo "${SRC_DIR}/ba_demo.cpp")
target_link_libraries (ba_demo ${LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${CHOLMOD_LIBRARY})
