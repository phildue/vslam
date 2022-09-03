# Copyright 2022 Philipp.Duernay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


function(pd_setup_lib name version sources headers namespace)

    # Construct library from sources
  add_library( ${name}
    ${sources}
    ${headers} )
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_features(${name} PUBLIC cxx_std_17)

    # Configure alias so there is no difference whether we link from source/from already built
  add_library(${namespace}::${name} ALIAS ${name})

    # Set include path (can be different in build/install)
  target_include_directories( ${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
    $<INSTALL_INTERFACE:include/vslam/${name}> )
  target_include_directories( ${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>
    $<INSTALL_INTERFACE:include/vslam> )

  install( DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/
    DESTINATION include/vslam/${name}
    FILES_MATCHING # install only matched files
    PATTERN "*.h" )# select header files

  install( DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/
    DESTINATION include/vslam/${name}
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" )# select header files

  install( DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/vslam/
    FILES_MATCHING # install only matched files
    PATTERN "*.h" )# select header files



endfunction()


macro(pd_add_test unit lib)
    add_executable( ${unit}Test test/test_${unit}.cpp )
    target_compile_features(${unit}Test PUBLIC cxx_std_17)

    target_link_libraries( ${unit}Test PRIVATE pd::${lib} GTest::gtest_main )
    add_test( NAME ${unit}.UnitTest COMMAND ${unit}Test )
    set_property(TARGET ${unit}Test PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    set(ExtraMacroArgs ${ARGN})
    list(LENGTH ExtraMacroArgs NumExtraMacroArgs)
    if(NumExtraMacroArgs GREATER 0)
        foreach(ExtraArg ${ExtraMacroArgs})
            target_compile_definitions(${unit}Test PUBLIC ${ExtraArg})
        endforeach()
    endif()
endmacro()

