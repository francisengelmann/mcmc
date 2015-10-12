cmake_minimum_required(VERSION 2.6)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Wall : print all warnings
# std=c++0x : use c++11
add_definitions(-Wall -std=c++0x)

add_executable(simple simple.cpp)
target_link_libraries(simple)

add_executable(sudoku sudoku.cpp)
target_link_libraries(sudoku ${OpenCV_LIBS})

install(TARGETS simple sudoku RUNTIME DESTINATION bin)
