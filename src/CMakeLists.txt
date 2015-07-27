cmake_minimum_required(VERSION 2.6)
find_package(OpenCV REQUIRED)

# Wall : print all warninga
# std=c++0x : use c++11
add_definitions(-Wall -std=c++0x)

add_executable(mcmc main.cpp)
target_link_libraries(mcmc ${OpenCV_LIBS})

add_executable(sudoku SudokuApp.cpp)
target_link_libraries(sudoku ${OpenCV_LIBS})

install(TARGETS mcmc sudoku RUNTIME DESTINATION bin)
