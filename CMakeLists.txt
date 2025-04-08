cmake_minimum_required(VERSION 3.15)
include(${CMAKE_CURRENT_SOURCE_DIR}/Resources/pd.cmake/pd.cmake)
project(mlperceptron-object)



#╭──────────────────────────────────────╮
#│             Pd External              │
#╰──────────────────────────────────────╯
# Get all cpp files inside ${CMAKE_SOURCE_DIR}/Sources/ and add them to the project
file(GLOB EXTERNAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Sources/*.c")
add_pd_external(mlperceptron-object mlperceptron ${EXTERNAL_SOURCES})
target_link_libraries(mlperceptron-object gsl)
target_include_directories(mlperceptron-object PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/gsl/")

if(${MSVC})
    set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS "/wd4091 /wd4996")
endif()


target_compile_options(mlperceptron-object PUBLIC
                "-O3" 
                "-DPD"
)
