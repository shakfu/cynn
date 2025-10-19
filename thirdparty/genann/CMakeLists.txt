project(genann_project C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

add_library(genann
	STATIC
	genann.c
)

target_compile_options(genann
	PUBLIC
	-Wall
	-Wextra
	-pedantic
	-O3
	-ffast-math
	-flto
	-march=native
	-fPIC
)
