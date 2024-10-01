if(TARGET polyscope)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    polyscope
    SYSTEM
    GIT_REPOSITORY https://github.com/rjc8237/polyscope.git
)
FetchContent_MakeAvailable(polyscope)
