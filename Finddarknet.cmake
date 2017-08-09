find_path(darknet_ROOT_DIR
        NAMES include/darknet.h
        HINTS darknet
        )

find_library(darknet_LIBRARIES
        NAMES darknet
        HINTS ${CMAKE_BINARY_DIR}
        )

find_path(darknet_INCLUDE_DIRS
        NAMES darknet.h
        HINTS ${darknet_ROOT_DIR}/include
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(darknet DEFAULT_MSG
        darknet_LIBRARIES
        darknet_INCLUDE_DIRS
        )

mark_as_advanced(
        darknet_ROOT_DIR
        darknet_LIBRARIES
        darknet_INCLUDE_DIRS
)
