#[==[
# FindLibpcap.cmake
#
# Locates the system libpcap library and its headers.
#
# Imported targets:
#   Libpcap::Libpcap      — if found, a full imported library target
#
# Result variables:
#   LIBPCAP_FOUND         — TRUE if libpcap was found
#   LIBPCAP_INCLUDE_DIR   — directory containing pcap/pcap.h
#   LIBPCAP_LIBRARIES     — library to link against
#   LIBPCAP_VERSION       — version string (if obtainable)
#
# Usage in CMakeLists.txt:
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#   find_package(Libpcap REQUIRED)
#   target_link_libraries(mytarget PRIVATE ${LIBPCAP_LIBRARIES})
#   target_include_directories(mytarget PRIVATE ${LIBPCAP_INCLUDE_DIR})
#]==]

# ── Try pkg-config first ──────────────────────────────────────────────────
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_LIBPCAP QUIET libpcap)
endif()

# ── Find header ───────────────────────────────────────────────────────────
find_path(LIBPCAP_INCLUDE_DIR
    NAMES pcap/pcap.h pcap.h
    HINTS
        ${PC_LIBPCAP_INCLUDE_DIRS}
        /usr/include
        /usr/local/include
        /opt/homebrew/include
    DOC "libpcap include directory"
)

# ── Find library ──────────────────────────────────────────────────────────
find_library(LIBPCAP_LIBRARIES
    NAMES pcap wpcap
    HINTS
        ${PC_LIBPCAP_LIBRARY_DIRS}
        /usr/lib
        /usr/local/lib
        /opt/homebrew/lib
    DOC "libpcap library"
)

# ── Extract version from pcap.h if available ─────────────────────────────
if(LIBPCAP_INCLUDE_DIR AND EXISTS "${LIBPCAP_INCLUDE_DIR}/pcap/pcap.h")
    file(STRINGS "${LIBPCAP_INCLUDE_DIR}/pcap/pcap.h" _pcap_version_line
         REGEX "^#define[ \t]+PCAP_VERSION_MAJOR[ \t]+[0-9]+")
    if(_pcap_version_line)
        string(REGEX REPLACE ".*PCAP_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
               _pcap_ver_major "${_pcap_version_line}")
        set(LIBPCAP_VERSION "${_pcap_ver_major}.x")
    endif()
endif()
if(NOT LIBPCAP_VERSION AND PC_LIBPCAP_VERSION)
    set(LIBPCAP_VERSION "${PC_LIBPCAP_VERSION}")
endif()

# ── Standard find_package result handling ────────────────────────────────
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libpcap
    REQUIRED_VARS LIBPCAP_LIBRARIES LIBPCAP_INCLUDE_DIR
    VERSION_VAR   LIBPCAP_VERSION
)

# ── Create imported target ────────────────────────────────────────────────
if(LIBPCAP_FOUND AND NOT TARGET Libpcap::Libpcap)
    add_library(Libpcap::Libpcap UNKNOWN IMPORTED)
    set_target_properties(Libpcap::Libpcap PROPERTIES
        IMPORTED_LOCATION             "${LIBPCAP_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBPCAP_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(LIBPCAP_INCLUDE_DIR LIBPCAP_LIBRARIES)
