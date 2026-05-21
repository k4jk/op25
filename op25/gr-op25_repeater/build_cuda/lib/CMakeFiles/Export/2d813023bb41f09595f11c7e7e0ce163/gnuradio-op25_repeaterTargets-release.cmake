#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gnuradio::gnuradio-op25_repeater" for configuration "Release"
set_property(TARGET gnuradio::gnuradio-op25_repeater APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gnuradio::gnuradio-op25_repeater PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "gnuradio::gnuradio-runtime;gnuradio::gnuradio-filter"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libgnuradio-op25_repeater.so.1.0.0.0"
  IMPORTED_SONAME_RELEASE "libgnuradio-op25_repeater.so.1.0.0"
  )

list(APPEND _cmake_import_check_targets gnuradio::gnuradio-op25_repeater )
list(APPEND _cmake_import_check_files_for_gnuradio::gnuradio-op25_repeater "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libgnuradio-op25_repeater.so.1.0.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
