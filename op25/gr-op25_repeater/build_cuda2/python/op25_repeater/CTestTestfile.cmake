# CMake generated Testfile for 
# Source directory: /home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater
# Build directory: /home/james/ham/dev/op25/op25/gr-op25_repeater/build_cuda2/python/op25_repeater
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(qa_vocoder "/usr/bin/sh" "qa_vocoder_test.sh")
set_tests_properties(qa_vocoder PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;68;GR_ADD_TEST;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;0;")
add_test(qa_gardner_costas_cc "/usr/bin/sh" "qa_gardner_costas_cc_test.sh")
set_tests_properties(qa_gardner_costas_cc PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;69;GR_ADD_TEST;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;0;")
add_test(qa_p25_frame_assembler "/usr/bin/sh" "qa_p25_frame_assembler_test.sh")
set_tests_properties(qa_p25_frame_assembler PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;70;GR_ADD_TEST;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;0;")
add_test(qa_fsk4_slicer_fb "/usr/bin/sh" "qa_fsk4_slicer_fb_test.sh")
set_tests_properties(qa_fsk4_slicer_fb PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;119;add_test;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;71;GR_ADD_TEST;/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/CMakeLists.txt;0;")
subdirs("bindings")
