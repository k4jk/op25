#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir=/home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/james/ham/dev/op25/op25/gr-op25_repeater/build_cuda2/python/op25_repeater":"$PATH"
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/james/ham/dev/op25/op25/gr-op25_repeater/build_cuda2/test_modules:$PYTHONPATH
/usr/bin/python3 /home/james/ham/dev/op25/op25/gr-op25_repeater/python/op25_repeater/qa_gardner_costas_cc.py 
