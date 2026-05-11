
# P25 TDMA Decoder (C) Copyright 2013 KA1RBI
# Error correction Copyright 2026 James Kirkham, K4JK    
#
# This file is part of OP25
# 
# OP25 is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# OP25 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with OP25; see the file COPYING. If not, write to the Free
# Software Foundation, Inc., 51 Franklin Street, Boston, MA
# 02110-1301, USA.

import numpy as np
from bit_utils import *

def extract_duid(b):
    duid0 = b[10]   # duid 3,2
    duid1 = b[47]   # duid 1,0
    duid2 = b[132]  # par 3,2
    duid3 = b[169]  # par 1,0
    v = (duid0 << 6) + (duid1 << 4) + (duid2 << 2) + duid3
    va = mk_array(v, 8)
    return mk_str(va)

def mk_duid_lookup():
    duid_map = {}
    codeword_ints = []  # (int_value, duid_index) for Hamming-distance ECC
    g = np.array(np.asmatrix('1 0 0 0 1 1 0 1; 0 1 0 0 1 0 1 1; 0 0 1 0 1 1 1 0; 0 0 0 1 0 1 1 1'))
    for i in range(16):
        codeword = mk_str(np.dot(mk_array(i, 4), g))
        duid_map[codeword] = i
        codeword_ints.append((int(codeword, 2), i))
    return duid_map, codeword_ints

class p25p2_duid(object):
    def __init__(self):
        self.duid_str = {}
        self.duid_str[0] = "4v"
        self.duid_str[3] = "sacch w"
        self.duid_str[6] = "2v"
        self.duid_str[9] = "facch w"
        self.duid_str[12] = "sacch w/o"
        self.duid_str[15] = "facch w/o"

        self.duid_map, self.duid_codewords = mk_duid_lookup()

    def decode_duid(self, burst):
        cw_str = extract_duid(burst)
        duid_val = self.duid_map.get(cw_str)
        if duid_val is None:
            # [8,4] code d_min=4: correct up to 1-bit errors
            cw_int = int(cw_str, 2)
            best_dist, best_val = 9, None
            for valid_int, val in self.duid_codewords:
                dist = bin(cw_int ^ valid_int).count('1')
                if dist < best_dist:
                    best_dist, best_val = dist, val
            if best_dist <= 1:
                duid_val = best_val
        if duid_val is not None:
            return self.duid_str.get(duid_val, 'unknown' + cw_str)
        return 'unknown' + cw_str
