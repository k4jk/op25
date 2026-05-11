
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

def mk_isch(v):
    v1 = v & 3
    v  = v >> 2
    v2 = v & 1
    v  = v >> 1
    v3 = v & 3
    v  = v >> 2
    v4 = v & 3
    v  = v >> 2
    v5 = v & 3
    return v4, v3, v2, v1

class p25p2_isch(object):
    def __init__(self):
        self.isch_map, self.isch_codewords, self.isch_max_errors = self.mk_isch_lookup()

    def mk_isch_lookup(self):
        isch_map = {}
        codeword_ints = []  # (int_value, isch_index) for Hamming-distance ECC
        g = np.array(np.asmatrix('1 0 0 0 1 0 0 0 0 0 0 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1; 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 0 1 0 0; 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 0 1 0 1 1 0 0 0; 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 0 1 1 0 1 0 0 0 1 1 0 0 0 1 1 1 0; 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1; 0 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 1 1 0 1 1 0 1 1 1 0 0 1 0; 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 1 1 0 1 0 0 0 1 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 0 1; 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 1 1 0 0 1 0 1 1 1 0 1 0 1 0 1 0 0 1 0 0 1 1 1 0; 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 1 0 1 1 0 0 1 0 1 1 1'))
        c0 = 0x184229d461
        for i in range(0, 2**7):
            codeword = mk_int(np.dot(mk_array(i, 9), g)) ^ c0
            isch_map['%x' % codeword] = i
            codeword_ints.append((codeword, i))
        # compute minimum Hamming distance to determine error correction capacity
        min_dist = 40
        for j in range(len(codeword_ints)):
            for k in range(j + 1, len(codeword_ints)):
                d = bin(codeword_ints[j][0] ^ codeword_ints[k][0]).count('1')
                if d < min_dist:
                    min_dist = d
        max_errors = (min_dist - 1) // 2
        return isch_map, codeword_ints, max_errors

    def decode_isch(self, syms):
        sync0 = 0x575d57f7ff
        v = mk_int(dibits_to_bits(syms))
        vp = '%x' % v
        if v == sync0:
            return -2, -2, -2, -2
        if vp in self.isch_map:
            chn, loc, fr, cnt = mk_isch(self.isch_map[vp])
            return chn, loc, fr, cnt
        # find closest valid codeword via Hamming distance
        best_dist, best_val = 41, None
        for valid_int, val in self.isch_codewords:
            dist = bin(v ^ valid_int).count('1')
            if dist < best_dist:
                best_dist, best_val = dist, val
        if best_dist <= self.isch_max_errors:
            chn, loc, fr, cnt = mk_isch(best_val)
            return chn, loc, fr, cnt
        return -1, -1, -1, -1
