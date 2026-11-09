#!/usr/bin/env python3
"""Self-contained GGUF reader + quantised-weight decoders in numpy.

Exists so NITA's kernels and forward pass can be checked against an
implementation that shares no code with src/. Only the decoders the llama
family needs are implemented; every decoder follows the upstream ggml format.
"""
import numpy as np

# ---------------------------------------------------------------- GGUF
class GGUF:
    def __init__(self, path):
        import struct
        self.f = open(path, 'rb')
        r = self.r
        assert r(4) == b'GGUF'
        self.version = struct.unpack('<I', r(4))[0]
        self.n_tensors = struct.unpack('<Q', r(8))[0]
        self.n_kv = struct.unpack('<Q', r(8))[0]
        self.kv = {}
        for _ in range(self.n_kv):
            key = self.string()
            vtype = struct.unpack('<I', r(4))[0]
            self.kv[key] = self.value(vtype)
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self.string()
            n_dims = struct.unpack('<I', r(4))[0]
            dims = [struct.unpack('<Q', r(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', r(4))[0]
            offset = struct.unpack('<Q', r(8))[0]
            self.tensors[name] = (ttype, dims, offset)
        self.data_offset = (self.f.tell() + 31) // 32 * 32
        self.size = os.fstat(self.f.fileno()).st_size

    def r(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError('truncated GGUF')
        return b

    def string(self):
        (n,) = __import__('struct').unpack('<Q', self.r(8))
        return self.r(n).decode('utf-8', 'replace')

    def value(self, t):
        import struct
        scalar = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        if t in scalar:
            (v,) = struct.unpack('<'+scalar[t], self.r(struct.calcsize(scalar[t])))
            return v
        if t == 8:
            return self.string()
        if t == 9:
            (it,) = struct.unpack('<I', self.r(4)); (n,) = struct.unpack('<Q', self.r(8))
            if it == 8:
                return [self.string() for _ in range(n)]
            step = struct.calcsize('<'+scalar[it])
            self.f.seek(step*n, 1)
            return None
        raise ValueError('unknown metadata type %d' % t)

    def tensor_bytes(self, name):
        ttype, dims, offset = self.tensors[name]
        self.f.seek(self.data_offset + offset)
        nbytes = None
        return ttype, dims, self.f.read(self._need(ttype, dims))

    _BLK = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),
            10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),
            16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),
            22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),28:(1,8),29:(256,56),
            30:(1,2),34:(256,54),35:(256,66),39:(32,17),41:(128,18),42:(64,18),
            142:(128,34),143:(128,28)}

    @classmethod
    def _need(cls, ttype, dims):
        import math
        blk, bsz = cls._BLK[ttype]
        ne = 1
        for d in dims:
            ne *= d
        return (ne // blk) * bsz

import os

# ------------------------------------------------------- fp16 helper
def f16_at(b, off):
    return np.uint16(b[off] | (b[off+1] << 8)).view(np.float16).astype(np.float32)

KVALUES_IQ4NL = np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113], dtype=np.float32)

def dequant_row(ttype, buf):
    """Dequantise one weight row (dims[0] elements) to float32."""
    if ttype == 0:
        return np.frombuffer(buf, dtype='<f4').copy()
    if ttype == 1:
        return np.frombuffer(buf, dtype='<f2').astype(np.float32).copy()
    if ttype == 30:  # BF16
        u = np.frombuffer(buf, dtype='<u2').astype(np.uint32) << 16
        return u.view(np.float32).copy()
    if ttype == 28:
        return np.frombuffer(buf, dtype='<f8').astype(np.float32).copy()
    if ttype == 8:   # Q8_0
        out = np.empty((len(buf)//34)*32, dtype=np.float32)
        for i in range(0, len(out), 32):
            blk = buf[i//32*34: i//32*34+34]
            out[i:i+32] = f16_at(blk, 0) * np.frombuffer(blk[2:34], dtype=np.int8).astype(np.float32)
        return out
    if ttype == 2:   # Q4_0
        out = np.empty((len(buf)//18)*32, dtype=np.float32)
        for i in range(0, len(out), 32):
            blk = buf[i//32*18: i//32*18+18]
            d = f16_at(blk, 0)
            q = np.frombuffer(blk[2:18], dtype=np.uint8)
            out[i:i+16] = d * ((q & 0xF).astype(np.float32) - 8.0)
            out[i+16:i+32] = d * ((q >> 4).astype(np.float32) - 8.0)
        return out
    if ttype == 3:   # Q4_1
        out = np.empty((len(buf)//20)*32, dtype=np.float32)
        for i in range(0, len(out), 32):
            blk = buf[i//32*20: i//32*20+20]
            d = f16_at(blk, 0); m = f16_at(blk, 2)
            q = np.frombuffer(blk[4:20], dtype=np.uint8)
            out[i:i+16] = d * (q & 0xF).astype(np.float32) + m
            out[i+16:i+32] = d * (q >> 4).astype(np.float32) + m
        return out
    if ttype == 6:   # Q5_0
        out = np.empty((len(buf)//22)*32, dtype=np.float32)
        for i in range(0, len(out), 32):
            blk = buf[i//32*22: i//32*22+22]
            d = f16_at(blk, 0)
            qh = int.from_bytes(blk[2:6], 'little')
            q = np.frombuffer(blk[6:22], dtype=np.uint8)
            for j in range(16):
                x0 = (int(q[j]) & 0x0F) | (((qh >> (j+0)) << 4) & 0x10)
                x1 = (int(q[j]) >> 4)   | (((qh >> (j+12)))     & 0x10)
                out[i+j] = d * (x0 - 16)
                out[i+16+j] = d * (x1 - 16)
        return out
    if ttype == 7:   # Q5_1
        out = np.empty((len(buf)//24)*32, dtype=np.float32)
        for i in range(0, len(out), 32):
            blk = buf[i//32*24: i//32*24+24]
            d = f16_at(blk, 0); m = f16_at(blk, 2)
            qh = int.from_bytes(blk[4:8], 'little')
            q = np.frombuffer(blk[8:24], dtype=np.uint8)
            for j in range(16):
                x0 = (int(q[j]) & 0x0F) | (((qh >> (j+0)) << 4) & 0x10)
                x1 = (int(q[j]) >> 4)   | (((qh >> (j+12)))     & 0x10)
                out[i+j] = d * x0 + m
                out[i+16+j] = d * x1 + m
        return out
    if ttype == 12:  # Q4_K
        out = np.empty((len(buf)//144)*256, dtype=np.float32)
        def scale_min(j, sc):
            if j < 4:
                return sc[j] & 63, sc[j+4] & 63
            return ((sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4),
                    (sc[j+4] >> 4) | ((sc[j] >> 6) << 4))
        for i in range(0, len(out), 256):
            blk = buf[i//256*144: i//256*144+144]
            d = f16_at(blk, 0); dmin = f16_at(blk, 2)
            sc = blk[4:16]; q = blk[16:144]
            is_ = 0
            for c in range(4):
                sc0, m0 = scale_min(is_+0, sc)
                sc1, m1 = scale_min(is_+1, sc)
                qb = np.frombuffer(q[c*32:(c+1)*32], dtype=np.uint8)
                d1, d2 = d*sc0, d*sc1
                m1v, m2v = dmin*m0, dmin*m1
                out[i+c*64:   i+c*64+32] = d1*(qb & 0x0F).astype(np.float32) - m1v
                out[i+c*64+32:i+c*64+64] = d2*(qb >> 4).astype(np.float32) - m2v
                is_ += 2
        return out
    if ttype == 20:  # IQ4_NL
        out = np.empty((len(buf)//18)*32, dtype=np.float32)
        for i in range(0, len(out), 32):
            blk = buf[i//32*18: i//32*18+18]
            d = f16_at(blk, 0)
            q = np.frombuffer(blk[2:18], dtype=np.uint8)
            out[i:i+16] = d * KVALUES_IQ4NL[q & 0x0F]
            out[i+16:i+32] = d * KVALUES_IQ4NL[q >> 4]
        return out
    raise ValueError('numpy decoder missing for ggml type %d' % ttype)

def load_matrix(g, name):
    """Return the tensor as a 2-D float32 array shaped [ne1, ne0]."""
    ttype, dims, raw = g.tensor_bytes(name)
    ne0 = dims[0] if dims else 1
    ne1 = 1
    for d in dims[1:]:
        ne1 *= d
    row_elems = ne0
    row_bytes = len(raw)//ne1
    out = np.empty((ne1, ne0), dtype=np.float32)
    for r in range(ne1):
        out[r] = dequant_row(ttype, raw[r*row_bytes:(r+1)*row_bytes])
    return out
