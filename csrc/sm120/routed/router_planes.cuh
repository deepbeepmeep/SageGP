// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// Every thread moves one contiguous 16-byte vector from a plane of a 64-key tile.
template<int mode, class Shared>
__device__ __forceinline__ void load_planes(const int8_t* keys, Shared& sm, unsigned head, unsigned n, unsigned tile) {
    unsigned t = threadIdx.y * 32 + threadIdx.x;
    unsigned row = t >> 1, half = t & 1;
    const int8_t* base = keys + (size_t(head) * ((n+63)/64) + tile) * (176*64);
    constexpr int parts = mode == 0 ? 1 : mode == 1 ? 2 : mode == 2 ? 3 : 4;
#pragma unroll
    for (int part=0; part<parts; ++part) {
        int plane = mode==2 && part==2 ? 4 : part;
        sm.load_128b_async(sm.get_permuted_offset(row,2*part+half),base+plane*2048+t*16);
    }
    if constexpr(mode==2) {
        if(t<64) sm.load_128b_async(sm.get_permuted_offset(t,6),base+160*64+t*16);
    }
}

