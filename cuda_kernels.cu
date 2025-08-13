// CUDA port of OpenCL kernels in keccak.cl and profanity.cl
// Focus: identical math, dynamic inverseSize (<= 255), multi-GPU friendly

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// Shared struct layouts must match host types.hpp
typedef uint32_t mp_word;
struct mp_number { mp_word d[8]; };
struct point { mp_number x; mp_number y; };
struct result { uint32_t found; uint32_t foundId; uint8_t foundHash[20]; };

#define MP_WORDS 8
#define PROFANITY_MAX_INVERSE_SIZE 255
#define PROFANITY_MAX_SCORE 40

// ----------------- utils -----------------
__device__ __forceinline__ uint32_t rotl32(uint32_t x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ uint64_t rotl64(uint64_t x, unsigned int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ __forceinline__ uint32_t bswap32(uint32_t n) {
    return ((n & 0x000000FFu) << 24) |
           ((n & 0x0000FF00u) << 8)  |
           ((n & 0x00FF0000u) >> 8)  |
           ((n & 0xFF000000u) >> 24);
}

__device__ __forceinline__ uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
    return __umulhi(a, b);
}

// ----------------- keccak -----------------
typedef union {
    uint8_t  b[200];
    uint64_t q[25];
    uint32_t d[50];
} ethhash;

__device__ __constant__ uint64_t keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

#define TH_ELT(t, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4) \
    t = rotl64((uint64_t)(d0 ^ d1 ^ d2 ^ d3 ^ d4), 1u) ^ (uint64_t)(c0 ^ c1 ^ c2 ^ c3 ^ c4);

#define THETA(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
    uint64_t t0, t1, t2, t3, t4; \
    TH_ELT(t0, s40, s41, s42, s43, s44, s10, s11, s12, s13, s14); \
    TH_ELT(t1, s00, s01, s02, s03, s04, s20, s21, s22, s23, s24); \
    TH_ELT(t2, s10, s11, s12, s13, s14, s30, s31, s32, s33, s34); \
    TH_ELT(t3, s20, s21, s22, s23, s24, s40, s41, s42, s43, s44); \
    TH_ELT(t4, s30, s31, s32, s33, s34, s00, s01, s02, s03, s04); \
    s00 ^= t0; s01 ^= t0; s02 ^= t0; s03 ^= t0; s04 ^= t0; \
    s10 ^= t1; s11 ^= t1; s12 ^= t1; s13 ^= t1; s14 ^= t1; \
    s20 ^= t2; s21 ^= t2; s22 ^= t2; s23 ^= t2; s24 ^= t2; \
    s30 ^= t3; s31 ^= t3; s32 ^= t3; s33 ^= t3; s34 ^= t3; \
    s40 ^= t4; s41 ^= t4; s42 ^= t4; s43 ^= t4; s44 ^= t4; \
}

#define RHOPI(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
    uint64_t t0; \
    t0  = rotl64(s10,  1u);  \
    s10 = rotl64(s11, 44u); \
    s11 = rotl64(s41, 20u); \
    s41 = rotl64(s24, 61u); \
    s24 = rotl64(s42, 39u); \
    s42 = rotl64(s04, 18u); \
    s04 = rotl64(s20, 62u); \
    s20 = rotl64(s22, 43u); \
    s22 = rotl64(s32, 25u); \
    s32 = rotl64(s43,  8u); \
    s43 = rotl64(s34, 56u); \
    s34 = rotl64(s03, 41u); \
    s03 = rotl64(s40, 27u); \
    s40 = rotl64(s44, 14u); \
    s44 = rotl64(s14,  2u); \
    s14 = rotl64(s31, 55u); \
    s31 = rotl64(s13, 45u); \
    s13 = rotl64(s01, 36u); \
    s01 = rotl64(s30, 28u); \
    s30 = rotl64(s33, 21u); \
    s33 = rotl64(s23, 15u); \
    s23 = rotl64(s12, 10u); \
    s12 = rotl64(s21,  6u); \
    s21 = rotl64(s02,  3u); \
    s02 = t0; \
}

#define KHI(s00, s01, s02, s03, s04, \
            s10, s11, s12, s13, s14, \
            s20, s21, s22, s23, s24, \
            s30, s31, s32, s33, s34, \
            s40, s41, s42, s43, s44) \
{ \
    uint64_t t0, t1, t2, t3, t4; \
    t0 = s00 ^ (~s10 &  s20); \
    t1 = s10 ^ (~s20 &  s30); \
    t2 = s20 ^ (~s30 &  s40); \
    t3 = s30 ^ (~s40 &  s00); \
    t4 = s40 ^ (~s00 &  s10); \
    s00 = t0; s10 = t1; s20 = t2; s30 = t3; s40 = t4; \
    t0 = s01 ^ (~s11 &  s21); \
    t1 = s11 ^ (~s21 &  s31); \
    t2 = s21 ^ (~s31 &  s41); \
    t3 = s31 ^ (~s41 &  s01); \
    t4 = s41 ^ (~s01 &  s11); \
    s01 = t0; s11 = t1; s21 = t2; s31 = t3; s41 = t4; \
    t0 = s02 ^ (~s12 &  s22); \
    t1 = s12 ^ (~s22 &  s32); \
    t2 = s22 ^ (~s32 &  s42); \
    t3 = s32 ^ (~s42 &  s02); \
    t4 = s42 ^ (~s02 &  s12); \
    s02 = t0; s12 = t1; s22 = t2; s32 = t3; s42 = t4; \
    t0 = s03 ^ (~s13 &  s23); \
    t1 = s13 ^ (~s23 &  s33); \
    t2 = s23 ^ (~s33 &  s43); \
    t3 = s33 ^ (~s43 &  s03); \
    t4 = s43 ^ (~s03 &  s13); \
    s03 = t0; s13 = t1; s23 = t2; s33 = t3; s43 = t4; \
    t0 = s04 ^ (~s14 &  s24); \
    t1 = s14 ^ (~s24 &  s34); \
    t2 = s24 ^ (~s34 &  s44); \
    t3 = s34 ^ (~s44 &  s04); \
    t4 = s44 ^ (~s04 &  s14); \
    s04 = t0; s14 = t1; s24 = t2; s34 = t3; s44 = t4; \
}

#define IOTA(s00, r) { s00 ^= r; }

__device__ inline void sha3_keccakf(ethhash *h)
{
    uint64_t *st = h->q;
    h->d[33] ^= 0x80000000u;
    for (int i = 0; i < 24; ++i) {
        THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
        RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
        KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
        IOTA(st[0], keccakf_rndc[i]);
    }
}

// ----------------- mp math constants -----------------
__device__ __constant__ mp_number const_mod = { {0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU} };
__device__ __constant__ mp_number const_tripleNegativeGx = { {0xbb17b196U, 0xf2287becU, 0x76958573U, 0xf82c096eU, 0x946adeeaU, 0xff1ed83eU, 0x1269ccfaU, 0x92c4cc83U } };
__device__ __constant__ mp_number const_doubleNegativeGy = { {0x09de52bfU, 0xc7705edfU, 0xb2f557ccU, 0x05d0976eU, 0xe3ddeeaeU, 0x44b60807U, 0xb2b87735U, 0x6f8a4b11U} };
__device__ __constant__ mp_number const_negativeGy       = { {0x04ef2777U, 0x63b82f6fU, 0x597aabe6U, 0x02e84bb7U, 0xf1eef757U, 0xa25b0403U, 0xd95c3b9aU, 0xb7c52588U } };
__device__ __constant__ mp_number const_negativeGx       = { {0xe907e497U, 0xa60d7ea3U, 0xd231d726U, 0xfd640324U, 0x3178f4f8U, 0xaa5f9d6aU, 0x06234453U, 0x86419981U } };

// ----------------- mp math -----------------
__device__ __forceinline__ mp_word mp_sub(mp_number *r, const mp_number *a, const mp_number *b) {
    mp_word t, c = 0;
    for (mp_word i = 0; i < MP_WORDS; ++i) { t = a->d[i] - b->d[i] - c; c = t > a->d[i] ? 1 : (t == a->d[i] ? c : 0); r->d[i] = t; } return c; }

__device__ __forceinline__ mp_word mp_sub_mod(mp_number *r) {
    mp_number mod = const_mod; mp_word t, c = 0;
    for (mp_word i = 0; i < MP_WORDS; ++i) { t = r->d[i] - mod.d[i] - c; c = t > r->d[i] ? 1 : (t == r->d[i] ? c : 0); r->d[i] = t; } return c; }

__device__ __forceinline__ void mp_mod_sub(mp_number *r, const mp_number *a, const mp_number *b) {
    mp_word i, t, c = 0;
    for (i = 0; i < MP_WORDS; ++i) { t = a->d[i] - b->d[i] - c; c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1); r->d[i] = t; }
    if (c) { c = 0; mp_number mod = const_mod; for (i = 0; i < MP_WORDS; ++i) { r->d[i] += mod.d[i] + c; c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0); } }
}

__device__ __forceinline__ void mp_mod_sub_const(mp_number *r, const mp_number *a, const mp_number *b) {
    mp_word i, t, c = 0;
    for (i = 0; i < MP_WORDS; ++i) { t = a->d[i] - b->d[i] - c; c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1); r->d[i] = t; }
    if (c) { c = 0; mp_number mod = const_mod; for (i = 0; i < MP_WORDS; ++i) { r->d[i] += mod.d[i] + c; c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0); } }
}

__device__ __forceinline__ void mp_mod_sub_gx(mp_number *r, const mp_number *a) {
    mp_word i, t, c = 0; const mp_number gx = { {0x16f81798U,0x59f2815bU,0x2dce28d9U,0x029bfcdbU,0xce870b07U,0x55a06295U,0xf9dcbbacU,0x79be667eU} };
    for (i = 0; i < MP_WORDS; ++i) { t = a->d[i] - gx.d[i] - c; c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1); r->d[i] = t; }
    if (c) { c = 0; mp_number mod = const_mod; for (i = 0; i < MP_WORDS; ++i) { r->d[i] += mod.d[i] + c; c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0); } }
}

__device__ __forceinline__ void mp_mod_sub_gy(mp_number *r, const mp_number *a) {
    mp_word i, t, c = 0; const mp_number gy = { {0xfb10d4b8U,0x9c47d08fU,0xa6855419U,0xfd17b448U,0x0e1108a8U,0x5da4fbfcU,0x26a3c465U,0x483ada77U} };
    for (i = 0; i < MP_WORDS; ++i) { t = a->d[i] - gy.d[i] - c; c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1); r->d[i] = t; }
    if (c) { c = 0; mp_number mod = const_mod; for (i = 0; i < MP_WORDS; ++i) { r->d[i] += mod.d[i] + c; c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0); } }
}

__device__ __forceinline__ mp_word mp_add(mp_number *r, const mp_number *a) {
    mp_word c = 0; for (mp_word i = 0; i < MP_WORDS; ++i) { uint32_t t = r->d[i] + a->d[i]; uint32_t t2 = t + c; c = (t2 < t) ? 1 : (t2 == t ? c : 0); r->d[i] = t2; } return c; }

__device__ __forceinline__ mp_word mp_add_mod(mp_number *r) {
    mp_number mod = const_mod; mp_word c = 0; for (mp_word i = 0; i < MP_WORDS; ++i) { uint32_t t = r->d[i] + mod.d[i]; uint32_t t2 = t + c; c = (t2 < t) ? 1 : (t2 == t ? c : 0); r->d[i] = t2; } return c; }

__device__ __forceinline__ mp_word mp_add_more(mp_number *r, mp_word *extraR, const mp_number *a, const mp_word *extraA) {
    const mp_word c = mp_add(r, a); uint32_t prev = *extraR; *extraR = prev + *extraA + c; return (*extraR < prev) ? 1 : ((*extraR == prev) ? c : 0);
}

__device__ __forceinline__ mp_word mp_gte(const mp_number *a, const mp_number *b) {
    mp_word l = 0, g = 0; for (mp_word i = 0; i < MP_WORDS; ++i) { if (a->d[i] < b->d[i]) l |= (1u << i); if (a->d[i] > b->d[i]) g |= (1u << i); } return (g >= l);
}

__device__ __forceinline__ void mp_shr_extra(mp_number *r, mp_word *e) {
    r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
    r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
    r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
    r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
    r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
    r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
    r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
    r->d[7] = (*e << 31) | (r->d[7] >> 1);
    *e >>= 1;
}

__device__ __forceinline__ void mp_shr(mp_number *r) {
    r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
    r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
    r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
    r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
    r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
    r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
    r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
    r->d[7] = (r->d[7] >> 1);
}

__device__ __forceinline__ mp_word mp_mul_word_add_extra(mp_number *r, const mp_number *a, const mp_word w, mp_word *extra) {
    mp_word cM = 0, cA = 0; uint32_t tM = 0;
    for (mp_word i = 0; i < MP_WORDS; ++i) {
        tM = a->d[i] * w + cM; cM = mul_hi_u32(a->d[i], w) + (tM < cM);
        uint32_t t = r->d[i] + tM; uint32_t t2 = t + cA; cA = (t2 < t) ? 1 : (t2 == t ? cA : 0); r->d[i] = t2;
    }
    uint32_t prev = *extra; *extra = prev + cM + cA; return (*extra < prev) ? 1 : ((*extra == prev) ? cA : 0);
}

__device__ __forceinline__ void mp_mul_mod_word_sub(mp_number *r, const mp_word w, const bool withModHigher) {
    mp_number mod = const_mod;
    mp_number modhigher = { {0x00000000U, 0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU} };
    mp_word cM = 0, cS = 0, tS = 0, tM = 0, cA = 0;
    for (mp_word i = 0; i < MP_WORDS; ++i) {
        tM = (mod.d[i] * w + cM); cM = mul_hi_u32(mod.d[i], w) + (tM < cM);
        uint32_t addend = withModHigher ? modhigher.d[i] : 0u; uint32_t t = tM + addend; uint32_t t2 = t + cA; cA = (t2 < t) ? 1 : (t2 == t ? cA : 0); tM = t2;
        tS = r->d[i] - tM - cS; cS = tS > r->d[i] ? 1 : (tS == r->d[i] ? cS : 0); r->d[i] = tS;
    }
}

__device__ __forceinline__ void mp_mod_mul(mp_number *r, const mp_number *X, const mp_number *Y) {
    mp_number Z = { {0} }; mp_word extraWord;
    for (int i = MP_WORDS - 1; i >= 0; --i) {
        extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;
        bool overflow = mp_mul_word_add_extra(&Z, X, Y->d[i], &extraWord);
        mp_mul_mod_word_sub(&Z, extraWord, overflow);
    }
    *r = Z;
}

__device__ __forceinline__ void mp_mod_inverse(mp_number *r) {
    mp_number A = { {1} }, C = { {0} }, v = const_mod; mp_word extraA = 0, extraC = 0;
    while (r->d[0] || r->d[1] || r->d[2] || r->d[3] || r->d[4] || r->d[5] || r->d[6] || r->d[7]) {
        while (!(r->d[0] & 1)) { mp_shr(r); if (A.d[0] & 1) { extraA += mp_add_mod(&A); } mp_shr_extra(&A, &extraA); }
        while (!(v.d[0] & 1)) { mp_shr(&v); if (C.d[0] & 1) { extraC += mp_add_mod(&C); } mp_shr_extra(&C, &extraC); }
        if (mp_gte(r, &v)) { mp_sub(r, r, &v); mp_add_more(&A, &extraA, &C, &extraC); }
        else { mp_sub(&v, &v, r); mp_add_more(&C, &extraC, &A, &extraA); }
    }
    while (extraC) { extraC -= mp_sub_mod(&C); }
    v = const_mod; mp_sub(r, &v, &C);
}

// ----------------- EC helpers -----------------
__device__ __forceinline__ void point_add(point *r, point *p, point *o) {
    mp_number tmp, newX, newY;
    mp_mod_sub(&tmp, &o->x, &p->x);
    mp_mod_inverse(&tmp);
    mp_mod_sub(&newX, &o->y, &p->y);
    mp_mod_mul(&tmp, &tmp, &newX);
    mp_mod_mul(&newX, &tmp, &tmp);
    mp_mod_sub(&newX, &newX, &p->x);
    mp_mod_sub(&newX, &newX, &o->x);
    mp_mod_sub(&newY, &p->x, &newX);
    mp_mod_mul(&newY, &newY, &tmp);
    mp_mod_sub(&newY, &newY, &p->y);
    r->x = newX; r->y = newY;
}

__device__ __forceinline__ void profanity_init_seed(const point *precomp, point *p, bool *pIsFirst, const size_t precompOffset, const uint64_t seed, const uint32_t inverseSize) {
    point o;
    for (uint8_t i = 0; i < 8; ++i) {
        const uint8_t shift = i * 8;
        const uint8_t byte = (seed >> shift) & 0xFFu;
        if (byte) {
            o = precomp[precompOffset + i * inverseSize + (byte - 1)];
            if (*pIsFirst) { *p = o; *pIsFirst = false; }
            else { point_add(p, p, &o); }
        }
    }
}

// ----------------- kernels -----------------
__global__ void profanity_init(const point *precomp, mp_number *pDeltaX, mp_number *pPrevLambda, result *pResult, ulonglong4 seed, ulonglong4 seedX, ulonglong4 seedY, uint32_t sizeTotal, uint32_t inverseSize) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= sizeTotal) return;
    point p;
    p.x.d[0] = (uint32_t)(seedX.x & 0xFFFFFFFFull); p.x.d[1] = (uint32_t)(seedX.x >> 32);
    p.x.d[2] = (uint32_t)(seedX.y & 0xFFFFFFFFull); p.x.d[3] = (uint32_t)(seedX.y >> 32);
    p.x.d[4] = (uint32_t)(seedX.z & 0xFFFFFFFFull); p.x.d[5] = (uint32_t)(seedX.z >> 32);
    p.x.d[6] = (uint32_t)(seedX.w & 0xFFFFFFFFull); p.x.d[7] = (uint32_t)(seedX.w >> 32);
    p.y.d[0] = (uint32_t)(seedY.x & 0xFFFFFFFFull); p.y.d[1] = (uint32_t)(seedY.x >> 32);
    p.y.d[2] = (uint32_t)(seedY.y & 0xFFFFFFFFull); p.y.d[3] = (uint32_t)(seedY.y >> 32);
    p.y.d[4] = (uint32_t)(seedY.z & 0xFFFFFFFFull); p.y.d[5] = (uint32_t)(seedY.z >> 32);
    p.y.d[6] = (uint32_t)(seedY.w & 0xFFFFFFFFull); p.y.d[7] = (uint32_t)(seedY.w >> 32);
    point p_random; bool bIsFirst = true; mp_number tmp1, tmp2; point tmp3;
    profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * inverseSize * 0, seed.x, inverseSize);
    profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * inverseSize * 1, seed.y, inverseSize);
    profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * inverseSize * 2, seed.z, inverseSize);
    profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * inverseSize * 3, seed.w + id, inverseSize);
    point_add(&p, &p, &p_random);
    mp_mod_sub_gx(&tmp1, &p.x);
    mp_mod_inverse(&tmp1);
    mp_mod_sub_gy(&tmp2, &p.y); 
    mp_mod_mul(&tmp1, &tmp1, &tmp2);
    tmp3 = precomp[0];
    point_add(&p, &tmp3, &p);
    mp_mod_sub_gx(&p.x, &p.x);
    pDeltaX[id] = p.x; pPrevLambda[id] = tmp1;
    if (id == 0) { for (uint8_t i = 0; i < PROFANITY_MAX_SCORE + 1; ++i) { pResult[i].found = 0; } }
}

__global__ void profanity_inverse(const mp_number *pDeltaX, mp_number *pInverse, uint32_t sizeTotal, uint32_t inverseSize) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t groupStart = (uint64_t)gid * (uint64_t)inverseSize;
    if (groupStart >= sizeTotal) return;
    mp_number negativeDoubleGy = const_doubleNegativeGy;
    mp_number copy1, copy2;
    mp_number buffer[PROFANITY_MAX_INVERSE_SIZE];
    mp_number buffer2[PROFANITY_MAX_INVERSE_SIZE];
    buffer[0] = pDeltaX[groupStart];
    for (uint32_t i = 1; i < inverseSize; ++i) { buffer2[i] = pDeltaX[groupStart + i]; mp_mod_mul(&buffer[i], &buffer2[i], &buffer[i - 1]); }
    copy1 = buffer[inverseSize - 1]; mp_mod_inverse(&copy1);
    mp_mod_mul(&copy1, &copy1, &negativeDoubleGy);
    for (int i = (int)inverseSize - 1; i > 0; --i) { mp_mod_mul(&copy2, &copy1, &buffer[i - 1]); mp_mod_mul(&copy1, &copy1, &buffer2[i]); pInverse[groupStart + i] = copy2; }
    pInverse[groupStart] = copy1;
}

__global__ void profanity_iterate(mp_number *pDeltaX, mp_number *pInverse, mp_number *pPrevLambda, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= sizeTotal) return;
    mp_number negativeGx = const_negativeGx;
    ethhash h; for (int i = 0; i < 50; ++i) h.d[i] = 0;
    mp_number dX = pDeltaX[id]; mp_number tmp = pInverse[id]; mp_number lambda = pPrevLambda[id];
    mp_mod_sub(&lambda, &tmp, &lambda);
    mp_mod_mul(&tmp, &lambda, &lambda);
    mp_mod_sub(&dX, &dX, &tmp);
    mp_mod_sub_const(&dX, &const_tripleNegativeGx, &dX);
    pDeltaX[id] = dX; pPrevLambda[id] = lambda;
    mp_mod_mul(&tmp, &lambda, &dX);
    mp_mod_sub_const(&tmp, &const_negativeGy, &tmp);
    mp_mod_sub(&dX, &dX, &negativeGx);
    h.d[0] = bswap32(dX.d[MP_WORDS - 1]); h.d[1] = bswap32(dX.d[MP_WORDS - 2]); h.d[2] = bswap32(dX.d[MP_WORDS - 3]); h.d[3] = bswap32(dX.d[MP_WORDS - 4]);
    h.d[4] = bswap32(dX.d[MP_WORDS - 5]); h.d[5] = bswap32(dX.d[MP_WORDS - 6]); h.d[6] = bswap32(dX.d[MP_WORDS - 7]); h.d[7] = bswap32(dX.d[MP_WORDS - 8]);
    h.d[8] = bswap32(tmp.d[MP_WORDS - 1]); h.d[9] = bswap32(tmp.d[MP_WORDS - 2]); h.d[10] = bswap32(tmp.d[MP_WORDS - 3]); h.d[11] = bswap32(tmp.d[MP_WORDS - 4]);
    h.d[12] = bswap32(tmp.d[MP_WORDS - 5]); h.d[13] = bswap32(tmp.d[MP_WORDS - 6]); h.d[14] = bswap32(tmp.d[MP_WORDS - 7]); h.d[15] = bswap32(tmp.d[MP_WORDS - 8]);
    h.d[16] ^= 0x01; // length 64
    sha3_keccakf(&h);
    pInverse[id].d[0] = h.d[3]; pInverse[id].d[1] = h.d[4]; pInverse[id].d[2] = h.d[5]; pInverse[id].d[3] = h.d[6]; pInverse[id].d[4] = h.d[7];
}

__global__ void profanity_transform_contract(mp_number *pInverse, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= sizeTotal) return;
    const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d);
    ethhash h; for (int i = 0; i < 50; ++i) h.d[i] = 0;
    h.b[0] = 214; h.b[1] = 148; for (int i = 0; i < 20; ++i) h.b[i + 2] = hash[i]; h.b[22] = 128; h.b[23] ^= 0x01; // length 23
    sha3_keccakf(&h);
    pInverse[id].d[0] = h.d[3]; pInverse[id].d[1] = h.d[4]; pInverse[id].d[2] = h.d[5]; pInverse[id].d[3] = h.d[6]; pInverse[id].d[4] = h.d[7];
}

__device__ __forceinline__ void result_update(uint32_t id, const uint8_t *hash, result *pResult, uint8_t score, uint8_t scoreMax) {
    if (score && score > scoreMax) {
        uint32_t hasResult = atomicAdd(&pResult[score].found, 1u);
        if (hasResult == 0) { pResult[score].foundId = id; for (int i = 0; i < 20; ++i) pResult[score].foundHash[i] = hash[i]; }
    }
}

__global__ void profanity_score_benchmark(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_matching(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 20; ++i) { if (data1[i] > 0 && (hash[i] & data1[i]) == data2[i]) ++score; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_leading(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 20; ++i) { uint8_t first = (hash[i] & 0xF0) >> 4; uint8_t second = (hash[i] & 0x0F); if (first == data1[0]) ++score; else break; if (second == data1[0]) ++score; else break; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_range(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 20; ++i) { uint8_t first = (hash[i] & 0xF0) >> 4; uint8_t second = (hash[i] & 0x0F); if (first >= data1[0] && first <= data2[0]) ++score; else break; if (second >= data1[0] && second <= data2[0]) ++score; else break; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_zerobytes(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 20; ++i) { if (hash[i] == 0) ++score; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_leadingrange(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 20; ++i) { uint8_t first = (hash[i] & 0xF0) >> 4; uint8_t second = (hash[i] & 0x0F); if (first >= data1[0] && first <= data2[0]) ++score; else break; if (second >= data1[0] && second <= data2[0]) ++score; else break; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_mirror(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 10; ++i) { uint8_t leftLeft = (hash[9 - i] & 0xF0) >> 4; uint8_t leftRight = (hash[9 - i] & 0x0F); uint8_t rightLeft = (hash[10 + i] & 0xF0) >> 4; uint8_t rightRight = (hash[10 + i] & 0x0F); if (leftRight != rightLeft) break; ++score; if (leftLeft != rightRight) break; ++score; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

__global__ void profanity_score_doubles(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= sizeTotal) return; const uint8_t *hash = reinterpret_cast<const uint8_t*>(pInverse[id].d); int score = 0; for (int i = 0; i < 20; ++i) { uint8_t v = hash[i]; if ((v >> 4) == (v & 0x0F)) ++score; else break; } result_update(id, hash, pResult, (uint8_t)score, scoreMax);
}

} // extern "C"
