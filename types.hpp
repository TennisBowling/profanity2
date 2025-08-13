#ifndef HPP_TYPES
#define HPP_TYPES

/* Types used for host-side buffers exchanged with GPU code.
 * These are defined in terms of fixed-width integers to avoid
 * coupling to any specific GPU host API headers. The layouts
 * must remain identical to the corresponding types in kernels.
 */

#include <cstdint>

#define MP_NWORDS 8

typedef uint32_t mp_word;

typedef struct {
    mp_word d[MP_NWORDS];
} mp_number;

typedef struct {
    mp_number x;
    mp_number y;
} point;

typedef struct {
    uint32_t found;
    uint32_t foundId;
    uint8_t  foundHash[20];
} result;

#endif /* HPP_TYPES */
