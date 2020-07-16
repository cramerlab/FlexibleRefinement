#pragma once

/** Speed up temporary variables */
#define SPEED_UP_temps0 \
    RDOUBLE spduptmp0;

/** Speed up temporary variables */
#define SPEED_UP_temps01 \
	SPEED_UP_temps0; \
    RDOUBLE spduptmp1;

/** Speed up temporary variables */
#define SPEED_UP_temps012 \
	SPEED_UP_temps01; \
    RDOUBLE spduptmp2;

#define dMn(m, n)  ((m).mdata[(n)])

#define M3x3_BY_V3x1(a, M, b) { \
        spduptmp0 = dMn(M, 0) * XX(b) + dMn(M, 1) * YY(b) + dMn(M, 2) * ZZ(b); \
        spduptmp1 = dMn(M, 3) * XX(b) + dMn(M, 4) * YY(b) + dMn(M, 5) * ZZ(b); \
        spduptmp2 = dMn(M, 6) * XX(b) + dMn(M, 7) * YY(b) + dMn(M, 8) * ZZ(b); \
        XX(a) = spduptmp0; YY(a) = spduptmp1; ZZ(a) = spduptmp2; }


/** Speed up temporary variables */
#define SPEED_UP_tempsInt \
    int   ispduptmp0, ispduptmp1, ispduptmp2, \
    ispduptmp3, ispduptmp4, ispduptmp5;
