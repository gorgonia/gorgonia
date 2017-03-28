// +build !noasm

#include "textflag.h"

// divmod(a, b int) (q,r int)
TEXT ·divmod(SB),NOSPLIT,$0
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), CX
	MOVQ	SI, AX
	CMPQ	CX, $-1
	JEQ	$1, denomIsOne 	// if denominator is 1, then jump to end

	CQO
	IDIVQ	CX
	MOVQ	AX, q+16(FP)
	MOVQ	DX, r+24(FP)
bye:
	RET
denomIsOne:
	NEGQ	AX
	MOVQ	AX, q+16(FP)
	MOVQ	$0, r+24(FP)
	JMP	bye

// popcnt(uint64) int
TEXT ·popcnt(SB),NOSPLIT,$0
	POPCNTQ    x+0(FP), AX
	MOVQ       AX, ret+8(FP)
	RET

// clz(uint64) int
TEXT ·clz(SB),4,$0-16
        BSRQ  x+0(FP), AX
        JZ zero
        SUBQ  $63, AX
        NEGQ AX
        MOVQ AX, ret+8(FP)
        RET
zero:
        MOVQ $64, ret+8(FP)
        RET
