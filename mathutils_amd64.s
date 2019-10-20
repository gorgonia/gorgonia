// +build !noasm,!wasm, !arm64

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
