// +build sse
// +build amd64

#include "textflag.h"

// func vecAdd(a, b []float32)
TEXT ·vecAdd(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI // use destination index register for this

	MOVQ a_len+8(FP), AX  // len(a) into AX
	MOVQ b_len+32(FP), BX // len(b) into BX

	// check if they're the same length
	CMPQ AX, BX
	JNE  panic  // jump to panic if not the same length. TOOD: return bloody errors

	// check if there are at least 16 elements
	SUBQ $16, AX
	JL   remainder

loop:

	// a[0]
	MOVUPS (SI), X0
	MOVUPS (DI), X1
	ADDPS  X0, X1
	MOVUPS X1, (SI)

	MOVUPS 16(SI), X2
	MOVUPS 16(DI), X3
	ADDPS  X2, X3
	MOVUPS X3, 16(SI)

	MOVUPS 32(SI), X4
	MOVUPS 32(DI), X5
	ADDPS  X4, X5
	MOVUPS X5, 32(SI)

	MOVUPS 48(SI), X6
	MOVUPS 48(DI), X7
	ADDPS  X6, X7
	MOVUPS X7, 48(SI)

	// update pointers. 4 registers, 4 elements each, 4 bytes per element
	ADDQ $64, SI
	ADDQ $64, DI

	// len(a) is now 4*4 elements less
	SUBQ $16, AX
	JGE  loop

remainder:
	ADDQ $16, AX
	JE   done

remainderloop:
	MOVSS (SI), X0
	MOVSS (DI), X1
	ADDSS X0, X1
	MOVSS X1, (SI)

	// update pointer to the top of the data
	ADDQ $4, SI
	ADDQ $4, DI

	DECQ AX
	JNE  remainderloop

done:
	RET

panic:
	CALL runtime·panicindex(SB)
	RET
