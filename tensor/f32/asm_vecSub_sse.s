// +build sse
// +build amd64

#include "textflag.h"

// func vecSub(a, b []float32)
TEXT ·vecSub(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI // use destination index register for this

	MOVQ a_len+8(FP), AX  // len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap
	MOVQ b_len+32(FP), BX // len(b) into BX

	// check if they're the same length
	CMPQ AX, BX
	JNE  panic  // jump to panic if not the same length. TOOD: return bloody errors

	SUBQ $16, AX   // at least  16 elements?
	JL   remainder

loop:

	// a[0]
	MOVAPS (SI), X0
	MOVAPS (DI), X1
	SUBPS  X1, X0
	MOVAPS X0, (SI)

	MOVAPS 16(SI), X2
	MOVAPS 16(DI), X3
	SUBPS  X3, X2
	MOVAPS X2, 16(SI)

	MOVAPS 32(SI), X4
	MOVAPS 32(DI), X5
	SUBPS  X5, X4
	MOVAPS X4, 32(SI)

	MOVAPS 48(SI), X6
	MOVAPS 48(DI), X7
	SUBPS  X7, X6
	MOVAPS X6, 48(SI)

	// update pointers. 4 element per register, 4 registers, 4 bytes per element. So jump 4*4*4 bytes ahead
	ADDQ $64, SI
	ADDQ $64, DI

	// len(a) now 4*4 less
	SUBQ $16, AX
	JGE  loop

remainder:
	ADDQ $16, AX
	JE   done

remainderloop:

	// copy into the appropriate registers
	MOVSS (SI), X0
	MOVSS (DI), X1
	SUBSS X1, X0

	// save it back
	MOVSS X0, (SI)

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

