// +build sse
// +build amd64

// func vecAdd(a, b []float64)
TEXT ·vecAdd(SB), 7, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI // use detination index register for this

	MOVQ a_len+8(FP), AX  // len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap
	MOVQ b_len+32(FP), BX // len(b) into BX

	// check if they're the same length
	CMPQ AX, BX
	JNE  panic  // jump to panic if not the same length. TOOD: return bloody errors

	SUBQ $16, AX   // at least 16 elements?
	JL   remainder

loop:

	// a[0]
	MOVAPS (SI), X0
	MOVAPS (DI), X1
	ADDPS  X0, X1
	MOVAPS X1, (SI)

	MOVAPS 16(SI), X2
	MOVAPS 16(DI), X3
	ADDPS  X2, X3
	MOVAPS X3, 16(SI)

	MOVAPS 32(SI), X4
	MOVAPS 32(DI), X5
	ADDPS  X4, X5
	MOVAPS X5, 32(SI)

	MOVAPS 48(SI), X6
	MOVAPS 48(DI), X7
	ADDPS  X6, X7
	MOVAPS X7, 48(SI)

	// update pointers (4 * 4 * 4) - 4 elements each register, 4 registers used, each element is 4 bytes
	ADDQ $64, SI
	ADDQ $64, DI

	// start of array is now 4*4 less
	SUBQ $16, AX
	JGE  loop

remainder:
	ADDQ $16, AX
	JE   done

remainderloop:

	// copy into the appropriate registers
	MOVSS (SI), X0
	MOVSS (DI), X1
	ADDSS X0, X1

	// save it back
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

