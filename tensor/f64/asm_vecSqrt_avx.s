// +build avx
// +build amd64
// +build !fastmath

/*
vecSqrt takes a []float32 and square roots every element in the slice.
*/
#include "textflag.h"

// func vecSqrt(a []float64)
TEXT ·vecSqrt(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX  // len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $4, AX
	JL   remainder

loop:
	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06 // vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06 // vmovupd %ymm0,(%rsi)

	ADDQ $32, SI
	SUBQ $4, AX
	JGE  loop

remainder:
	ADDQ $4, AX
	JE   done

	SUBQ $2, AX
	JL   remainder1head

remainder2:
	// VSQRTPS (SI), X0
	// VMOVUPS X0, (SI)
	BYTE $0xc5; BYTE $0xf9; BYTE $0x51; BYTE $0x06 // vsqrtpd (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xf9; BYTE $0x11; BYTE $0x06 // vmovupd %xmm0,(%rsi)

	ADDQ $16, SI
	SUBQ $2, AX
	JGE  remainder2

remainder1head:
	ADDQ $2, AX
	JE   done

remainder1:
	MOVSD  (SI), X0
	SQRTSD X0, X0
	MOVSD  X0, (SI)

	ADDQ $8, SI
	DECQ AX
	JNE  remainder1

done:
	RET

panic:
	CALL runtime·panicindex(SB)
	RET
