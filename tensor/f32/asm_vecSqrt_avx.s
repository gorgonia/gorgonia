// +build avx
// +build amd64

/*
vecSqrt takes a []float32 and square roots every element in the slice.
*/
#include "textflag.h"

// func vecSqrt(a []float32)
TEXT ·vecSqrt(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $8, AX
	JL remainder

loop:
	// a[0] to a[7]
	// VSQRTPS (SI), Y0
	// VMOVUPS Y0, (SI)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x51; BYTE $0x06;          			// vsqrtps (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x06;          			// vmovups %ymm0,(%rsi)

	ADDQ $32, SI
	SUBQ $8, AX
	JGE loop

remainder:
	ADDQ	$8, AX
	JE		done

	SUBQ 	$4, AX
	JL 		remainder1head

remainder4:
	// VSQRTPS (SI), X0
	// VMOVUPS X0, (SI)
	BYTE $0xc5; BYTE $0xf8; BYTE $0x51; BYTE $0x06;          			// vsqrtps (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xf8; BYTE $0x11; BYTE $0x06;          			// vmovups %xmm0,(%rsi)

	ADDQ $16, SI
	SUBQ $4, AX
	JGE remainder4

remainder1head:
	ADDQ $4, AX
	JE done

remainder1:
	MOVSS 	(SI), X0
	SQRTSS	X0, X0
	MOVSS	X0, (SI)
	
	ADDQ	$4, SI
	DECQ	AX
	JNE		remainder1


done:
	RET
panic:
	CALL 	runtime·panicindex(SB)
	RET
