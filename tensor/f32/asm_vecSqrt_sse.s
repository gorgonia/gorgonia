// +build sse
// +build amd64
// +build !fastmath

/*
vecSqrt takes a []float32 and square roots every element in the slice.
*/
#include "textflag.h"

// func vecSqrt(a []float32)
TEXT ·vecSqrt(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $4, AX
	JL remainder

loop:
	SQRTPS	(SI), X0
	MOVUPS	X0, (SI)
	
	// we processed 4 elements. Each element is 4 bytes. So jump 16 ahead
	ADDQ	$16, SI

	SUBQ	$4, AX
	JGE		loop

remainder:
	ADDQ	$4, AX
	JE		done

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
