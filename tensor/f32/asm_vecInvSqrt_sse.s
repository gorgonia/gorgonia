// +build sse
// +build amd64
// +build !fastmath

/*

*/
#include "textflag.h"


#define one 0x3f800000


// func vecInvSqrt(a []float32)
TEXT ·vecInvSqrt(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	// make sure that len(a) >= 1
	XORQ	BX, BX
	CMPQ	BX,AX
	JGE		done

	// const 1.0
	MOVL	$one, DX 


	SUBQ $4, AX
	JL remainder

	// back up the first element of the slice
	MOVL	(SI), BX
	MOVL	DX, (SI)

	// broadcast 1.0 to all elements of X1
	// 0x00 shuffles the least significant bits of the X1 reg, which means the first element is repeated
	MOVUPS	(SI), X1
	SHUFPS 	$0x00, X1, X1 

	MOVAPS 	X1, X2 // backup, because X1 will get clobbered in DIVPS

	// restore the first element now we're done
	MOVL	BX, (SI)


loop:
	MOVAPS 	X2, X1
	SQRTPS	(SI), X0
	DIVPS	X0, X1
	MOVUPS	X1, (SI)
	
	// we processed 4 elements. Each element is 4 bytes. So jump 16 ahead
	ADDQ	$16, SI

	SUBQ	$4, AX
	JGE		loop

remainder:
	ADDQ	$4, AX
	JE		done

remainder1:
	MOVQ 	DX, X1
	MOVSS 	(SI), X0
	SQRTSS	X0, X0
	DIVSS 	X0, X1
	MOVSS	X1, (SI)
	
	ADDQ	$4, SI
	DECQ	AX
	JNE		remainder1


done:
	RET
panic:
	CALL 	runtime·panicindex(SB)
	RET
