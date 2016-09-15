// +build sse
// +build amd64
// +build !fastmath

/*

*/

TEXT ·vecSqrt(SB), 7, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $2, AX
	JL remainder

loop:
	SQRTPD	(SI), X0
	MOVUPD	X0, (SI)
	
	// we processed 2 elements. Each element is 8 bytes. So jump 16 ahead
	ADDQ	$16, SI

	SUBQ	$2, AX
	JGE		loop

remainder:
	ADDQ	$2, AX
	JE		done

remainder1:
	MOVSD 	(SI), X0
	SQRTSD	X0, X0
	MOVSD	X0, (SI)
	
	ADDQ	$8, SI
	DECQ	AX
	JNE		remainder1


done:
	RET
panic:
	CALL 	runtime·panicindex(SB)
	RET
