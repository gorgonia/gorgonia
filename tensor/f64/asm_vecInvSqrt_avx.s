// +build avx
// +build amd64

/*

*/


TEXT ·vecInvSqrt(SB), 7, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	// make sure that len(a) >= 1
	XORQ	BX, BX
	CMPQ	BX,AX
	JGE		done

	// const 1.0
	MOVQ	$0x3ff0000000000000, DX 

	SUBQ $4, AX
	JL remainder


	// store the first element in BX
	// This is done so that we can move 1.0 into the first element of the slice
	// because AVX instruction vbroadcastsd can only read from memory location not from registers
	MOVQ	(SI), BX

	// load 1.0 into the first element
	MOVQ	DX, (SI)

	// VBROADCASTSD (SI), Y1
	BYTE $0xc4; BYTE $0xe2; BYTE $0x7d; BYTE $0x19; BYTE $0x0e //vbroadcastsd (%rbx),%ymm1

	// now that we're done with the ghastly business of trying to broadcast 1.0 without using any extra memory...
	// we restore the first element
	MOVQ	BX, (SI)

loop:

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// VDIVPD Y0, Y1, Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xf5; BYTE $0x5e; BYTE $0xc0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06 

	ADDQ	$32, SI
	SUBQ	$4, AX
	JGE		loop

remainder:
	ADDQ	$4, AX
	JE		done
	//MOVQ	$0x3ff0000000000000, DX  // because DX will be clobbered

remainder1:
	MOVQ	DX, X1
	MOVSD 	(SI), X0
	SQRTSD	X0, X0
	DIVSD	X0, X1
	MOVSD	X1, (SI)
	//MOVQ X0, (CX)
	//RET
	
	ADDQ	$8, SI
	DECQ	AX
	JNE		remainder1


done:
	RET
panic:
	CALL 	runtime·panicindex(SB)
	RET
