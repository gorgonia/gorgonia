// +build avx
// +build amd64

/*

*/

TEXT ·vecSqrt(SB), 7, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $32, AX
	JL remainder

loop:
	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	// a[8] to a[11]
	// VSQRTPD 64(SI), Y2
	// VMOVUPD Y2, 64(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x56; BYTE $0x40;       	// vsqrtpd 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x40;       	// vmovupd %ymm2,0x40(%rsi)

	// a[12] to a[15]
	// VSQRTPD 96(SI), Y3
	// VMOVUPD Y3, 96(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x5e; BYTE $0x60;       	// vsqrtpd 0x60(%rsi),%ymm3
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x5e; BYTE $0x60;       	// vmovupd %ymm3,0x60(%rsi)

	// a[16] to a[19]
	// VSQRTPD 128(SI), Y4
	// VMOVUPD Y4, 128(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0x80(%rsi),%ymm4
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xa6; BYTE $0x80;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm4,0x80(%rsi)
	BYTE $0x00; 


	// a[20] to a[23]
	// VSQRTPD 160(SI), Y5
	// VMOVUPD Y5, 160(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xae; BYTE $0xa0;  BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0xa0(%rsi),%ymm5
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xae; BYTE $0xa0;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm5,0xa0(%rsi)
	BYTE $0x00; 

	// a[24] to a[27]
	// VSQRTPD 192(SI), Y6
	// VMOVUPD Y6, 192(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xb6; BYTE $0xc0;  BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0xc0(%rsi),%ymm6
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xb6; BYTE $0xc0;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm6,0xc0(%rsi)
	BYTE $0x00; 

	// a[28] to a[31]
	// VSQRTPD 128(SI), Y7
	// VMOVUPD Y7, 128(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xbe; BYTE $0xe0;  BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0xe0(%rsi),%ymm7
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xbe; BYTE $0xe0;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm7,0xe0(%rsi)
	BYTE $0x00; 

	
	// we processed 32 elements. Each element is 8 bytes. So jump 256 ahead
	ADDQ	$256, SI

	SUBQ	$32, AX
	JGE		loop

remainder:
	ADDQ	$32, AX
	JE 		done

	SUBQ 	$28, AX
	JL 		remainder28

	// otherwise there are 28 <= x < 32 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	// a[8] to a[11]
	// VSQRTPD 64(SI), Y2
	// VMOVUPD Y2, 64(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x56; BYTE $0x40;       	// vsqrtpd 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x40;       	// vmovupd %ymm2,0x40(%rsi)

	// a[12] to a[15]
	// VSQRTPD 96(SI), Y3
	// VMOVUPD Y3, 96(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x5e; BYTE $0x60;       	// vsqrtpd 0x60(%rsi),%ymm3
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x5e; BYTE $0x60;       	// vmovupd %ymm3,0x60(%rsi)

	// a[16] to a[19]
	// VSQRTPD 128(SI), Y4
	// VMOVUPD Y4, 128(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0x80(%rsi),%ymm4
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xa6; BYTE $0x80;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm4,0x80(%rsi)
	BYTE $0x00; 


	// a[20] to a[23]
	// VSQRTPD 160(SI), Y5
	// VMOVUPD Y5, 160(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xae; BYTE $0xa0;  BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0xa0(%rsi),%ymm5
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xae; BYTE $0xa0;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm5,0xa0(%rsi)
	BYTE $0x00; 

	// a[24] to a[27]
	// VSQRTPD 192(SI), Y6
	// VMOVUPD Y6, 192(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xb6; BYTE $0xc0;  BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0xc0(%rsi),%ymm6
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xb6; BYTE $0xc0;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm6,0xc0(%rsi)
	BYTE $0x00; 

	ADDQ	$224, SI
	SUBQ	$28, AX

remainder28:
	ADDQ 	$28, AX
	JE 		done

	SUBQ	$24, AX
	JL		remainder24

	// otherwise there are 24 <= x < 28 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	// a[8] to a[11]
	// VSQRTPD 64(SI), Y2
	// VMOVUPD Y2, 64(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x56; BYTE $0x40;       	// vsqrtpd 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x40;       	// vmovupd %ymm2,0x40(%rsi)

	// a[12] to a[15]
	// VSQRTPD 96(SI), Y3
	// VMOVUPD Y3, 96(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x5e; BYTE $0x60;       	// vsqrtpd 0x60(%rsi),%ymm3
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x5e; BYTE $0x60;       	// vmovupd %ymm3,0x60(%rsi)

	// a[16] to a[19]
	// VSQRTPD 128(SI), Y4
	// VMOVUPD Y4, 128(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0x80(%rsi),%ymm4
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xa6; BYTE $0x80;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm4,0x80(%rsi)
	BYTE $0x00; 


	// a[20] to a[23]
	// VSQRTPD 160(SI), Y5
	// VMOVUPD Y5, 160(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xae; BYTE $0xa0;  BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0xa0(%rsi),%ymm5
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xae; BYTE $0xa0;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm5,0xa0(%rsi)
	BYTE $0x00; 

	ADDQ	$192, SI
	SUBQ	$24, AX

remainder24:
	ADDQ	$24, AX
	JE 		done

	SUBQ	$20, AX
	JL		remainder20

	// otherwise there are 20 <= x < 24 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	// a[8] to a[11]
	// VSQRTPD 64(SI), Y2
	// VMOVUPD Y2, 64(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x56; BYTE $0x40;       	// vsqrtpd 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x40;       	// vmovupd %ymm2,0x40(%rsi)

	// a[12] to a[15]
	// VSQRTPD 96(SI), Y3
	// VMOVUPD Y3, 96(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x5e; BYTE $0x60;       	// vsqrtpd 0x60(%rsi),%ymm3
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x5e; BYTE $0x60;       	// vmovupd %ymm3,0x60(%rsi)

	// a[16] to a[19]
	// VSQRTPD 128(SI), Y4
	// VMOVUPD Y4, 128(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00; 	// vsqrtpd 0x80(%rsi),%ymm4
	BYTE $0x00; 
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0xa6; BYTE $0x80;  BYTE $0x00; BYTE $0x00; 	// vmovupd %ymm4,0x80(%rsi)
	BYTE $0x00;

	ADDQ	$160, SI
	SUBQ	$20, AX

remainder20:
	ADDQ	$20, AX
	JE		done

	SUBQ 	$16, AX
	JL		remainder16

	// otherwise there are 16 <= x < 20 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	// a[8] to a[11]
	// VSQRTPD 64(SI), Y2
	// VMOVUPD Y2, 64(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x56; BYTE $0x40;       	// vsqrtpd 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x40;       	// vmovupd %ymm2,0x40(%rsi)

	// a[12] to a[15]
	// VSQRTPD 96(SI), Y3
	// VMOVUPD Y3, 96(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x5e; BYTE $0x60;       	// vsqrtpd 0x60(%rsi),%ymm3
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x5e; BYTE $0x60;       	// vmovupd %ymm3,0x60(%rsi)	

	ADDQ	$128, SI
	SUBQ	$16, AX

remainder16:
	ADDQ	$16, AX
	JE		done

	SUBQ 	$12, AX
	JL		remainder12

	// otherwise there are 12 <= x < 16 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	// a[8] to a[11]
	// VSQRTPD 64(SI), Y2
	// VMOVUPD Y2, 64(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x56; BYTE $0x40;       	// vsqrtpd 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x40;       	// vmovupd %ymm2,0x40(%rsi)


	ADDQ	$96, SI
	SUBQ	$12, AX

remainder12:
	ADDQ	$12, AX
	JE		done

	SUBQ 	$8, AX
	JL		remainder8

	// otherwise there are 8 <= x < 12 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	// a[4] to a[7]
	// VSQRTPD 32(SI), Y1
	// VMOVUPD Y1, 32(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x4e; BYTE $0x20;       	// vsqrtpd 0x20(%rsi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x4e; BYTE $0x20;       	// vmovupd %ymm1,0x20(%rsi)

	ADDQ	$64, SI
	SUBQ	$8, AX

remainder8:
	ADDQ	$8, AX
	JE		done

	SUBQ 	$4, AX
	JL		remainder4

	// otherwise there are 4 <= x < 8 items left

	// a[0] to a[3]
	// VSQRTPD (SI), Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x51; BYTE $0x06;          			// vsqrtpd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;          			// vmovupd %ymm0,(%rsi)

	ADDQ	$32, SI
	SUBQ	$4, AX

remainder4:
	ADDQ	$4, AX
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
