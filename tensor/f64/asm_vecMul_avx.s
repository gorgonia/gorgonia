// +build avx
// +build amd64

/*
This function adds two []float64 with some SIMD optimizations using AVX.

Instead of doing this:
	for i := 0; i < len(a); i++ {
		a[i] += b[i]
	}

Here, I use the term "pairs" to denote an element of `a` and and element of `b` that will be added together. a[i], b[i] is a pair.

Using AVX, we can simultaneously add 16 pairs at the same time, which will look something like this:
	for i := 0; i < len(a); i+=16{
		a[i:i+15] += b[i:i+15] // this code won't run.
	}

AVX registers are 256 bits, meaning we can put 4 float64s in there. 
There are a total of 16 YMM registers, but somehow when writing this, I could only access the first 8 (%YMM0-%YMM7). This means theoretically up to 64 pairs (each register takes 4 items, there are 16 registers) can be parallelly processed. However, because we're only limited to 8 registers, we can for now only process 16 pairs.

The function uses as many registers as can be used, and the loop is aggressively unrolled. Here I explain exactly how the function works, illustrated with an example.

Given a slice of 100 items, it'll process 16 elements at a time, decrementing the count by 16 each time (see the label `loop`). Once it is no longer able to process 16 elements - i.e. there is a remainder of 4, the code will enter the `remainder` loop. 

At the start of the remainder loop, 16 is added back to the count. And then the count is subtracted by 12. If it's less than 0, the code will jump to the remainder12 loop, where a check is made to see if there are 8 items or fewer. This will go on until a remainder of <4 is found. Then those pairs will be manually added together without AVX.

These are the registers I use to store the relevant information:
	SI - Used to store the top element of slice A (index 0). This register is incremented every loop
	DI - used to store the top element of slice B. Incremented every loop
	AX - len(a) is stored in here. Volatile register. AX is also used as the "working" count of the length that is decremented.
	BX - len(b) is stored in here. Volatile register. Used to compare against AX at the beginning to make sure both a and b have the same lengths
	Y0-Y7 - YMM registers. 256 bit registers for fun. Don't have to be particularly safe about them (like saving values etc) because most golang code does not use the YMM registers

*/

#include "textflag.h"

// func vecMul(a, b []float64)
TEXT ·vecMul(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI  		// use detination index register for this

	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap
	MOVQ b_len+32(FP), BX			// len(b) into BX

	// check if they're the same length
	CMPQ AX, BX
	JNE panic						// jump to panic if not the same length. TOOD: return bloody errors

	// each ymm register can take up to 4 float64s. 
	SUBQ $4, AX
	JL remainder



loop:
	// a[0] to a[3]
	// VMOVUPD 0(SI), Y0
	// VMOVUPD 0(DI), Y1
	// VMULPD Y0, Y1, Y0
	// VMOVUPD  Y0, 0(SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x06;       // vmovupd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x0f;       // vmovupd (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf5; BYTE $0x59; BYTE $0xc0;       // vmulpd %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;       // vmovupd %ymm0,(%rsi)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, AX
	JGE loop

remainder:
	ADDQ 	$4, AX
	JE 		done

	SUBQ 	$2, AX
	JL 		remainder1head

remainder2:
	// VMOVUPD (SI), X0
	// VMOVUPD (DI), X1
	// VMULPD X0, X1, X0
	// VMOVUPD X0, (SI)

	BYTE $0xc5; BYTE $0xf9; BYTE $0x10; BYTE $0x06;    // vmovupd (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xf9; BYTE $0x10; BYTE $0x0f;    // vmovupd (%rdi),%xmm1
	BYTE $0xc5; BYTE $0xf1; BYTE $0x59; BYTE $0xc0;    // vmulpd %xmm0,%xmm1,%xmm0
	BYTE $0xc5; BYTE $0xf9; BYTE $0x11; BYTE $0x06;    // vmovupd %xmm0,(%rsi)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, AX
	JGE remainder2

remainder1head:
	ADDQ $2, AX
	JE done

remainder1:
	// copy into the appropriate registers
	// VMOVSD	(SI), X0
	// VMOVSD	(DI), X1
	// VADDSD	X0, X1, X0
	// VMOVSD	X0, (SI)
	BYTE $0xc5; BYTE $0xfb; BYTE $0x10; BYTE $0x06    // vmovsd (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xfb; BYTE $0x10; BYTE $0x0f    // vmovsd (%rdi),%xmm1
	BYTE $0xc5; BYTE $0xf3; BYTE $0x59; BYTE $0xc0    // vmulsd %xmm0,%xmm1,%xmm0
	BYTE $0xc5; BYTE $0xfb; BYTE $0x11; BYTE $0x06    // vmovsd %xmm0,(%rsi)

	// update pointer to the top of the data
	ADDQ 	$8, SI
	ADDQ	$8, DI
	
	DECQ AX
	JNE 	remainder1

done:
	RET

panic:
	CALL 	runtime·panicindex(SB)
	RET


