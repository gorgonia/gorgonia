// +build DONOTUSE
// +build avx
// +build amd64


/*
This function adds two []float32 with some SIMD optimizations using AVX.

Instead of doing this:
	for i := 0; i < len(a); i++ {
		a[i] += b[i]
	}

Here, I use the term "pairs" to denote an element of `a` and and element of `b` that will be added together. 
a[i], b[i] is a pair.

Using AVX, we can simultaneously add 32 pairs at the same time, which will look something like this:
	for i := 0; i < len(a); i+=16{
		a[i:i+15] += b[i:i+15] // this code won't run.
	}

AVX registers are 256 bits, meaning we can put 8 float32s in there. 
There are a total of 16 YMM registers, but somehow when writing this, 
I could only access the first 8 (%YMM0-%YMM7). 
This means theoretically up to 128 pairs (each register takes 8 items, there are 16 registers) can be parallelly processed. 
However, because we're only limited to 8 registers, we can for now only process 32 pairs.

The function uses as many registers as can be used, and the loop is aggressively unrolled. 
Here I explain exactly how the function works, illustrated with an example.

Given a slice of 100 items, it'll process 32 elements at a time, 
decrementing the count by 32 each time (see the label `loop`). 
Once it is no longer able to process 32 elements - i.e. there is a remainder of 8, 
the code will enter the `remainder` loop. 

At the start of the remainder loop, 32 is added back to the count. And then the count is subtracted by 24. 
If it's less than 0, the code will jump to the remainder24 loop, where a check is made to see if there are 16 items or fewer. 
This will go on until a remainder of <8 is found. Then those pairs will be manually added together without AVX.

These are the registers I use to store the relevant information:
	SI - Used to store the top element of slice A (index 0). This register is incremented every loop
	DI - used to store the top element of slice B. Incremented every loop
	AX - len(a) is stored in here. AX is also used as the "working" count of the length that is decremented.
	BX - len(b) is stored in here. Used to compare against AX at the beginning to make sure both a and b have the same lengths
	Y0-Y7 - YMM registers. 

*/

#include "textflag.h"

// func add64(a, b []float32)
TEXT ·vecAdd(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI  		// use detination index register for this

	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap
	MOVQ b_len+32(FP), BX			// len(b) into BX
	MOVQ AX, AX						// len(a) into AX for working purposes

	// check if they're the same length
	CMPQ AX, BX
	JNE panic						// jump to panic if not the same length. TOOD: return bloody errors

	// each ymm register can take up to 8 float32s. 
	// There are 8 ymm registers (8 pairs to do addition) available (TODO: check how to access the other 8 ymm registers without fucking things up)
	// Therefore a total of 32 elements can be processed at a time

	SUBQ $32, AX
	JL remainder



loop:
	// a[0] to a[7]
	// VMOVUPS 0(%rsi), %ymm0
	// VMOVUPS 0(%rdi), %ymm1
	// VADDPS %ymm0, %ymm1, %ymm0
	// VMOVUPS %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x06;       // vmovups (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x0f;       // vmovups (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf4; BYTE $0x58; BYTE $0xc0;       // vaddps %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x06;       // vmovups %ymm0,(%rsi)

	// a[8] to a[15]
	// VMOVUPS 64(%rsi), %ymm2
	// VMOVUPS 64(%rdi), %ymm3
	// VADDPS %ymm2, %ymm3, %ymm2
	// VMOVUPS %ymm2, 64(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x56; BYTE $0x40        // vmovups 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x5f; BYTE $0x40        // vmovups 0x40(%rdi),%ymm3
	BYTE $0xc5; BYTE $0xe4; BYTE $0x58; BYTE $0xd2                    // vaddps %ymm2,%ymm3,%ymm2
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x56; BYTE $0x40        // vmovups %ymm2,0x40(%rsi)

	// a[16] to a[23]
	// VMOVUPS 128(%rsi), %ymm4
	// VMOVUPS 128(%rdi), %ymm5
	// VADDPS %ymm4, %ymm5, %ymm4
	// VMOVUPS %ymm4, 128(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00		// vmovups 0x80(%rsi),%ymm4
	BYTE $0x00;
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0xaf; BYTE $0x80; BYTE $0x00; BYTE $0x00		// vmovups 0x80(%rdi),%ymm5
	BYTE $0x00;
	BYTE $0xc5; BYTE $0xd4; BYTE $0x58; BYTE $0xe4;                  						// vaddps %ymm4,%ymm5,%ymm4
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00		// vmovups %ymm6,0x80(%rsi)
	BYTE $0x00;

  	// a[24] to a[31]
	// VMOVUPS 256(%rsi), %ymm6
	// VMOVUPS 256(%rdi), %ymm7
	// VADDPS %ymm6, %ymm7, %ymm6
	// VMOVUPS %ymm6, 256(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0xb6; BYTE $0x00; BYTE $0x01; BYTE $0x00		// vmovups 0x60(%rsi),%ymm6
	BYTE $0x00;
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0xbf; BYTE $0x00; BYTE $0x01; BYTE $0x00		// vmovups 0x60(%rdi),%ymm7
	BYTE $0x00;
	BYTE $0xc5; BYTE $0xc4; BYTE $0x58; BYTE $0xf6;                   						// vaddps %ymm6,%ymm7,%ymm6
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0xb6; BYTE $0x60; BYTE $0x01; BYTE $0x00		// vmovups %ymm8,0x60(%rsi)
	BYTE $0x00;


	// update pointers (4*8 * 8) - 4*8 elements each time, each element is 8 bytes
	// so jump ahead 256 bytes for next i
	ADDQ $256, SI
	ADDQ $256, DI

	SUBQ $32, AX
	JGE loop

remainder:
	ADDQ 	$32, AX
	JE 		done

	SUBQ 	$24, AX
	JL 		remainder12

	// otherwise,
	// there are 24 <= x < 32 items left!

	// a[0] to a[7]
	// VMOVUPS 0(%rsi), %ymm0
	// VMOVUPS 0(%rdi), %ymm1
	// VADDPS %ymm0, %ymm1, %ymm0
	// VMOVUPS %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x06;       // vmovups (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x0f;       // vmovups (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf4; BYTE $0x58; BYTE $0xc0;       // vaddps %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x06;       // vmovups %ymm0,(%rsi)

	// a[8] to a[15]
	// VMOVUPS 64(%rsi), %ymm2
	// VMOVUPS 64(%rdi), %ymm3
	// VADDPS %ymm2, %ymm3, %ymm2
	// VMOVUPS %ymm2, 64(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x56; BYTE $0x40        // vmovups 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x5f; BYTE $0x40        // vmovups 0x40(%rdi),%ymm3
	BYTE $0xc5; BYTE $0xe4; BYTE $0x58; BYTE $0xd2                    // vaddps %ymm2,%ymm3,%ymm2
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x56; BYTE $0x40        // vmovups %ymm2,0x40(%rsi)

	// a[16] to a[23]
	// VMOVUPS 128(%rsi), %ymm4
	// VMOVUPS 128(%rdi), %ymm5
	// VADDPS %ymm4, %ymm5, %ymm4
	// VMOVUPS %ymm4, 128(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00		// vmovups 0x80(%rsi),%ymm4
	BYTE $0x00;
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0xaf; BYTE $0x80; BYTE $0x00; BYTE $0x00		// vmovups 0x80(%rdi),%ymm5
	BYTE $0x00;
	BYTE $0xc5; BYTE $0xd4; BYTE $0x58; BYTE $0xe4;                  						// vaddps %ymm4,%ymm5,%ymm4
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0xa6; BYTE $0x80; BYTE $0x00; BYTE $0x00		// vmovups %ymm6,0x80(%rsi)
	BYTE $0x00;


  	// update pointers (3*8 * 8) - 3*4 elements each time, each element is 8 bytes
	// so jump ahead 192 bytes for next i
	ADDQ $192, SI
	ADDQ $192, DI

	SUBQ $24, AX

remainder12: 
	ADDQ	$24, AX
	JE 		done

	SUBQ 	$16, AX
	JL 		remainder8

	// otherwise, there are 16 <= x < 24 items left

	// a[0] to a[7]
	// VMOVUPS 0(%rsi), %ymm0
	// VMOVUPS 0(%rdi), %ymm1
	// VADDPS %ymm0, %ymm1, %ymm0
	// VMOVUPS %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x06;       // vmovups (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x0f;       // vmovups (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf4; BYTE $0x58; BYTE $0xc0;       // vaddps %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x06;       // vmovups %ymm0,(%rsi)

	// a[8] to a[15]
	// VMOVUPS 64(%rsi), %ymm2
	// VMOVUPS 64(%rdi), %ymm3
	// VADDPS %ymm2, %ymm3, %ymm2
	// VMOVUPS %ymm2, 64(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x56; BYTE $0x40        // vmovups 0x40(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x5f; BYTE $0x40        // vmovups 0x40(%rdi),%ymm3
	BYTE $0xc5; BYTE $0xe4; BYTE $0x58; BYTE $0xd2                    // vaddps %ymm2,%ymm3,%ymm2
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x56; BYTE $0x40        // vmovups %ymm2,0x40(%rsi)


  	// update pointers (4*8 * 8) - 4*8 elements each time, each element is 8 bytes
	// so jump ahead 128 bytes for next i
	ADDQ $128, SI
	ADDQ $128, DI

	SUBQ $16, AX

remainder8:
	ADDQ 	$16, AX
	JE 		done

	/*
	here it gets a bit weird. We subtract 7 because of data alignment issues.
	Here's the specific issue: given a slice of 5 or 6 elements, the first 4 elements will not all be aligned to 32bytes.
	In fact, on Go 1.6.1 RC, a[0], a[1] will be aligned to 32 bytes, while a[2], a[3] will not. This will cause problems 
	with loading the code to the YMM register.

	The tests have been updated to take note of this.
	*/

	SUBQ 	$8, AX
	JL		remainder4

	// a[0] to a[7]
	// VMOVUPS 0(%rsi), %ymm0
	// VMOVUPS 0(%rdi), %ymm1
	// VADDPS %ymm0, %ymm1, %ymm0
	// VMOVUPS %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x06;       // vmovups (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x0f;       // vmovups (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf4; BYTE $0x58; BYTE $0xc0;       // vaddps %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x06;       // vmovups %ymm0,(%rsi)

	// otherwise, update pointers (1*8 * 8) - 1*8 elements each time, each element is 8 bytes
	// so jump ahead 64 bytes for next i
	ADDQ $64, SI
	ADDQ $64, DI

	SUBQ $8, AX

remainder4:
	ADDQ 	$8, AX
	JE 		done


remainder1:
	// copy into the appropriate registers
	// VMOVSS	(SI), X0
	// VMOVSS	(DI), X1
	// VADDSS	X0, X1, X0
	// VMOVSS	X0, (SI)
	BYTE $0xc5; BYTE $0xfa; BYTE $0x10; BYTE $0x06 			// vmovss (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xfa; BYTE $0x10; BYTE $0x0f			// vmovss (%rdi),%xmm1
	BYTE $0xc5; BYTE $0xf2; BYTE $0x58; BYTE $0xc0			// vaddss %xmm0,%xmm1,%xmm0
	BYTE $0xc5; BYTE $0xfa; BYTE $0x11; BYTE $0x06			// vmovss %xmm0,(%rsi)

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


