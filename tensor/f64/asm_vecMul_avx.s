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


// func add64(a, b []float64)
TEXT ·vecMul(SB), 7, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI  		// use detination index register for this

	MOVQ a_len+8(FP), AX 			// len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap
	MOVQ b_len+32(FP), BX			// len(b) into BX
	MOVQ AX, AX						// len(a) into AX for working purposes

	// check if they're the same length
	CMPQ AX, BX
	JNE panic						// jump to panic if not the same length. TOOD: return bloody errors

	// each ymm register can take up to 4 float64s. 
	// There are 8 ymm registers (8 pairs to do addition) available (TODO: check how to access the other 8 ymm registers without fucking things up)
	// Therefore a total of 16 elements can be processed at a time

	SUBQ $16, AX
	JL remainder



loop:
	// a[0] to a[3]
	// VMOVUPD 0(%rsi), %ymm0
	// VMOVUPD 0(%rdi), %ymm1
	// VADDPD %ymm0, %ymm1, %ymm0
	// VMOVUPD %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x06;       // vmovupd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x0f;       // vmovupd (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf5; BYTE $0x59; BYTE $0xc0;       // vaddpd %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;       // vmovupd %ymm0,(%rsi)

	// a[3] to a[7]
	// VMOVUPD 32(%rsi), %ymm2
	// VMOVUPD 32(%rdi), %ymm3
	// VADDPD %ymm2, %ymm3, %ymm2
	// VMOVUPD %ymm2, 32(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x56; BYTE $0x20        // vmovupd 0x20(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x5f; BYTE $0x20        // vmovupd 0x20(%rdi),%ymm3
	BYTE $0xc5; BYTE $0xe5; BYTE $0x59; BYTE $0xd2                    // vaddpd %ymm2,%ymm3,%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x20        // vmovupd %ymm2,0x20(%rsi)

	// a[8] to a[11]
	// VMOVUPD 64(%rsi), %ymm4
	// VMOVUPD 64(%rdi), %ymm5
	// VADDPD %ymm4, %ymm5, %ymm4
	// VMOVUPD %ymm4, 64(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x66; BYTE $0x40       // vmovupd 0x40(%rsi),%ymm4
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x6f; BYTE $0x40       // vmovupd 0x40(%rdi),%ymm5
	BYTE $0xc5; BYTE $0xd5; BYTE $0x59; BYTE $0xe4;                  // vaddpd %ymm4,%ymm5,%ymm4
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x66; BYTE $0x40       // vmovupd %ymm4,0x40(%rsi)

  	// a[12] to a[15]
	// VMOVUPD 96(%rsi), %ymm6
	// VMOVUPD 96(%rdi), %ymm7
	// VADDPD %ymm6, %ymm7, %ymm6
	// VMOVUPD %ymm6, 96(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x76; BYTE $0x60        // vmovupd 0x60(%rsi),%ymm6
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x7f; BYTE $0x60        // vmovupd 0x60(%rdi),%ymm7
	BYTE $0xc5; BYTE $0xc5; BYTE $0x59; BYTE $0xf6;                   // vaddpd %ymm6,%ymm7,%ymm6
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x76; BYTE $0x60        // vmovupd %ymm6,0x60(%rsi)



	// update pointers (4*4 * 8) - 4*4 elements each time, each element is 8 bytes
	// so jump ahead 128 bytes for next i
	ADDQ $128, SI
	ADDQ $128, DI

	SUBQ $16, AX
	JGE loop

remainder:
	ADDQ 	$16, AX
	JE 		done

	SUBQ 	$12, AX
	JL 		remainder12

	// otherwise,
	// there are 12 <= x < 16 items left!

	// a[0] to a[3]
	// VMOVUPD 0(%rsi), %ymm0
	// VMOVUPD 0(%rdi), %ymm1
	// VADDPD %ymm0, %ymm1, %ymm0
	// VMOVUPD %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x06;       // vmovupd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x0f;       // vmovupd (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf5; BYTE $0x59; BYTE $0xc0;       // vaddpd %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;       // vmovupd %ymm0,(%rsi)

	// a[3] to a[7]
	// VMOVUPD 32(%rsi), %ymm2
	// VMOVUPD 32(%rdi), %ymm3
	// VADDPD %ymm2, %ymm3, %ymm2
	// VMOVUPD %ymm2, 32(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x56; BYTE $0x20        // vmovupd 0x20(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x5f; BYTE $0x20        // vmovupd 0x20(%rdi),%ymm3
	BYTE $0xc5; BYTE $0xe5; BYTE $0x59; BYTE $0xd2                    // vaddpd %ymm2,%ymm3,%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x20        // vmovupd %ymm2,0x20(%rsi)

	// a[8] to a[11]
	// VMOVUPD 64(%rsi), %ymm4
	// VMOVUPD 64(%rdi), %ymm5
	// VADDPD %ymm4, %ymm5, %ymm4
	// VMOVUPD %ymm4, 64(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x66; BYTE $0x40       // vmovupd 0x40(%rsi),%ymm4
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x6f; BYTE $0x40       // vmovupd 0x40(%rdi),%ymm5
	BYTE $0xc5; BYTE $0xd5; BYTE $0x59; BYTE $0xe4;                  // vaddpd %ymm4,%ymm5,%ymm4
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x66; BYTE $0x40       // vmovupd %ymm4,0x40(%rsi)


  	// update pointers (3*4 * 8) - 3*4 elements each time, each element is 8 bytes
	// so jump ahead 128 bytes for next i
	ADDQ $96, SI
	ADDQ $96, DI

	SUBQ $12, AX

remainder12: 
	ADDQ	$12, AX
	JE 		done

	SUBQ 	$8, AX
	JL 		remainder8

	// otherwise, there are 8 <= x < 12 items left

	// a[0] to a[3]
	// VMOVUPD 0(%rsi), %ymm0
	// VMOVUPD 0(%rdi), %ymm1
	// VADDPD %ymm0, %ymm1, %ymm0
	// VMOVUPD %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x06;       // vmovupd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x0f;       // vmovupd (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf5; BYTE $0x59; BYTE $0xc0;       // vaddpd %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;       // vmovupd %ymm0,(%rsi)

	// a[3] to a[7]
	// VMOVUPD 32(%rsi), %ymm2
	// VMOVUPD 32(%rdi), %ymm3
	// VADDPD %ymm2, %ymm3, %ymm2
	// VMOVUPD %ymm2, 32(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x56; BYTE $0x20        // vmovupd 0x20(%rsi),%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x5f; BYTE $0x20        // vmovupd 0x20(%rdi),%ymm3
	BYTE $0xc5; BYTE $0xe5; BYTE $0x59; BYTE $0xd2                    // vaddpd %ymm2,%ymm3,%ymm2
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x56; BYTE $0x20        // vmovupd %ymm2,0x20(%rsi)


  	// update pointers (2*4 * 8) - 2*4 elements each time, each element is 8 bytes
	// so jump ahead 64 bytes for next i
	ADDQ $64, SI
	ADDQ $64, DI

	SUBQ $8, AX

remainder8:
	ADDQ 	$8, AX
	JE 		done

	/*
	here it gets a bit weird. We subtract 7 because of data alignment issues.
	Here's the specific issue: given a slice of 5 or 6 elements, the first 4 elements will not all be aligned to 32bytes.
	In fact, on Go 1.6.1 RC, a[0], a[1] will be aligned to 32 bytes, while a[2], a[3] will not. This will cause problems 
	with loading the code to the YMM register.

	The tests have been updated to take note of this.
	*/

	SUBQ 	$4, AX
	JL		remainder4

	// a[0] to a[3]
	// VMOVUPD 0(%rsi), %ymm0
	// VMOVUPD 0(%rdi), %ymm1
	// VADDPD %ymm0, %ymm1, %ymm0
	// VMOVUPD %ymm0, 0(%rsi)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x06;       // vmovupd (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x0f;       // vmovupd (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xf5; BYTE $0x59; BYTE $0xc0;       // vaddpd %ymm0,%ymm1,%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06;       // vmovupd %ymm0,(%rsi)

	// otherwise, update pointers (1*4 * 8) - 1*4 elements each time, each element is 8 bytes
	// so jump ahead 64 bytes for next i
	ADDQ $32, SI
	ADDQ $32, DI

	SUBQ $4, AX

remainder4:
	ADDQ 	$4, AX
	JE 		done


remainder1:
	// copy into the appropriate registers
	// VMOVSD	(SI), X0
	// VMOVSD	(DI), X1
	// VADDSD	X0, X1, X0
	// VMOVSD	X0, (SI)
	BYTE $0xc5; BYTE $0xfb; BYTE $0x10; BYTE $0x06 
	BYTE $0xc5; BYTE $0xfb; BYTE $0x10; BYTE $0x0f
	BYTE $0xc5; BYTE $0xf3; BYTE $0x59; BYTE $0xc0
	BYTE $0xc5; BYTE $0xfb; BYTE $0x11; BYTE $0x06

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


