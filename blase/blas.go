// Package blase is a thin wrapper over Gonum's BLAS interface that provides a queue
// so that cgo calls are batched. This package was created so MKL usage can be improved.
//
// Any cblas function that is not handled will result in the blocking BLAS call being called
package blase

/*
#include <stdint.h>
#include <stdio.h>
#include "cblas.h"
#include "work.h"

// useful to help print stuff to see if things are correct
void prrintfnargs(struct fnargs* args){
	printf("HELLO\n");
	printf("fn: %d\n", args->fn);
	printf("o: %d\n", args->order);
	printf("tA: %d\n",args->tA);
	printf("tB: %d\n",args->tB);
	printf("----\n");
	// printf("a0: %f\n", (double*)args->a0);
	// printf("a1: %f\n", (double*)args->a1);
	// printf("a2: %f\n", (double*)args->a2);
	// printf("a3: %f\n", (double*)args->a3);
	printf("----\n");
	printf("i0: %d\n", args->i0);
	printf("i1: %d\n", args->i1);
	printf("i2: %d\n", args->i2);
	printf("i3: %d\n", args->i3);
	printf("i4: %d\n", args->i4);
	printf("i5: %d\n", args->i5);
	printf("----\n");
	printf("d0: %f\n", args->d0);
	printf("d1: %f\n", args->d1);
	printf("d2: %f\n", args->d2);
	printf("d3: %f\n", args->d3);
	printf("=========\n");
}
*/
import "C"

import (
	"unsafe"

	"github.com/gonum/blas"
)

const rowMajor = 101 // rowMajor and rowMajor ONLY

func (ctx *context) Dgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	fn := &fnargs{
		fn:    C.cblasFn(fn_cblas_dgemm),
		order: C.cblas_order(rowMajor),
		tA:    C.cblas_transpose(tA),
		tB:    C.cblas_transpose(tB),
		i0:    C.int(m),
		i1:    C.int(n),
		i2:    C.int(k),
		d0:    C.double(alpha),
		a0:    uintptr(unsafe.Pointer(&a[0])),
		i3:    C.int(lda),
		a1:    uintptr(unsafe.Pointer(&b[0])),
		i4:    C.int(ldb),
		d1:    C.double(beta),
		a2:    uintptr(unsafe.Pointer(&c[0])),
		i5:    C.int(ldc),
	}
	call := call{args: fn, blocking: false}
	ctx.enqueue(call)
}

func (ctx *context) Dgemv(tA blas.Transpose, m int, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	fn := &fnargs{
		fn:    C.cblasFn(fn_cblas_dgemv),
		order: C.cblas_order(rowMajor),
		tA:    C.cblas_transpose(tA),
		i0:    C.int(m),
		i1:    C.int(m),
		d0:    C.double(alpha),
		a0:    uintptr(unsafe.Pointer(&a[0])),
		i2:    C.int(lda),
		a1:    uintptr(unsafe.Pointer(&x[0])),
		i3:    C.int(incX),
		d1:    C.double(beta),
		a2:    uintptr(unsafe.Pointer(&y[0])),
		i4:    C.int(incY),
	}

	// Cs := fn.toCStruct()
	// C.prrintfnargs(&Cs)
	// fmt.Println("Sleeping")
	// time.Sleep(10)
	// fmt.Println("Slept")
	// ctx.Implementation.Dgemv(tA, m, n, alpha, a, lda, x, incX, beta, y, incY)

	call := call{args: fn, blocking: false}
	ctx.enqueue(call)
}
