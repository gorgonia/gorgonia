package blase

/*
#cgo CFLAGS: -g -O3 -std=gnu99

#include <stdio.h>
#include <stdint.h>
#include "work.h"
#include "cblas.h"

uintptr_t process(struct fnargs* fa, int count) {
	uintptr_t ret;

	// printf("How much work: %d\n", count);

	ret = processFn(&fa[0]);
	if (count > 1) {
		ret = processFn(&fa[1]);
	}
	if (count > 2) {
		ret = processFn(&fa[2]);
	}

	return ret;
}

*/
import "C"
import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/cgo"
)

var impl = newContext()

var (
	_ blas.Float32    = impl
	_ blas.Float64    = impl
	_ blas.Complex64  = impl
	_ blas.Complex128 = impl
)

// Implementation returns a BLAS implementation that implements Float32, Float64, Complex64 and Complex128
func Implementation() *context { return impl }

const workbufLen int = 3

//A Worker is a BLAS implementation that reports back if there is anything in the queue (WorkAvailable())
// and a way to flush that queue
type Worker interface {
	WorkAvailable() int
	DoWork()
}

type call struct {
	args *fnargs

	/*
		this flag only applies to any BLAS function that has a return value:
			cblas_sdsdot
			cblas_dsdot
			cblas_sdot
			cblas_ddot
			cblas_cdotu_sub
			cblas_snrm2
			cblas_sasum
			cblas_dnrm2
			cblas_dasum
			cblas_scnrm2
			cblas_scasum
			cblas_dznrm2
			cblas_dzasum

		These are routines that are recast as functions
			cblas_cdotc_sub
			cblas_zdotu_sub
			cblas_zdotc_sub

		Not sure about these (they return CBLAS_INDEX)
			cblas_isamax
			cblas_idamax
			cblas_icamax
			cblas_izamax

		For the rest of the BLAS routines (i.e. they return void), don't set the blocking
	*/
	blocking bool
}

type context struct {
	cgo.Implementation

	fns   []C.struct_fnargs
	queue []call
}

func newContext() *context {
	return &context{
		fns:   make([]C.struct_fnargs, workbufLen+1, workbufLen+1),
		queue: make([]call, 0, workbufLen+1), // the extra 1 is for cases where the queue is full, and a blocking call comes in
	}
}

func (ctx *context) enqueue(c call) {
	if len(ctx.queue) == workbufLen-1 || c.blocking {
		ctx.queue = append(ctx.queue, c)
		ctx.DoWork()
		return
	}
	ctx.queue = append(ctx.queue, c)
	return
}

// DoWork basically drops everything and just performs the work
func (ctx *context) DoWork() {
	// runtime.LockOSThread()
	// defer runtime.UnlockOSThread()
	for i, c := range ctx.queue {
		fn := c.args.toCStruct()
		ctx.fns[i] = fn
	}
	C.process(&ctx.fns[0], C.int(len(ctx.queue)))

	// cleanup - clear queue
	ctx.queue = ctx.queue[:0]
}

func (ctx *context) WorkAvailable() int { return len(ctx.queue) }

func (ctxt *context) String() string { return "Blase" }
