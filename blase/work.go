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
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/netlib/blas/netlib"
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
	WorkAvailable() <-chan struct{}
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
	netlib.Implementation

	workAvailable chan struct{}
	work          chan call

	fns   []C.struct_fnargs
	queue []call
}

func newContext() *context {
	return &context{
		workAvailable: make(chan struct{}, 1),
		work:          make(chan call, workbufLen),

		fns:   make([]C.struct_fnargs, workbufLen, workbufLen),
		queue: make([]call, 0, workbufLen),
	}
}

func (ctx *context) enqueue(c call) {
	ctx.work <- c
	select {
	case ctx.workAvailable <- struct{}{}:
	default:
	}
	if c.blocking {
		// do something
		ctx.DoWork()
	}
}

// DoWork retrieves as many work items as possible, puts them into a queue, and then processes the queue.
// The function may return without doing any work.
func (ctx *context) DoWork() {
	for {
		select {
		case w := <-ctx.work:
			ctx.queue = append(ctx.queue, w)
		default:
			return
		}

		blocking := ctx.queue[len(ctx.queue)-1].blocking
	enqueue:
		for len(ctx.queue) < cap(ctx.queue) && !blocking {
			select {
			case w := <-ctx.work:
				ctx.queue = append(ctx.queue, w)
				blocking = ctx.queue[len(ctx.queue)-1].blocking
			default:
				break enqueue

			}

			for i, c := range ctx.queue {
				ctx.fns[i] = *(*C.struct_fnargs)(unsafe.Pointer(c.args))
			}
			C.process(&ctx.fns[0], C.int(len(ctx.queue)))

			// clear queue
			ctx.queue = ctx.queue[:0]
		}
	}
}

// WorkAvailable is the channel which users should subscribe to to know if there is work incoming.
func (ctx *context) WorkAvailable() <-chan struct{} { return ctx.workAvailable }

// String implements runtime.Stringer and fmt.Stringer. It returns the name of the BLAS implementation.
func (ctxt *context) String() string { return "Blase" }
