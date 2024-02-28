package nnops

import (
	"context"
	"runtime"
	"runtime/trace"
	"sync"

	"gorgonia.org/gorgonia/internal"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/internal/kernels"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type col2im[DT any, T values.Value[DT]] struct {
	input shapes.Shape
	im2col[DT, T]
}

func (op col2im[DT, T]) Arity() int { return 1 }

func (op col2im[DT, T]) ShapeExpr() shapes.Expr {
	return shapes.MakeArrow(shapes.Var('a'), op.input)
}

func (op col2im[DT, T]) Do(ctx context.Context, inputs ...T) (retVal T, err error) {
	x := inputs[0]
	e := x.Engine().Workhorse()
	var prepper tensor.FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	ret, _, err := prepper.HandleFuncOpts(x, op.input)
	if err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	retVal = ret.(T)
	return op.do(ctx, retVal, x)
}

func (op col2im[DT, T]) PreallocDo(ctx context.Context, prealloc T, inputs ...T) (retVal T, err error) {
	return op.do(ctx, prealloc, inputs[0])
}

func (op col2im[DT, T]) do(ctx context.Context, prealloc, input T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}
	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	b := op.input[0]
	c := op.input[1]
	retH := op.input[2]
	retW := op.input[3]
	batchStrideIm := c * retH * retW

	s := input.Shape()
	h := s[1]
	w := s[2]
	chanStride := retH * retW
	batchStrideCol := h * w * s[3]

	var imStart, imEnd, colStart, colEnd int
	imEnd = imStart + batchStrideIm
	colEnd = colStart + batchStrideCol

	var wg sync.WaitGroup
	workers := make(chan struct{}, runtime.NumCPU())
	colData := input.Data()
	imData := prealloc.Data()
	// No, you are not reading this wrong. This is correct. It's using the kernels.Im2Col param struct
	kernelParams := kernels.Im2ColOp{
		H: op.h, W: op.w,
		PadH: op.padH, PadW: op.padW,
		StrideH: op.strideH, StrideW: op.strideW,
		DilationH: op.dilationH, DilationW: op.dilationW,

		Chans:      c,
		RetH:       retH,
		RetW:       retW,
		ChanStride: chanStride,
		Height:     h,
		Width:      w,
	}

	for i := 0; i < b; i++ {
		wg.Add(1)
		go kernels.Col2Im(ctx2, kernelParams, c, retH, retW, chanStride, h, w, colData[colStart:colEnd], imData[imStart:imEnd], &wg, workers)

		colStart += batchStrideCol
		colEnd += batchStrideCol

		imStart += batchStrideIm
		imEnd += batchStrideIm

		if imEnd > len(imData) {
			imEnd = len(imData)
		}
		if colEnd > len(colData) {
			colEnd = len(colData)
		}
	}
}
