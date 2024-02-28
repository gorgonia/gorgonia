package nnops

import (
	"context"
	"fmt"
	"runtime"
	"runtime/trace"
	"sync"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/internal"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/internal/kernels"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"

	"gorgonia.org/tensor"
)

type im2col[DT any, T values.Value[DT]] struct {
	h, w                 int // kernel height and  width
	padH, padW           int
	strideH, strideW     int
	dilationH, dilationW int
}

// Im2Col creates a PreallocOp that converts a BCHW image block to column. The kernel, pad and stride parameter must be shape of size 2, no more no less.
// This poor naming scheme clearly comes from matlab
func Im2Col[DT any, T values.Value[DT]](kernel, pad, stride, dilation shapes.Shape) (retVal ops.PreallocOp[DT, T], err error) {
	if kernel.Dims() != 2 {
		return nil, errors.Errorf("kernel shape is supposed to have a dim of 2")
	}
	if pad.Dims() != 2 {
		return nil, errors.Errorf("pad is supposed to have a dim of 2")
	}
	if stride.Dims() != 2 {
		return nil, errors.Errorf("strides is supposed to have a dim of 2")
	}
	if dilation.Dims() != 2 {
		return nil, errors.Errorf("dilation is supposed to have a dim of 2")
	}

	if kernel[0] <= 0 || kernel[1] <= 0 {
		return nil, errors.Errorf("cannot have negative or 0 in kernel shape")
	}

	if stride[0] <= 0 || stride[1] <= 0 {
		return nil, errors.Errorf("cannot have negative or 0 in stride: %v", stride)
	}

	if pad[0] < 0 || pad[1] < 0 {
		return nil, errors.Errorf("cannot have negative padding")
	}

	if dilation[0] <= 0 || dilation[1] <= 0 {
		return nil, errors.Errorf("cannot have negative or 0 in dilation. %v", dilation)
	}

	return im2col[DT, T]{
		h: kernel[0], w: kernel[1],
		padH: pad[0], padW: pad[1],
		strideH: stride[0], strideW: stride[1],
		dilationH: dilation[0], dilationW: dilation[1],
	}, nil
}

func (op im2col[DT, T]) Arity() int { return 1 }

// im2col :: (Floats a) ⇒ Tensor a →  Tensor a
func (op im2col[DT, T]) Type() hm.Type {
	t := types.MakeTensorType(4, hm.TypeVariable('a'))
	return hm.NewFnType(t, t)
}

func (op im2col[DT, T]) ShapeExpr() shapes.Expr {
	b := shapes.Var('b')
	c := shapes.Var('c')
	h := shapes.Var('h')
	w := shapes.Var('w')
	in := shapes.Abstract{b, c, h, w}

	//TODO: h2 =  (h+2*op.padH-(op.dilationH*(op.h-1)+1))/op.strideH + 1
	h2 := shapes.BinOp{
		Op: shapes.Add,
		A: shapes.E2{shapes.BinOp{
			Op: shapes.Div,
			A: shapes.E2{shapes.BinOp{
				Op: shapes.Sub,
				A: shapes.E2{shapes.BinOp{
					Op: shapes.Add,
					A:  h,
					B: shapes.E2{shapes.BinOp{
						Op: shapes.Mul,
						A:  shapes.Size(2),
						B:  shapes.Size(op.padH),
					}},
				}},
				B: shapes.Size(op.dilationH*(op.h-1) + 1),
			}},
			B: shapes.Size(op.strideH),
		}},
		B: shapes.Size(1),
	}

	//  w2 = (w+2*op.padW-(op.dilationW*(op.w-1)+1))/op.strideW + 1
	w2 := shapes.BinOp{
		Op: shapes.Add,
		A: shapes.E2{shapes.BinOp{
			Op: shapes.Div,
			A: shapes.E2{shapes.BinOp{
				Op: shapes.Sub,
				A: shapes.E2{shapes.BinOp{
					Op: shapes.Add,
					A:  w,
					B: shapes.E2{shapes.BinOp{
						Op: shapes.Mul,
						A:  shapes.Size(2),
						B:  shapes.Size(op.padW),
					}},
				}},
				B: shapes.Size(op.dilationW*(op.w-1) + 1),
			}},
			B: shapes.Size(op.strideW),
		}},
		B: shapes.Size(1),
	}
	c2 := shapes.BinOp{
		Op: shapes.Mul,
		A:  c,
		B:  shapes.Size(op.w * op.h),
	}

	out := shapes.Abstract{b, h2, w2, c2}
	return shapes.MakeArrow(in, out)
}

func (op im2col[DT, T]) Do(ctx context.Context, inputs ...T) (retVal T, err error) {
	// compute expected shape
	var expExpr shapes.Expr
	var expShape shapes.Shape
	var ok bool
	x := inputs[0]
	expr := op.ShapeExpr()
	if expExpr, err = shapes.InferApp(expr, x.Shape()); err != nil {
		return retVal, errors.Wrapf(err, "Failed to infer shape for %v", op)
	}
	if expShape, ok = expExpr.(shapes.Shape); !ok {
		return retVal, errors.Wrapf(err, "Failed to infer shape for %v : %v @ %v → %v (not a shape)", op, expr, x.Shape(), expExpr)
	}

	// now we got the expected shape, we use the engine to make a new retVal for us
	e := x.Engine().Workhorse()
	var prepper tensor.FuncOptHandler[DT]
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	ret, _, err := prepper.HandleFuncOpts(x, expShape)
	if err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	retVal = ret.(T)

	// finally we actually do the im2col op
	return op.do(ctx, retVal, x)
}

func (op im2col[DT, T]) PreallocDo(ctx context.Context, prealloc T, inputs ...T) (retVal T, err error) {
	return op.do(ctx, prealloc, inputs[0])
}

func (op im2col[DT, T]) do(ctx context.Context, prealloc, input T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}
	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	s := input.Shape()
	r := prealloc.Shape()
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]
	retH, retW := r[1], r[2]

	inputStrides := input.Strides()
	batchStrideIm := inputStrides[0]
	batchStrideCol := prealloc.Strides()[0]
	chanStride := h * w

	var wg sync.WaitGroup
	workers := make(chan struct{}, runtime.NumCPU())
	imData := input.Data()
	colData := prealloc.Data()

	kernelParams := kernels.Im2ColOp{
		H: op.h, W: op.w,
		PadH: op.padH, PadW: op.padW,
		StrideH: op.strideH, StrideW: op.strideW,
		DilationH: op.dilationH, DilationW: op.dilationW,

		Chans:  c,
		Height: h,
		Width:  w,

		ChanStride: chanStride,

		RetH: retH,
		RetW: retW,
	}

	for i := 0; i < b; i++ {
		imStart := i * batchStrideIm
		colStart := i * batchStrideCol
		imEnd := imStart + batchStrideIm
		colEnd := colStart + batchStrideCol

		if imEnd >= len(imData) {
			imEnd = len(imData)
		}
		if colEnd >= len(colData) {
			colEnd = len(colData)
		}
		wg.Add(1)
		go kernels.Im2Col(ctx2, kernelParams, imData[imStart:imEnd], colData[colStart:colEnd], &wg, workers)
	}
	wg.Wait()
	return prealloc, nil
}

func (op im2col[DT, T]) ReturnsPtr() bool     { return false }
func (op im2col[DT, T]) CallsExtern() bool    { return false }
func (op im2col[DT, T]) OverwritesInput() int { return -1 }

func (op im2col[DT, T]) String() string {
	return fmt.Sprintf("im2col<(%d,%d), (%d, %d), (%d,%d) (%d, %d)>", op.h, op.w, op.padH, op.padW, op.strideH, op.strideW, op.dilationH, op.dilationW)
}

func (op im2col[DT, T]) DiffWRT(i int) []bool { return []bool{true} }
