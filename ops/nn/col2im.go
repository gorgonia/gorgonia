package nnops

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/internal"
	"gorgonia.org/gorgonia/internal/kernels"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"

	"gorgonia.org/tensor"
)

type im2col[DT tensor.Num, T values.Value[DT]] struct {
	h, w                 int // kernel height and  width
	padH, padW           int
	strideH, strideW     int
	dilationH, dilationW int
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
	panic("NYI")
}

func (op im2col[DT, T]) PreallocDo(ctx context.Context, prealloc T, inputs ...T) (retVal T, err error) {
	return op.do(ctx, prealloc, inputs[0])
}

func (op im2col[DT, T]) do(ctx context.Context, prealloc, input T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

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
	inRowStride := inputStrides[2]

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

		ChanStride:  chanStride,
		InRowStride: inRowStride,

		RetHeight: retH,
		RetWidth:  retW,
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
		go kernels.Im2Col(kernelParams, retH, retW, imData[imStart:imEnd], colData[colStart:colEnd], &wg, workers)
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
