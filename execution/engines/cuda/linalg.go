package cuda

import (
	"context"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

var (
	_ tensor.BLA[float64, *dense.Dense[float64]] = &Engine[float64, *dense.Dense[float64]]{}
)

// this file implements all the tensor linalg engine interfaces

// MatVecMul performs matrix vector multiplication
func (e *Engine[DT, T]) MatVecMul(ctx context.Context, a, b, prealloc T, incr []DT) (err error) {
	if err := gctx.Handle(ctx); err != nil {
		return err
	}
	if incr != nil {
		panic("NYI incr")
	}

	tA := blas.Trans
	do := a.DataOrder()
	z := do.IsTransposed()

	m := a.Shape()[0]
	n := a.Shape()[1]

	var lda int
	switch {
	case do.IsRowMajor() && z:
		tA = blas.NoTrans
		lda = m
	case do.IsRowMajor() && !z:
		lda = n
		m, n = n, m
	case do.IsColMajor() && z:
		tA = blas.Trans
		lda = n
		m, n = n, m
	case do.IsColMajor() && !z:
		lda = m
		tA = blas.NoTrans
	}

	e.Signal()
	incX, incY := 1, 1 // step size

	// ASPIRATIONAL TODO: different incX and incY
	// TECHNICAL DEBT. TECHDEBT. TECH DEBT
	// Example use case:
	// log.Printf("a %v %v", ad.Strides(), ad.ostrides())
	// log.Printf("b %v", b.Strides())
	// incX := a.Strides()[0]
	// incY = b.Strides()[0]
	var z0 DT
	A := a.Data()
	X := b.Data()
	Y := prealloc.Data()
	switch any(z0).(type) {
	case float64:
		A, X, Y := any(A).([]float64), any(X).([]float64), any(Y).([]float64)
		alpha, beta := float64(1), float64(0)
		e.c.Do(func() error { e.b.Dgemv(tA, m, n, alpha, A, lda, X, incX, beta, Y, incY); return e.b.Err() })
	case float32:
		A, X, Y := any(A).([]float32), any(X).([]float32), any(Y).([]float32)
		alpha, beta := float32(1), float32(0)
		e.c.Do(func() error { e.b.Sgemv(tA, m, n, alpha, A, lda, X, incX, beta, Y, incY); return e.b.Err() })
	default:
		return errors.New("Unsupported Dtype")
	}
	return e.b.Err()
}

// MatMul performs matrix multiplication
func (e *Engine[DT, T]) MatMul(ctx context.Context, a, b, prealloc T, incr []DT) (err error) {
	if err := gctx.Handle(ctx); err != nil {
		return err
	}

	if incr != nil {
		panic("NYI: incr")
	}

	ado := a.DataOrder()
	bdo := b.DataOrder()
	if !ado.HasSameOrder(bdo) {
		return errors.Errorf("a does not have the same data order as b. a is %v. b is %v", a.DataOrder(), b.DataOrder())
	}

	// get result shapes. k is the shared dimension
	// a is (m, k)
	// b is (k, n)
	// c is (m, n)
	var m, n, k int
	m = a.Shape()[0]
	k = a.Shape()[1]
	n = b.Shape()[1]

	// // wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	// // lda in colmajor = number of rows;
	// // lda in row major = number of cols
	var lda, ldb, ldc int
	tA, tB := blas.Trans, blas.Trans
	za := ado.IsTransposed()
	zb := bdo.IsTransposed()

	// swapping around the operands if they are row major (a becomes b, and b becomes a)
	switch {
	case ado.IsColMajor() && bdo.IsColMajor() && !za && !zb:
		lda = m
		ldb = k
		ldc = prealloc.Shape()[0]
		tA, tB = blas.NoTrans, blas.NoTrans
	case ado.IsColMajor() && bdo.IsColMajor() && za && !zb:
		lda = k
		ldb = k
		ldc = prealloc.Shape()[0]
		tA, tB = blas.Trans, blas.NoTrans
	case ado.IsColMajor() && bdo.IsColMajor() && za && zb:
		lda = k
		ldb = n
		ldc = prealloc.Shape()[0]
		tA, tB = blas.Trans, blas.Trans
	case ado.IsColMajor() && bdo.IsColMajor() && !za && zb:
		lda = m
		ldb = n
		ldc = prealloc.Shape()[0]
		tA, tB = blas.NoTrans, blas.Trans
	case ado.IsRowMajor() && bdo.IsRowMajor() && !za && !zb:
		lda = k
		ldb = n
		ldc = prealloc.Shape()[1]
		tA, tB = blas.NoTrans, blas.NoTrans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		a, b = b, a
	case ado.IsRowMajor() && bdo.IsRowMajor() && za && !zb:
		lda = m
		ldb = n
		ldc = prealloc.Shape()[1]
		tA, tB = blas.Trans, blas.NoTrans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		a, b = b, a
	case ado.IsRowMajor() && bdo.IsRowMajor() && za && zb:
		lda = m
		ldb = k
		ldc = prealloc.Shape()[1]
		tA, tB = blas.Trans, blas.Trans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		a, b = b, a
	case ado.IsRowMajor() && bdo.IsRowMajor() && !za && zb:
		lda = k
		ldb = k
		ldc = prealloc.Shape()[1]
		tA, tB = blas.NoTrans, blas.Trans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		a, b = b, a

	default:
		panic("Unreachable")
	}
	e.Signal()

	A := a.Data()
	B := b.Data()
	C := prealloc.Data()
	var z DT
	switch any(z).(type) {
	case float64:
		A, B, C := any(A).([]float64), any(B).([]float64), any(C).([]float64)
		alpha, beta := float64(1), float64(0)

		e.c.Do(func() error { e.b.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); return nil })

	case float32:
		A, B, C := any(A).([]float32), any(B).([]float32), any(C).([]float32)
		alpha, beta := float32(1), float32(0)
		e.c.Do(func() error { e.b.Sgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); return nil })
	default:
		return errors.Errorf("Unsupported Dtype %v", a.Dtype())
	}
	e.Signal()

	return e.b.Err()
}

// Outer performs outer product (kronecker) multiplication
func (e *Engine[DT, T]) Outer(ctx context.Context, a, b, prealloc T, incr []DT) (err error) {
	if err := gctx.Handle(ctx); err != nil {
		return err
	}
	if incr == nil {
		panic("NYI incr")
	}

	m := a.Size()
	n := b.Size()
	pdo := prealloc.DataOrder()

	var lda int
	switch {
	case pdo.IsColMajor():
		lda = prealloc.Shape()[0]
	case pdo.IsRowMajor():
		aShape := a.Shape().Clone()
		bShape := b.Shape().Clone()
		if err = a.Reshape(aShape[0], 1); err != nil {
			return err
		}
		if err = b.Reshape(1, bShape[0]); err != nil {
			return err
		}

		if err = e.MatMul(ctx, a, b, prealloc, nil); err != nil {
			return err
		}

		if err = b.Reshape(bShape...); err != nil {
			return
		}
		if err = a.Reshape(aShape...); err != nil {
			return
		}
		return nil
	}

	e.Signal()
	x := a.Data()
	y := b.Data()
	A := prealloc.Data()
	incX, incY := 1, 1
	var z DT
	switch any(z).(type) {
	case float64:
		x, y, A := any(x).([]float64), any(y).([]float64), any(A).([]float64)
		alpha := float64(1)
		e.c.Do(func() error { e.b.Dger(m, n, alpha, x, incX, y, incY, A, lda); return nil })
	case float32:
		x, y, A := any(x).([]float32), any(y).([]float32), any(A).([]float32)
		alpha := float32(1)
		e.c.Do(func() error { e.b.Sger(m, n, alpha, x, incX, y, incY, A, lda); return nil })
	}
	return e.b.Err()
}
