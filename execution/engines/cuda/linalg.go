package cuda

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
	"gorgonia.org/tensor"
)

var (
	_ tensor.MatVecMuler = &Engine{}
	_ tensor.MatMuler    = &Engine{}
	_ tensor.OuterProder = &Engine{}
)

// this file implements all the tensor linalg engine interfaces

func (e *Engine) checkThreeFloat(a, b, ret tensor.Tensor) (ad, bd, retVal *tensor.Dense, err error) {
	if /*a.IsNativelyAccessible() &&*/ !a.IsManuallyManaged() {
		return nil, nil, nil, errors.New("CUDA Engine only takes non-natively accessible memory (memory on graphics cards). a isn't.")
	}

	if /* b.IsNativelyAccessible() && */ !b.IsManuallyManaged() {
		return nil, nil, nil, errors.New("CUDA Engine only takes non-natively accessible memory (memory on graphics cards). b isn't")
	}

	if /* ret.IsNativelyAccessible() && */ !ret.IsManuallyManaged() {
		return nil, nil, nil, errors.New("CUDA Engine only takes non-natively accessible memory (memory on graphics cards). ret isn't")
	}

	if a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}
	var ok bool
	if ad, ok = a.(*tensor.Dense); !ok {
		return nil, nil, nil, errors.New("Expected a to be a *tensor.Dense")
	}
	if bd, ok = b.(*tensor.Dense); !ok {
		return nil, nil, nil, errors.New("Expected b to be a *tensor.Dense")
	}
	if retVal, ok = ret.(*tensor.Dense); !ok {
		return nil, nil, nil, errors.New("Expected ret to be a *tensor.Dense")
	}
	return
}

// MatVecMul performs matrix vector multiplication
func (e *Engine) MatVecMul(a, b, prealloc tensor.Tensor) (err error) {
	var ad, bd, pd *tensor.Dense
	if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
		return errors.Wrapf(err, "MatVecMul failed pre check")
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

	e.c.DoWork()
	incX, incY := 1, 1 // step size

	// ASPIRATIONAL TODO: different incX and incY
	// TECHNICAL DEBT. TECHDEBT. TECH DEBT
	// Example use case:
	// log.Printf("a %v %v", ad.Strides(), ad.ostrides())
	// log.Printf("b %v", b.Strides())
	// incX := a.Strides()[0]
	// incY = b.Strides()[0]

	switch ad.Dtype() {
	case tensor.Float64:
		A := ad.Float64s()
		X := bd.Float64s()
		Y := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		e.c.DoWork()
		e.c.Do(func() error { e.b.Dgemv(tA, m, n, alpha, A, lda, X, incX, beta, Y, incY); return e.b.Err() })
	case tensor.Float32:
		A := ad.Float32s()
		X := bd.Float32s()
		Y := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		e.c.DoWork()
		e.c.Do(func() error { e.b.Sgemv(tA, m, n, alpha, A, lda, X, incX, beta, Y, incY); return e.b.Err() })
	default:
		return errors.New("Unsupported Dtype")
	}
	return e.b.Err()
}

// MatMul performs matrix multiplication
func (e *Engine) MatMul(a, b, prealloc tensor.Tensor) (err error) {
	var ad, bd, pd *tensor.Dense
	if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
		return errors.Wrapf(err, "MatVecMul failed pre check")
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
	m = ad.Shape()[0]
	k = ad.Shape()[1]
	n = bd.Shape()[1]

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
		ad, bd = bd, ad
	case ado.IsRowMajor() && bdo.IsRowMajor() && za && !zb:
		lda = m
		ldb = n
		ldc = prealloc.Shape()[1]
		tA, tB = blas.Trans, blas.NoTrans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		ad, bd = bd, ad
	case ado.IsRowMajor() && bdo.IsRowMajor() && za && zb:
		lda = m
		ldb = k
		ldc = prealloc.Shape()[1]
		tA, tB = blas.Trans, blas.Trans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		ad, bd = bd, ad
	case ado.IsRowMajor() && bdo.IsRowMajor() && !za && zb:
		lda = k
		ldb = k
		ldc = prealloc.Shape()[1]
		tA, tB = blas.NoTrans, blas.Trans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		ad, bd = bd, ad

	default:
		panic("Unreachable")
	}

	e.c.DoWork()
	switch ad.Dtype() {
	case tensor.Float64:
		A := ad.Float64s()
		B := bd.Float64s()
		C := pd.Float64s()
		alpha, beta := float64(1), float64(0)

		e.c.Do(func() error { e.b.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); return nil })

	case tensor.Float32:
		A := ad.Float32s()
		B := bd.Float32s()
		C := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		e.c.Do(func() error { e.b.Sgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); return nil })
	default:
		return errors.Errorf("Unsupported Dtype %v", ad.Dtype())
	}

	return e.b.Err()
}

// Outer performs outer product (kronecker) multiplication
func (e *Engine) Outer(a, b, prealloc tensor.Tensor) (err error) {
	var ad, bd, pd *tensor.Dense
	if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
		return errors.Wrapf(err, "MatVecMul failed pre check")
	}
	m := ad.Size()
	n := bd.Size()
	pdo := pd.DataOrder()

	var lda int
	switch {
	case pdo.IsColMajor():
		lda = pd.Shape()[0]
	case pdo.IsRowMajor():
		aShape := a.Shape().Clone()
		bShape := b.Shape().Clone()
		if err = a.Reshape(aShape[0], 1); err != nil {
			return err
		}
		if err = b.Reshape(1, bShape[0]); err != nil {
			return err
		}

		if err = e.MatMul(a, b, prealloc); err != nil {
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

	e.c.DoWork()
	incX, incY := 1, 1
	switch ad.Dtype() {
	case tensor.Float64:
		x := ad.Float64s()
		y := bd.Float64s()
		A := pd.Float64s()
		alpha := float64(1)
		e.c.Do(func() error { e.b.Dger(m, n, alpha, x, incX, y, incY, A, lda); return nil })
	case tensor.Float32:
		x := ad.Float32s()
		y := bd.Float32s()
		A := pd.Float32s()
		alpha := float32(1)
		e.c.Do(func() error { e.b.Sger(m, n, alpha, x, incX, y, incY, A, lda); return nil })
	}
	return e.b.Err()
}
