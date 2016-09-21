package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestTTrace(t *testing.T) {
	var trace float64
	var err error

	backing := []float64{
		0, 1, 2,
		3, 4, 5,
	}
	T := NewTensor(WithShape(2, 3), WithBacking(backing))

	trace, err = T.Trace()
	if err != nil {
		t.Error(err)
	}
	if trace != float64(4) {
		t.Error("Wrong trace")
	}

	backing = RangeFloat64(0, 24)
	T = NewTensor(WithShape(2, 3, 4), WithBacking(backing))
	if _, err = T.Trace(); err == nil {
		t.Error("Expected an error when tracing a 3D matrix")
	}

}

func TestVecVecDot(t *testing.T) {
	assert := assert.New(t)
	var a, b, r *Tensor
	var err error
	var expectedShape types.Shape = types.ScalarShape()
	var expectedData []float64

	// standard test
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r, err = a.inner(b)
	expectedData = []float64{5}

	assert.Nil(err)
	assert.Equal(expectedData, r.data)
	assert.True(expectedShape.Eq(r.Shape()))

	/* STUPIDS TEST */

	// a is not a vector
	a = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	_, err = a.Inner(b)
	assert.NotNil(err)

	// b is not a vector
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	_, err = a.Inner(b)
	assert.NotNil(err)

	// shape mismatch
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(4, 1), WithBacking(RangeFloat64(0, 4)))
	r, err = a.Inner(b)
	assert.NotNil(err)

	// this sort of shape mismatch is allowed (for numpy compatibility)
	// this however, is stupid in my opinion, hence it's filedunder STUPIDS TEST
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(1, 3), WithBacking(RangeFloat64(0, 3)))
	r, err = a.Inner(b)

	assert.Nil(err)
	assert.Equal(expectedData, r.data)
	assert.True(expectedShape.Eq(r.Shape()))

}

func TestTdotProd(t *testing.T) {
	assert := assert.New(t)
	var a, b, r *Tensor
	var err error
	var expectedShape types.Shape
	var expectedData []float64

	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r, err = a.inner(b)
	expectedData = []float64{5}
	expectedShape = types.ScalarShape()

	assert.Nil(err)
	assert.Equal(expectedData, r.data)
	assert.True(expectedShape.Eq(r.Shape()))
}

func TestTMatVecMul(t *testing.T) {
	assert := assert.New(t)
	var A, b, r, r1, incr *Tensor
	var err error
	var expectedShape types.Shape
	var expectedData []float64

	// bog standard basic test with no reuse tensor provided
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	// r = NewTensor(WithShape(2, 1))
	r, err = A.MatVecMul(b)
	if err != nil {
		t.Error(err)
	}

	expectedShape = types.Shape{2}
	expectedData = []float64{5, 14}

	assert.Equal(expectedShape, r.Shape())
	assert.Equal(expectedData, r.data)

	// bog standard basic test with reuse tensor provided
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r = NewTensor(WithShape(2, 1))
	r1, err = A.MatVecMul(b, types.WithReuse(r))
	if err != nil {
		t.Error(err)
	}
	if r1 != r {
		t.Error("Reused is not reused")
	}

	expectedShape = types.Shape{2}
	expectedData = []float64{5, 14}

	assert.Equal(expectedShape, r.Shape())
	assert.Equal(expectedData, r.data)

	// standard basic test with incr tensor provided
	incr = NewTensor(WithShape(2, 1), WithBacking([]float64{50, 100}))
	r1, err = A.MatVecMul(b, types.WithIncr(incr))
	if err != nil {
		t.Error(err)
	}
	if r1 != incr {
		t.Error("Incr is not reused")
	}

	expectedShape = types.Shape{2}
	expectedData = []float64{55, 114}

	assert.Equal(expectedShape, r1.Shape())
	assert.Equal(expectedData, r1.data)

	// standard basic test with incr and reuse provided
	incr = NewTensor(WithShape(2, 1), WithBacking([]float64{50, 100}))
	r1, err = A.MatVecMul(b, types.WithIncr(incr), types.WithReuse(r))
	if err != nil {
		t.Error(err)
	}
	if r1 != incr {
		t.Error("Incr is not reused")
	}

	expectedShape = types.Shape{2}
	expectedData = []float64{55, 114}

	assert.Equal(expectedShape, r1.Shape())
	assert.Equal(expectedData, r1.data)

	/* TEST CHECKS */

	// stupid shapes
	A = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	_, err = A.MatVecMul(b)
	assert.NotNil(err)

	// reuse has impossible shape
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r = NewTensor(WithShape(6, 1))
	_, err = A.MatVecMul(b, types.WithReuse(r))
	assert.NotNil(err)

	// incr has in impossible shape
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r = NewTensor(WithShape(6, 1))
	_, err = A.MatVecMul(b, types.WithIncr(r))
	assert.NotNil(err)

	// stupid inputs - 1
	A = NewTensor(WithShape(2, 1, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r, err = A.MatVecMul(b)
	assert.NotNil(err)

	// stupid inputs - 2
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 3), WithBacking(RangeFloat64(0, 9)))
	_, err = A.MatVecMul(b)
	assert.NotNil(err)

	// stupid inputs - 3
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	// simulate stupid creation of Tensor. In the future, Tensor will be unexported.
	r = new(Tensor)
	r.AP = new(types.AP)
	_, err = A.MatVecMul(b, types.WithReuse(r))

	/* Standard use test */
	A = NewTensor(WithShape(2, 3))
	b = NewTensor(WithShape(3, 1))
	A.data = []float64{1, -1, 2, 0, -3, 1}
	b.data = []float64{2, 1, 0}
	r, err = A.MatVecMul(b)
	assert.Equal([]float64{1, -3}, r.data)
}

func TestTmatVecMul(t *testing.T) {
	assert := assert.New(t)
	var A, b, r *Tensor
	var expectedShape types.Shape
	var expectedData []float64

	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r = NewTensor(WithShape(2, 1))
	A.matVecMul(b, r)
	expectedShape = types.Shape{2, 1}
	expectedData = []float64{5, 14}

	assert.Equal(expectedShape, r.Shape())
	assert.Equal(expectedData, r.data)

	// Testing transposition
	A = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r = NewTensor(WithShape(2, 1))

	A.T()
	A.matVecMul(b, r)
	expectedShape = types.Shape{2, 1}
	expectedData = []float64{10, 13}

	assert.Equal(expectedShape, r.Shape())
	assert.Equal(expectedData, r.data)

	//Testing reuse of r and whether the data is zeroed out
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))

	A.matVecMul(b, r)
	expectedShape = types.Shape{2, 1}
	expectedData = []float64{5, 14}

	assert.Equal(expectedShape, r.Shape())
	assert.Equal(expectedData, r.data)
}

func TestTMatMul(t *testing.T) {
	assert := assert.New(t)
	var A, B, R, incr *Tensor
	var err error
	var expectedShape types.Shape
	var expectedData []float64

	//standard test, correct reused tensor provided
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R = NewTensor(WithShape(2, 2))
	R, err = A.MatMul(B, types.WithReuse(R))
	if err != nil {
		t.Error(err)
	}
	expectedData = []float64{10, 13, 28, 40}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	//standard test, no reused tensor provided
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R, err = A.MatMul(B)
	if err != nil {
		t.Error(err)
	}
	expectedData = []float64{10, 13, 28, 40}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// With Incr provided
	incr = NewTensor(WithShape(2, 2), WithBacking([]float64{100, 200, -200, -100}))
	R, err = A.MatMul(B, types.WithIncr(incr))
	if err != nil {
		t.Error(err)
	}
	if R != incr {
		t.Error("Incr not returned")
	}

	expectedData = []float64{110, 213, -172, -60}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// With Incr and Reuse provided
	incr = NewTensor(WithShape(2, 2), WithBacking([]float64{100, 200, -200, -100}))
	R, err = A.MatMul(B, types.WithIncr(incr), types.WithReuse(R))
	if err != nil {
		t.Error(err)
	}
	if R != incr {
		t.Error("Incr not returned")
	}
	expectedData = []float64{110, 213, -172, -60}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	/* TEST CHECKS */

	// A is not 2D matrix
	A = NewTensor(WithShape(2, 1, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	_, err = A.MatMul(B)
	assert.NotNil(err)

	// B is not 2D matrix
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 1, 2), WithBacking(RangeFloat64(0, 6)))
	_, err = A.MatMul(B)
	assert.NotNil(err)

	// shape mismatches
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	_, err = A.MatMul(B)
	assert.NotNil(err)

	// reused tensor has the wrong size
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R = NewTensor(WithShape(4, 4))
	_, err = A.MatMul(B, types.WithReuse(R))
	assert.NotNil(err)

	// incr tensor has the wrong size
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	incr = NewTensor(WithShape(4, 4))
	_, err = A.MatMul(B, types.WithIncr(incr))
	assert.NotNil(err)

}

func TestTmatMul(t *testing.T) {
	assert := assert.New(t)
	var A, B, R *Tensor
	var expectedShape types.Shape
	var expectedData []float64

	// standard test
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R = NewTensor(WithShape(2, 2))

	A.matMul(B, R)
	expectedData = []float64{10, 13, 28, 40}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// transA
	A = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R = NewTensor(WithShape(2, 2))

	A.T()
	A.matMul(B, R)
	expectedData = []float64{20, 26, 26, 35}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// transB
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	R = NewTensor(WithShape(2, 2))

	B.T()
	A.matMul(B, R)
	expectedData = []float64{5, 14, 14, 50}
	expectedShape = types.Shape{2, 2}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// transA and transB
	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R = NewTensor(WithShape(3, 3))

	A.T()
	B.T()
	A.matMul(B, R)
	expectedData = []float64{3, 9, 15, 4, 14, 24, 5, 19, 33}
	expectedShape = types.Shape{3, 3}

	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())
}

func TestTOuter(t *testing.T) {
	assert := assert.New(t)
	var a, b, R, R1, incr *Tensor
	var expectedShape types.Shape
	var expectedData []float64
	var err error

	// standard test
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	R, err = a.Outer(b)
	expectedData = []float64{0, 0, 0, 0, 1, 2, 0, 2, 4}
	expectedShape = types.Shape{3, 3}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// standard test with reuse
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(1, 4)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	R = NewTensor(WithShape(3, 3))
	R1, err = a.Outer(b, types.WithReuse(R))

	if R != R1 {
		t.Error("reuse is not returned")
	}

	expectedData = []float64{0, 1, 2, 0, 2, 4, 0, 3, 6}
	expectedShape = types.Shape{3, 3}
	assert.Nil(err, err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// test that reused tensor is zeroed before anything is done
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	R, err = a.Outer(b, types.WithReuse(R))
	expectedData = []float64{0, 0, 0, 0, 1, 2, 0, 2, 4}
	expectedShape = types.Shape{3, 3}
	assert.Nil(err, err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// with incr
	incr = NewTensor(WithShape(3, 3), WithBacking([]float64{100, 200, 300, 400, 500, 600, 700, 800, 900}))
	R, err = a.Outer(b, types.WithIncr(incr))
	if R != incr {
		t.Error("incr not returned")
	}

	expectedData = []float64{100, 200, 300, 400, 501, 602, 700, 802, 904}
	expectedShape = types.Shape{3, 3}
	assert.Nil(err, err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	/* ONLY IDIOTS DO THESE */

	// wrong reuse size
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	R = NewTensor(WithShape(3, 4))
	_, err = a.Outer(b, types.WithReuse(R))
	assert.NotNil(err)

	incr = NewTensor(WithShape(3, 4))
	_, err = a.Outer(b, types.WithIncr(incr))
	assert.NotNil(err)

	// not a vector?
	a = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	R = NewTensor(WithShape(3, 4))
	_, err = a.Outer(b, types.WithReuse(R))
	assert.NotNil(err)

}

func TestTouter(t *testing.T) {
	assert := assert.New(t)
	var a, b, R *Tensor
	var expectedShape types.Shape
	var expectedData []float64
	var err error

	// standard test
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	R = NewTensor(WithShape(3, 3))
	a.outer(b, R)
	expectedData = []float64{0, 0, 0, 0, 1, 2, 0, 2, 4}
	expectedShape = types.Shape{3, 3}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// differently shaped vector
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(1, 3), WithBacking(RangeFloat64(0, 3)))
	R = NewTensor(WithShape(3, 3))
	a.outer(b, R)
	expectedData = []float64{0, 0, 0, 0, 1, 2, 0, 2, 4}
	expectedShape = types.Shape{3, 3}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// differently sized vectors
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(4, 1), WithBacking(RangeFloat64(0, 4)))
	R = NewTensor(WithShape(3, 4))
	a.outer(b, R)
	expectedData = []float64{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6}
	expectedShape = types.Shape{3, 4}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

}

func TestTensorMul(t *testing.T) {
	assert := assert.New(t)
	var A, B, C *Tensor
	var expectedShape types.Shape
	var expectedData []float64
	var err error

	// 3T-3T
	A = NewTensor(WithShape(3, 4, 5), WithBacking(RangeFloat64(0, 60)))
	B = NewTensor(WithShape(4, 3, 2), WithBacking(RangeFloat64(0, 24)))
	if C, err = A.TensorMul(B, []int{1, 0}, []int{0, 1}); err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{5, 2}
	expectedData = []float64{4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306}
	assert.Equal(expectedData, C.data)
	assert.Equal(expectedShape, C.Shape())

	// make sure nothing's changed
	assert.Equal(types.Shape{3, 4, 5}, A.Shape())
	assert.Equal(types.Shape{4, 3, 2}, B.Shape())
	assert.Equal(RangeFloat64(0, 60), A.data)
	assert.Equal(RangeFloat64(0, 24), B.data)

	// TODO:
	// nT-vec
	// nT-mat
	// stupids
}

/*
//TODO
func TestCrazyCases(t *testing.T) {
	var A, a, B, b *Tensor

	A = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	B = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(1, 3), WithBacking(RangeFloat64(0, 3)))

	A.Dot(a) // [5,14] (2,1)
	A.Dot(b) // error out due to misaligned shape
	B.Dot(a) // error out due to misaligned shape
	B.Dot(b) // error out due to misaligned shape
	a.Dot(b) // [0,0,0,0,1,2,0,2,4] (3,3)
	a.Dot(A) // error out due to misaligned shape
	a.Dot(B) // [10,13](2,) - numpy specific weirdness
	b.Dot(a) // [5](1)
	b.Dot(A) // error out
	b.Dot(B) // [10,13](2,1)
}
*/
