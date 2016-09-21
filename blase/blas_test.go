package blase

import (
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/stretchr/testify/assert"
)

const EPSILON float64 = 1e-10

func floatEquals(a, b float64) bool {
	if (a-b) < EPSILON && (b-a) < EPSILON {
		return true
	}
	return false
}

func floatsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !floatEquals(v, b[i]) {
			return false
		}
	}
	return true
}

func randomFloat64(r, c int) []float64 {
	retVal := make([]float64, r*c)
	for i := range retVal {
		retVal[i] = rand.Float64()
	}
	return retVal
}

func TestQueue(t *testing.T) {
	assert := assert.New(t)
	whichblas := Implementation()

	A := randomFloat64(2, 2)
	B := randomFloat64(2, 3)
	C := make([]float64, 2*3)

	tA := blas.NoTrans
	tB := blas.NoTrans
	m := 2
	n := 3
	k := 2
	alpha := 1.0
	lda := 2
	ldb := 3
	beta := 0.0
	ldc := 3

	whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

	correct := []float64{0, 0, 0, 0, 0, 0}
	assert.Equal(correct, C)

	correct = []float64{
		A[0]*B[0] + A[1]*B[3],
		A[0]*B[1] + A[1]*B[4],
		A[0]*B[2] + A[1]*B[5],

		A[2]*B[0] + A[3]*B[3],
		A[2]*B[1] + A[3]*B[4],
		A[2]*B[2] + A[3]*B[5],
	}

	whichblas.DoWork()
	assert.True(floatsEqual(correct, C))

	/* Test if the queueing works */
	correct = []float64{0, 0, 0, 0, 0, 0}

	A = randomFloat64(2, 2)
	B = randomFloat64(2, 3)
	C1 := make([]float64, 2*3)

	correct1 := []float64{
		A[0]*B[0] + A[1]*B[3],
		A[0]*B[1] + A[1]*B[4],
		A[0]*B[2] + A[1]*B[5],

		A[2]*B[0] + A[3]*B[3],
		A[2]*B[1] + A[3]*B[4],
		A[2]*B[2] + A[3]*B[5],
	}

	whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C1, ldc)

	A = randomFloat64(2, 2)
	B = randomFloat64(2, 3)
	C2 := make([]float64, 2*3)

	correct2 := []float64{
		A[0]*B[0] + A[1]*B[3],
		A[0]*B[1] + A[1]*B[4],
		A[0]*B[2] + A[1]*B[5],

		A[2]*B[0] + A[3]*B[3],
		A[2]*B[1] + A[3]*B[4],
		A[2]*B[2] + A[3]*B[5],
	}

	whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C2, ldc)

	assert.True(floatsEqual(correct, C1))
	assert.True(floatsEqual(correct, C2))

	A = randomFloat64(2, 2)
	B = randomFloat64(2, 3)
	C3 := make([]float64, 2*3)

	correct3 := []float64{
		A[0]*B[0] + A[1]*B[3],
		A[0]*B[1] + A[1]*B[4],
		A[0]*B[2] + A[1]*B[5],

		A[2]*B[0] + A[3]*B[3],
		A[2]*B[1] + A[3]*B[4],
		A[2]*B[2] + A[3]*B[5],
	}

	whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C3, ldc)

	assert.True(floatsEqual(correct1, C1))
	assert.True(floatsEqual(correct2, C2))
	assert.True(floatsEqual(correct3, C3))
}
