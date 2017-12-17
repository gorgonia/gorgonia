package blase

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/blas"
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

func testDGEMM(t *testing.T, whichblas *context) (C, correct []float64) {
	A := randomFloat64(2, 2)
	B := randomFloat64(2, 3)

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

	C = make([]float64, 2*3)
	correct = []float64{
		A[0]*B[0] + A[1]*B[3],
		A[0]*B[1] + A[1]*B[4],
		A[0]*B[2] + A[1]*B[5],

		A[2]*B[0] + A[3]*B[3],
		A[2]*B[1] + A[3]*B[4],
		A[2]*B[2] + A[3]*B[5],
	}

	whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	return
}

func TestQueue(t *testing.T) {
	assert := assert.New(t)
	whichblas := Implementation()

	workAvailable := whichblas.WorkAvailable()
	go func() {
		for range workAvailable {
			whichblas.DoWork()
		}
	}()

	var corrects [][]float64
	var Cs [][]float64
	for i := 0; i < 4; i++ {
		C, correct := testDGEMM(t, whichblas)
		Cs = append(Cs, C)
		corrects = append(corrects, correct)

		if i < workbufLen {
			assert.True(floatsEqual(make([]float64, 6), C))
		}
	}
	whichblas.DoWork()

	for i, C := range Cs {
		assert.True(floatsEqual(corrects[i], C))
	}
}
