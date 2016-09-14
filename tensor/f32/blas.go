package tensorf32

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas32"
	"github.com/gonum/blas/native"
)

var whichblas blas.Float32 = native.Implementation{}

func Use(b blas.Float32) {
	whichblas = b
	blas32.Use(b)
}

func WhichBLAS() blas.Float32 { return whichblas }
