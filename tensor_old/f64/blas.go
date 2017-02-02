package tensorf64

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/native"
)

var whichblas blas.Float64 = native.Implementation{}

func Use(b blas.Float64) {
	whichblas = b
	blas64.Use(b)
}

func WhichBLAS() blas.Float64 { return whichblas }
