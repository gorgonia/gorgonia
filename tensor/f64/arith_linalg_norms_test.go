package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
)

var normtests = []types.NormOrder{
	types.FrobeniusNorm(),
	types.NuclearNorm(),
	types.InfNorm(),
	types.NegInfNorm(),
	types.Norm(0),
	types.Norm(1),
	types.Norm(-1),
	types.Norm(2),
	types.Norm(-2),
}

func TestT_Norm(t *testing.T) {
	var T, retVal *Tensor
	var err error
	var backing []float64

	// empty
	backing = make([]float64, 0)
	T = NewTensor(WithBacking(backing))

	// vecktor
	backing = []float64{1, 2, 3, 4}
}
