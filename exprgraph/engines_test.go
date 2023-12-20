package exprgraph_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// this file tests some ideas about engines
type tensorCons func(g *exprgraph.Graph, name string, opts ...tensor.ConsOpt) gorgonia.Tensor

func nodeCons[DT any](g *exprgraph.Graph, name string, opts ...tensor.ConsOpt) gorgonia.Tensor {
	return exprgraph.New[DT](g, name, opts...)
}
func denseCons[DT any](g *exprgraph.Graph, name string, opts ...tensor.ConsOpt) gorgonia.Tensor {
	return dense.New[DT](opts...)
}

var workhorseTestCases = []struct {
	name              string
	engine            tensor.Engine
	cons              tensorCons
	isSameAsWorkhorse bool
	isGraphEngine     bool
}{
	{"tensor API", nil, denseCons[float64], true, false},
	{"tensor API with graph", nil, nodeCons[float64], false, true},
	{"fwdEngine", &FwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}, nodeCons[float64], false, true},
}

func TestWorkhorse(t *testing.T) {
	assert := assert.New(t)
	for _, tc := range workhorseTestCases {
		t.Run(tc.name, func(t *testing.T) {
			g := exprgraph.NewGraph(tc.engine)
			x := tc.cons(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
			e := x.Engine()
			w := e.Workhorse()
			_, ok := e.(GraphEngine)
			assert.Equal(tc.isSameAsWorkhorse, w == e, "%v Expected engine %T ==  workhorse %T to be %t", tc.name, e, w, tc.isSameAsWorkhorse)
			assert.Equal(tc.isGraphEngine, ok)
		})
	}

}
