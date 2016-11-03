package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"
	"time"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/leesper/go_rng"
)

/*
	This file contains all the Ops related to building a neural network.

	Bear in mind that not all things that are related to a neural network are here, as not everything
	are encoded as Ops the way theano does it.

	See also: nn.go for functions that relate to neural networks
*/

type randomness byte

const (
	uniform randomness = iota
	gaussian
	binomial
)

type randomOp struct {
	which randomness
	shape types.Shape
	dt    Dtype

	a, b float64 // when uniform, a,b = low, high; when gaussian, a,b = mean, stdev
}

func makeRandomOp(which randomness, dt Dtype, a, b float64, shape ...int) randomOp {
	return randomOp{
		which: which,
		shape: types.Shape(shape),
		dt:    dt,
		a:     a,
		b:     b,
	}
}

func (op randomOp) Arity() int { return 0 }

// randomOp :: a
// randomOp :: Tensor a
func (op randomOp) Type() Type {
	if op.shape.IsScalar() {
		return op.dt
	}
	tt := newTensorType(op.shape.Dims(), op.dt)
	return tt
}

func (op randomOp) InferShape(...DimSizer) (types.Shape, error) { return op.shape, nil }
func (op randomOp) DiffWRT(i int) []bool                        { r := make([]bool, i); return r }
func (op randomOp) SymDiff(Nodes, *Node, *Node) (Nodes, error)  { return nil, nondiffErr(op) }

func (op randomOp) Do(...Value) (retVal Value, err error) {
	if op.shape.IsScalar() {
		switch op.dt {
		case Float64:
			switch op.which {
			case uniform:
				rand := rng.NewUniformGenerator(time.Now().UnixNano())
				v := rand.Float64Range(op.a, op.b)
				return anyToValue(v)
			case gaussian:
				rand := rng.NewGaussianGenerator(time.Now().UnixNano())
				v := rand.Gaussian(op.a, op.b)
				return anyToValue(v)
			case binomial:
				rand := rng.NewBinomialGenerator(time.Now().UnixNano())
				v := float64(rand.Binomial(int64(op.a), op.b))
				return anyToValue(v)
			}
		case Float32:
			switch op.which {
			case uniform:
				rand := rng.NewUniformGenerator(time.Now().UnixNano())
				v := rand.Float32Range(float32(op.a), float32(op.b))
				return anyToValue(v)
			case gaussian:
				rand := rng.NewGaussianGenerator(time.Now().UnixNano())
				v := float32(rand.Gaussian(op.a, op.b))
				return anyToValue(v)
			case binomial:
				rand := rng.NewBinomialGenerator(time.Now().UnixNano())
				v := float32(rand.Binomial(int64(op.a), op.b))
				return anyToValue(v)
			}
		default:
			err = nyi("randomOp.do", op.dt)
		}
	}

	switch op.dt {
	case Float64:
		switch op.which {
		case uniform:
			backing := Uniform64(op.a, op.b, op.shape...)
			v := tf64.NewTensor(tf64.WithBacking(backing), tf64.WithShape(op.shape...))
			return anyToValue(v)
		case gaussian:
			backing := Gaussian64(op.a, op.b, op.shape...)
			v := tf64.NewTensor(tf64.WithBacking(backing), tf64.WithShape(op.shape...))
			return anyToValue(v)
		case binomial:
			backing := Binomial64(op.a, op.b, op.shape...)
			v := tf64.NewTensor(tf64.WithBacking(backing), tf64.WithShape(op.shape...))
			return anyToValue(v)
		}
	case Float32:
		switch op.which {
		case uniform:
			backing := Uniform32(op.a, op.b, op.shape...)
			v := tf32.NewTensor(tf32.WithBacking(backing), tf32.WithShape(op.shape...))
			return anyToValue(v)
		case gaussian:
			backing := Gaussian32(op.a, op.b, op.shape...)
			v := tf32.NewTensor(tf32.WithBacking(backing), tf32.WithShape(op.shape...))
			return anyToValue(v)
		case binomial:
			backing := Binomial32(op.a, op.b, op.shape...)
			v := tf32.NewTensor(tf32.WithBacking(backing), tf32.WithShape(op.shape...))
			return anyToValue(v)
		}
	}
	panic("Unreachable")
}

func (op randomOp) ReturnsPtr() bool     { return false }
func (op randomOp) CallsExtern() bool    { return false }
func (op randomOp) OverwritesInput() int { return -1 }
func (op randomOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "%d%v%f%f", op.which, op.shape, op.a, op.b)
}

func (op randomOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op randomOp) String() string {
	return fmt.Sprintf("%v(%v, %v) - %v", op.which, op.a, op.b, op.shape)
}
