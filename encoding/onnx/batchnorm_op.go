package gorgonnx

import (
	"fmt"
	"hash"
	"hash/fnv"
	"math"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

type fastBatchnorm struct {
	scale, bias, mean, varN gorgonia.Value
	epsilon                 float32
}

func (b *fastBatchnorm) Arity() int {
	return 1
}

func (b *fastBatchnorm) Type() hm.Type {
	t := gorgonia.TensorType{Dims: 4, Of: hm.TypeVariable('a')}
	return hm.NewFnType(t, t)
}

func (b *fastBatchnorm) InferShape(ns ...gorgonia.DimSizer) (tensor.Shape, error) {
	if len(ns) != b.Arity() {
		return nil, errors.New("wrong number of arguments for batchnorm")
	}

	return ns[0].(tensor.Shape).Clone(), nil
}

var errNotSupported = errors.New("not supported")

func (b *fastBatchnorm) check(v gorgonia.Value) (*tensor.Dense, error) {
	x, ok := v.(*tensor.Dense)
	if !ok {
		return nil, errNotSupported
	}

	if len(x.Shape()) != 4 {
		return nil, errNotSupported
	}
	if x.Shape()[0] != 1 {
		return nil, errNotSupported
	}
	if b.scale == nil || b.bias == nil ||
		b.mean == nil || b.varN == nil {
		return nil, errNotSupported
	}
	if len(b.scale.Shape()) != 1 || len(b.bias.Shape()) != 1 ||
		len(b.mean.Shape()) != 1 || len(b.varN.Shape()) != 1 {
		return nil, errNotSupported
	}
	ch := x.Shape()[1]
	if b.scale.Shape()[0] != ch || b.bias.Shape()[0] != ch ||
		b.mean.Shape()[0] != ch || b.varN.Shape()[0] != ch {
		return nil, errNotSupported
	}
	return x, nil
}

func (b *fastBatchnorm) Do(values ...gorgonia.Value) (gorgonia.Value, error) {
	// xNorm = (x - meanN) / sqrt( varN + b.epsilon)
	// output = scaleN * xNorm + biasN
	if len(values) != b.Arity() {
		return nil, errors.New("bad arity for fastBatchnorm")
	}
	x, err := b.check(values[0])
	if err != nil {
		return nil, err
	}
	// Reshape to CHW
	s := make([]int, len(x.Shape()))
	copy(s, x.Shape())
	err = x.Reshape(s[1:]...)
	if err != nil {
		return nil, err
	}
	defer func() {
		err := x.Reshape(s...)
		if err != nil {
			panic(err)
		}
	}()

	switch {
	case x.Dtype() == tensor.Float32:
		vals, err := native.Tensor3F32(x)
		if err != nil {
			return nil, err
		}
		// xNorm = (x - meanN) / sqrt( varN + b.epsilon)
		// output = scaleN * xNorm + biasN
		for c := 0; c < len(vals); c++ {
			mean := b.mean.Data().([]float32)[c]
			varV := b.varN.Data().([]float32)[c]
			scale := b.scale.Data().([]float32)[c]
			bias := b.bias.Data().([]float32)[c]
			for h := 0; h < len(vals[c]); h++ {
				for w := 0; w < len(vals[c][h]); w++ {
					x := vals[c][h][w]
					vals[c][h][w] = scale*((x-mean)/sqrtF32(varV+b.epsilon)) + bias
				}
			}
		}
	case x.Dtype() == tensor.Float64:
		vals, err := native.Tensor3F64(x)
		if err != nil {
			return nil, err
		}
		// xNorm = (x - meanN) / sqrt( varN + b.epsilon)
		// output = scaleN * xNorm + biasN
		for c := 0; c < len(vals); c++ {
			mean := b.mean.Data().([]float64)[c]
			varV := b.varN.Data().([]float64)[c]
			scale := b.scale.Data().([]float64)[c]
			bias := b.bias.Data().([]float64)[c]
			for h := 0; h < len(vals[c]); h++ {
				for w := 0; w < len(vals[c][h]); w++ {
					x := vals[c][h][w]
					vals[c][h][w] = scale*((x-mean)/math.Sqrt(varV+float64(b.epsilon))) + bias
				}
			}
		}
	default:
		return x, errors.New("type not handled")
	}
	return x, nil
}

func sqrtF32(v float32) float32 {
	return float32(math.Sqrt(float64(v)))
}

func (b *fastBatchnorm) ReturnsPtr() bool {
	return true
}

func (b *fastBatchnorm) CallsExtern() bool {
	return false
}

func (b *fastBatchnorm) OverwritesInput() int {
	//return -1
	return 0
}

func (b *fastBatchnorm) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "batchnorm-%1.1f", b.epsilon)
}

func (b *fastBatchnorm) Hashcode() uint32 {
	h := fnv.New32a()
	b.WriteHash(h)
	return h.Sum32()
}

func (b *fastBatchnorm) String() string {
	return fmt.Sprintf("batchnorm-%1.1f", b.epsilon)
}
