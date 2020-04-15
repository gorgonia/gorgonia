package gorgonnx

import (
	"encoding/binary"
	"errors"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

type imageScaler struct {
	bias  []float32
	scale float32
}

func (i *imageScaler) Arity() int {
	return 1
}

func (i *imageScaler) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (i *imageScaler) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, errors.New("imageScaler: infershape failed, nil shape")
	}

	return inputs[0].(tensor.Shape), nil
}

func (i *imageScaler) Do(values ...gorgonia.Value) (gorgonia.Value, error) {
	if len(values) != i.Arity() {
		return nil, errors.New("bad arity for fastBatchnorm")
	}
	x, ok := values[0].(*tensor.Dense)
	if !ok {
		return nil, errors.New("only dense are supported")
	}
	var err error
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
	vals, err := native.Tensor3F32(x)
	if err != nil {
		return nil, err
	}
	for c := 0; c < len(vals); c++ {
		bias := i.bias[c]
		for h := 0; h < len(vals[c]); h++ {
			for w := 0; w < len(vals[c][h]); w++ {
				x := vals[c][h][w]
				vals[c][h][w] = i.scale*x + bias
			}
		}
	}
	return x, nil

	panic("not implemented")
}

func (i *imageScaler) ReturnsPtr() bool {
	return true
}

func (i *imageScaler) CallsExtern() bool {
	return false
}

func (i *imageScaler) OverwritesInput() int {
	return 0
}

func (i *imageScaler) WriteHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, []byte(`imageScaler`)); err != nil {
		panic(err)
	}
}

func (i *imageScaler) Hashcode() uint32 {
	h := fnv.New32a()
	i.WriteHash(h)
	return h.Sum32()
}

func (i *imageScaler) String() string {
	return "imageScaler"
}

func init() {
	register("ImageScaler", newImageScaler)
}

func newImageScaler() operator {
	return &imageScaler{}
}

func (i *imageScaler) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	if len(children) != 1 {
		return errors.New("ImageScaler: bad number of children")
	}
	x := children[0].gorgoniaNode
	if x.Dtype() != tensor.Float32 {
		return &onnx.ErrNotImplemented{
			Operator: "ImageScaler",
			Message:  "Only float32 is supported",
		}
	}
	if len(x.Shape()) != 4 {
		return errors.New("Expected a 4D tensor [N,C,H,W]")
	}
	if len(i.bias) != x.Shape()[1] {
		return errors.New("bias should be the same size as the channel")
	}
	var err error
	n.gorgoniaNode, err = gorgonia.ApplyOp(i, x)
	return err
}

func (i *imageScaler) init(o onnx.Operation) error {
	i.scale = 1

	bias, ok := o.Attributes["bias"]
	if !ok {
		return errors.New("imageScaler: expected bias attribute is not found")
	}
	err := errors.New("bias in not a []float32")
	if bias, ok := bias.([]float32); ok {
		i.bias = []float32(bias)
		err = nil
	}
	if err != nil {
		return err
	}

	if scale, ok := o.Attributes["scale"]; ok {
		err = errors.New("scale in not a float32")
		if scale, ok := scale.(float32); ok {
			i.scale = scale
			err = nil
		}
	}
	return err
}
