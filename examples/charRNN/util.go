package main

import (
	. "gorgonia"
	"gorgonia.org/tensor"
)

func sample(val Value) int {
	var t tensor.Tensor
	var ok bool
	if t, ok = val.(tensor.Tensor); !ok {
		panic("Expects a tensor")
	}

	return tensor.SampleIndex(t)
}

func maxSample(val Value) int {
	var t tensor.Tensor
	var ok bool
	if t, ok = val.(tensor.Tensor); !ok {
		panic("expects a tensor")
	}
	indT, err := tensor.Argmax(t, -1)
	if err != nil {
		panic(err)
	}
	if !indT.IsScalar() {
		panic("Expected scalar index")
	}
	return indT.ScalarValue().(int)
}
