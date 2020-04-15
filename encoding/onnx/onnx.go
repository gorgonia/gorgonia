package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

func UnmarshalONNX(b []byte) (*gorgonia.ExprGraph, error) {
	backend := newGraph()
	model := onnx.NewModel(backend)
	err := model.UnmarshalBinary(b)
	if err != nil {
		return nil, err
	}
	err = backend.PopulateExprgraph()
	if err != nil {
		return nil, err
	}
	return backend.exprgraph, nil
}
