package tensor

import "github.com/pkg/errors"

func densesToTensors(a []*Dense) []Tensor {
	retVal := make([]Tensor, len(a))
	for i, t := range a {
		retVal[i] = t
	}
	return retVal
}

func densesToDenseTensors(a []*Dense) []DenseTensor {
	retVal := make([]DenseTensor, len(a))
	for i, t := range a {
		retVal[i] = t
	}
	return retVal
}

func tensorsToDenseTensors(a []Tensor) ([]DenseTensor, error) {
	retVal := make([]DenseTensor, len(a))
	var ok bool
	for i, t := range a {
		if retVal[i], ok = t.(DenseTensor); !ok {
			return nil, errors.Errorf("can only convert Tensors of the same type to DenseTensors. Trying to convert %T (#%d in slice)", t, i)
		}
	}
	return retVal, nil
}
