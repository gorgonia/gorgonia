package values

import "gorgonia.org/tensor"

func tensorInfo(t tensor.Tensor) (dt tensor.Dtype, dim int) {
	dt = t.Dtype()
	dim = t.Dims()
	return
}
