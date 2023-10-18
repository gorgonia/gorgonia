package values

import (
	"gorgonia.org/dtype"
	"gorgonia.org/tensor"
)

func tensorInfo(t tensor.Desc) (dt dtype.Dtype, dim int) {
	dt = t.Dtype()
	dim = t.Dims()
	return
}
