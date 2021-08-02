package cuda

import "gorgonia.org/tensor"

// Add performs a + b
func (e *Engine) Add(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// AddScalar adds a scalar to the tensor. leftTensor indicates if the tensor is the left operand.
// Whether or not the input tensor is clobbered is left to the implementation
func (e *Engine) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// Sub performs a - b
func (e *Engine) Sub(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// SubScalar subtracts a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
// Whether or not the input tensor is clobbered is left to the implementation
func (e *Engine) SubScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// Mul performs a * b
func (e *Engine) Mul(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// MulScalar multiplies a scalar to the tensor. leftTensor indicates if the tensor is the left operand.
// Whether or not the input tensor is clobbered is left to the implementation
func (e *Engine) MulScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// Div performs a / b
func (e *Engine) Div(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// DivScalar divides a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
// Whether or not the input tensor is clobbered is left to the implementation
func (e *Engine) DivScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// Pow performs a ^ b
func (e *Engine) Pow(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// PowScalar exponentiates a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
// Whether or not the input tensor is clobbered is left to the implementation
func (e *Engine) PowScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// Mod performs a % b
func (e *Engine) Mod(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

// ModScalar performs a % b where one of the operands is scalar. leftTensor indicates if the tensor is the left operand.
// Whether or not hte input tensor is clobbered is left to the implementation
func (e *Engine) ModScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) FMA(a tensor.Tensor, x tensor.Tensor, y tensor.Tensor) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) FMAScalar(a tensor.Tensor, x interface{}, y tensor.Tensor) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) MatMul(a tensor.Tensor, b tensor.Tensor, preallocated tensor.Tensor) error {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) MatVecMul(a tensor.Tensor, b tensor.Tensor, preallocated tensor.Tensor) error {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Outer(a tensor.Tensor, b tensor.Tensor, preallocated tensor.Tensor) error {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Dot(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) SVD(a tensor.Tensor, uv bool, full bool) (s tensor.Tensor, u tensor.Tensor, v tensor.Tensor, err error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Lt(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) LtScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Lte(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) LteScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Gt(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) GtScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Gte(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) GteScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) ElEq(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) EqScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) ElNe(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) NeScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}
