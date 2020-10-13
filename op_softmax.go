package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
	"gorgonia.org/tensor"
)

type softmaxOp struct {
	shape tensor.Shape
	axis  int
}

func newSoftmaxOp(inputShape tensor.Shape, axes ...int) *softmaxOp {
	axis := -1
	if len(axes) > 0 {
		axis = axes[0]
	}
	softmaxop := &softmaxOp{
		shape: inputShape,
		axis:  axis,
	}

	return softmaxop
}

// SoftMax performs softmax on the input. Specifically this is used:
//		e^(a[i]) / sum((e^(a[i])))
// For a more numerically stable SoftMax, use StableSoftMax.
func SoftMax(x *Node, axis ...int) (*Node, error) {
	xShape := x.Shape()
	op := newSoftmaxOp(xShape, axis...)

	return ApplyOp(op, x)
}

func (op *softmaxOp) Arity() int { return 1 }

func (op *softmaxOp) ReturnsPtr() bool { return false }

func (op *softmaxOp) CallsExtern() bool { return false }

func (op *softmaxOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Softmax{%v}()", op.axis)
}

func (op *softmaxOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxOp) String() string {
	return fmt.Sprintf("Softmax{%d}()", op.axis)
}

func (op *softmaxOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape)

	return s, nil
}

func (op *softmaxOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a) // f(float64) float64
}

func (op *softmaxOp) OverwritesInput() int { return -1 }

func (op *softmaxOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var (
		in tensor.Tensor
		ok bool
	)

	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	return in, nil
}

func (op *softmaxOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Softmax input: %w", err)
	}

	aShape := inputTensor.Shape()
	axis := aShape.Dims() - 1 // default: last dim
	if aShape.IsColVec() || (aShape.IsVector() && !aShape.IsRowVec()) {
		axis = 0
	}

	if op.axis != -1 {
		axis = op.axis
	}

	exp, err := tensor.Exp(inputTensor)
	if err != nil {
		return nil, fmt.Errorf("error calculating exp for SoftMax: %w", err)
	}

	sum, err := tensor.Sum(exp, axis)
	if err != nil {
		return nil, fmt.Errorf("error calculating sum for SoftMax: %w", err)
	}

	ss := sum.Shape()
	es := exp.Shape()

	dimsDiff := es.Dims() - ss.Dims()
	if dimsDiff == 0 {
		div, err := tensor.Div(exp, sum)
		if err != nil {
			return nil, fmt.Errorf("error calculating div for SoftMax: %w", err)
		}

		return div, nil
	}

	// MULTIDIMENSIONAL SOFTMAX

	// REPEAT SUM (traditionally called broadcasting)
	newShape := tensor.Shape(tensor.BorrowInts(ss.Dims() + dimsDiff))
	if axis+1 > len(newShape) {
		return nil, fmt.Errorf("can't calculate SoftMax for the given axis %d", axis)
	}

	copy(newShape, ss)
	copy(newShape[axis+1:], newShape[axis:])
	newShape[axis] = 1

	if err = sum.Reshape(newShape...); err != nil {
		return nil, fmt.Errorf("error reshaping sum for SoftMax: %w", err)
	}

	if sum, err = tensor.Repeat(sum, axis, es[axis]); err != nil {
		return nil, fmt.Errorf("error repeating sum for SoftMax: %w", err)
	}

	return tensor.Div(exp, sum)
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *softmaxOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("SoftmaxOp.DoDiff needs 1 arguments")
	}

	odv := output.boundTo.(*dualValue)
	idv := inputs[0].boundTo.(*dualValue)
	idvd := idv.d.(*tensor.Dense)
	diffOp := newSoftmaxOpDiff(op.axis)

	result, err := diffOp.Do(odv.Value, odv.d)
	if err != nil {
		return err
	}

	sum, err := idvd.Add(result.(*tensor.Dense), tensor.UseUnsafe())
	if err != nil {
		return err
	}

	odv.d = sum

	return nil
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *softmaxOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	diffOp := newSoftmaxOpDiff(op.axis)
	nodes := make(Nodes, 1)

	nodes[0], err = ApplyOp(diffOp, output, grad)

	return nodes, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *softmaxOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("softmax operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

type softmaxDiffOp struct {
	axis int
}

func newSoftmaxOpDiff(axis int) *softmaxDiffOp {
	return &softmaxDiffOp{axis: axis}
}

func (op *softmaxDiffOp) Arity() int { return 2 }

func (op *softmaxDiffOp) ReturnsPtr() bool { return false }

func (op *softmaxDiffOp) CallsExtern() bool { return false }

func (op *softmaxDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "SoftmaxDiff{}()")
}

func (op *softmaxDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxDiffOp) String() string {
	return fmt.Sprintf("SoftmaxDiff{}()")
}

func (op *softmaxDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *softmaxDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a) // f(float64) float64
}

func (op *softmaxDiffOp) OverwritesInput() int { return -1 }

func (op *softmaxDiffOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, err
	}

	var (
		in   tensor.Tensor
		grad tensor.Tensor
		ok   bool
	)

	switch t := inputs[0].(type) {
	case *dualValue:
		if in, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, errors.Errorf("input should be a tensor, got %T", inputs[0])
		}
	case tensor.Tensor:
		in = t
	default:
		return nil, nil, errors.Errorf("input type is not supported, got %T", inputs[0])
	}

	switch t := inputs[1].(type) {
	case *dualValue:
		if grad, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, errors.Errorf("input should be a tensor, got %T", inputs[1])
		}
	case tensor.Tensor:
		grad = t
	default:
		return nil, nil, errors.Errorf("input type is not supported, got %T", inputs[1])
	}

	return in, grad, nil
}

func (op *softmaxDiffOp) Do(inputs ...Value) (Value, error) {
	y, grad, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SoftmaxDiff input: %w", err)
	}

	s := y.Shape()
	axis := op.axis
	if axis == -1 {
		axis = s.Dims() - 1
	}

	/*
		What follows is an a bit of a splayed out algorithm
		Let's imagine Y, and dY are both (a,b,c)-shaped tensors.
		We reshape it to a matrix. Let's examine the cases:
		case axis = 0:
			we reshape it to (1, a*b*c)
		case axis = 1:
			we reshape it to (a, b*c)
		case axis = 2:
			we reshape it to (a*b*c, 1)

		We'll call the result matrix M, with shape (N, D)

		Now, we'll do some work:
		1. Make scalars of shape (N,).
		2. Make mulars of shape (D,). To facilitate multiplication, we set the initial values
		   to the identity of multiplication: 1.
		3. Populate scalars. This is abit tricky:
			scalars[i] = Y[i] · dY[i]
		   TODO: insert mathematical explanation of what accumulating gradients magic is happening here.
		4. Reshape the scalars to (N, 1)
		5. Reshape the mulars to (1, D)
		6. Perform matrix multiplication... WITH A TWIST. We need to multiply all the results by -1. Then add a bias of 1.

		Step 6 can be done in the usual manner. However, the BLAS libraries contain `(D|S)gemm`, which allows you to set alpha and beta.
	*/

	prodBefore := tensor.ProdInts([]int(s[:axis])) // N
	prodAfter := tensor.ProdInts([]int(s[axis:]))  // D
	if prodBefore == 0 {                           // indicating an error
		prodBefore = 1
	}
	if prodAfter == 0 {
		prodAfter = 1
	}

	scalars := tensor.New(tensor.WithShape(prodBefore), tensor.Of(y.Dtype()))
	mulars := tensor.New(tensor.WithShape(prodAfter), tensor.Of(y.Dtype()))
	mulars.Memset(one(y.Dtype()).Data()) // set all mulars to 1.

	impl := gonum.Implementation{}
	var val interface{}

	switch yy := y.Data().(type) {
	case []float64:
		gradData := grad.Data().([]float64)
		mulData := mulars.Data().([]float64)
		var scaleData []float64

		switch sd := scalars.Data().(type) {
		case float64:
			scaleData = make([]float64, 1)
			scaleData[0] = sd
		case []float64:
			scaleData = sd
		}

		for i := 0; i < prodBefore; i++ {
			scaleData[i] = impl.Ddot(prodAfter, yy[i*prodAfter:], 1, gradData[i*prodAfter:], 1)
		}
		C := make([]float64, s.TotalSize()) // output

		// important note: here, alpha is -1 and beta is 1.
		impl.Dgemm(blas.NoTrans, blas.NoTrans, prodBefore, prodAfter, 1, -1, scaleData, 1, mulData, prodAfter, 1, C, prodAfter)
		val = C
	case []float32:
		gradData := grad.Data().([]float32)
		mulData := mulars.Data().([]float32)
		var scaleData []float32

		switch sd := scalars.Data().(type) {
		case float32:
			scaleData = make([]float32, 1)
			scaleData[0] = sd
		case []float32:
			scaleData = sd
		}

		for i := 0; i < prodBefore; i++ {
			scaleData[i] = impl.Sdot(prodAfter, yy[i*prodAfter:], 1, gradData[i*prodAfter:], 1)
		}

		C := make([]float32, s.TotalSize()) // output

		// important note: here, alpha is -1 and beta is 1.
		impl.Sgemm(blas.NoTrans, blas.NoTrans, prodBefore, prodAfter, 1, -1, scaleData, 1, mulData, prodAfter, 1, C, prodAfter)
		val = C
	case []complex64:
		panic("Complex64 not done yet")
	case []complex128:
		panic("Complex128 not done yet")
	}

	retVal := tensor.New(tensor.WithShape(s.Clone()...), tensor.WithBacking(val))

	return tensor.Mul(retVal, y, tensor.UseUnsafe())
}

// ensure it complies with the Op interface
var (
	_ Op   = &softmaxOp{}
	_ ADOp = &softmaxOp{}
	_ SDOp = &softmaxOp{}

	_ Op = &softmaxDiffOp{}
)
