package gorgonia

import (
	"fmt"
	"hash"
	"math"
	"runtime"
	"sync"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type softmaxOp struct {
	shape tensor.Shape
	axis  int
	isLog bool
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

// SoftMax -  implements the softmax operation
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
	return fmt.Sprintf("Softmax{%d, %v}()", op.axis, op.isLog)
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

	ret := tensor.New(tensor.WithShape(aShape.Clone()...), tensor.Of(inputTensor.Dtype()))
	data := inputTensor.Data()
	output := ret.Data()
	op.do(aShape, axis, data, output)
	return ret, nil

}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *softmaxOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("SoftmaxOp.DoDiff needs 1 arguments")
	}

	odv := output.boundTo.(*dualValue)
	idv := inputs[0].boundTo.(*dualValue)
	idvd := idv.d.(*tensor.Dense)
	diffOp := newSoftmaxOpDiff(op.axis, op.isLog)

	result, err := diffOp.Do(idv.Value, odv.Value, odv.d)
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

	diffOp := newSoftmaxOpDiff(op.axis, op.isLog)
	nodes := make(Nodes, 1)

	nodes[0], err = ApplyOp(diffOp, inputs[0], output, grad)

	return nodes, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *softmaxOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("softmax operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

func (op *softmaxOp) f64skernel(data, output []float64, inner, ostride, dimSize, dimStride int) {
	for i := 0; i < len(data); i++ {
		oi := i / inner
		ii := i % inner
		xidx := oi*ostride + ii
		yidx := oi*ostride + ii

		if xidx >= len(data) {
			continue
		}

		if yidx >= len(data) {
			continue
		}

		x := data[xidx:]
		y := output[yidx:]
		if len(x) == 0 {
			continue
		}

		max := x[0]
		for d := 1; d < dimSize && d*dimStride < len(x); d++ {
			dm := x[d*dimStride]
			if dm > max {
				max = dm
			}
		}

		var tmp float64
		for d := 0; d < dimSize && d*dimStride < len(x); d++ {
			z := math.Exp(x[d*dimStride] - max)
			if !op.isLog {
				y[d*dimStride] = z
			}
			tmp += z
		}
		if op.isLog {
			tmp = math.Log(tmp)
		} else {
			tmp = 1 / tmp
		}

		// set output
		for d := 0; d < dimSize && d*dimStride < len(y); d++ {
			if op.isLog {
				output[d*dimStride] = data[d*dimStride] - max - tmp
			} else {
				y[d*dimStride] *= tmp
			}
		}
	}
}

func (op *softmaxOp) f32skernel(data, output []float32, inner, ostride, dimSize, dimStride int) {
	for i := 0; i < len(data); i++ {
		oi := i / inner
		ii := i % inner
		xidx := oi*ostride + ii
		yidx := oi*ostride + ii

		if xidx >= len(data) {
			continue
		}

		if yidx >= len(output) {
			continue
		}

		x := data[xidx:]
		y := output[yidx:]
		if len(x) == 0 {
			continue
		}

		max := x[0]
		for d := 1; d < dimSize && d*dimStride < len(x); d++ {
			dm := x[d*dimStride]
			if dm > max {
				max = dm
			}
		}

		var tmp float32
		for d := 0; d < dimSize && d*dimStride < len(x); d++ {
			z := math32.Exp(x[d*dimStride] - max)
			if !op.isLog {
				y[d*dimStride] = z
			}
			tmp += z
		}

		if op.isLog {
			tmp = math32.Log(tmp)
		} else {
			tmp = 1 / tmp
		}

		// set output
		for d := 0; d < dimSize && d*dimStride < len(y); d++ {
			if op.isLog {
				output[d*dimStride] = data[d*dimStride] - max - tmp
			} else {
				y[d*dimStride] *= tmp
			}
		}
	}
}

// output and data are of the same size
func (op *softmaxOp) do(shp tensor.Shape, axis int, data, output interface{}) {
	blocks := runtime.GOMAXPROCS(0) + 1
	//blocks := calcBlocks(len(data), defaultBlockSize)

	dimSize := shp[axis]
	outer := tensor.ProdInts([]int(shp[:axis]))
	inner := tensor.ProdInts([]int(shp[axis+1:]))
	if outer == 0 {
		outer = 1
	}
	if inner == 0 {
		inner = 1
	}
	dimStride := inner
	ostride := dimSize * dimStride

	datalen := shp.TotalSize()
	if blocks < minParallelBlocks {
		switch data := data.(type) {
		case []float64:
			output := output.([]float64)
			op.f64skernel(data, output, inner, ostride, dimSize, dimStride)
		case []float32:
		}

		return
	}

	workers := workersChan()
	var wg sync.WaitGroup

	blockSize := datalen / blocks
	if blockSize == 0 {
		blockSize = datalen // 1 block
	}

	for b := 0; b < datalen; b += blockSize {
		wg.Add(1)
		switch data := data.(type) {
		case []float64:
			output := output.([]float64)
			end := b + blockSize
			if end > len(data) {
				end = len(data)
			}
			newdata := data[b:end]
			newoutput := output[b:end]
			go func(data, output []float64, dimSize, dimStride int, wg *sync.WaitGroup) {
				workers <- struct{}{}
				op.f64skernel(data, output, inner, ostride, dimSize, dimStride)
				wg.Done()
				<-workers
			}(newdata, newoutput, dimSize, dimStride, &wg)
		case []float32:
		}
	}
	wg.Wait()
}

type softmaxDiffOp struct {
	axis  int
	isLog bool
}

func newSoftmaxOpDiff(axis int, isLog bool) *softmaxDiffOp {
	return &softmaxDiffOp{axis: axis, isLog: isLog}
}

func (op *softmaxDiffOp) Arity() int { return 3 }

func (op *softmaxDiffOp) ReturnsPtr() bool { return false }

func (op *softmaxDiffOp) CallsExtern() bool { return false }

func (op *softmaxDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "SoftmaxDiff{%d, %v}()", op.axis, op.isLog)
}

func (op *softmaxDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxDiffOp) String() string {
	return fmt.Sprintf("SoftmaxDiff{%d, %v}()", op.axis, op.isLog)
}

func (op *softmaxDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *softmaxDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a, a) // f(float64) float64
}

func (op *softmaxDiffOp) OverwritesInput() int { return -1 }

func (op *softmaxDiffOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, nil, err
	}

	var (
		in   tensor.Tensor
		out  tensor.Tensor
		grad tensor.Tensor
		ok   bool
	)

	switch t := inputs[0].(type) {
	case *dualValue:
		if in, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, nil, errors.Errorf("input should be a tensor, got %T", inputs[0])
		}
	case tensor.Tensor:
		in = t
	default:
		return nil, nil, nil, errors.Errorf("input type is not supported, got %T", inputs[0])
	}

	switch t := inputs[1].(type) {
	case *dualValue:
		if out, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, nil, errors.Errorf("output should be a tensor, got %T", inputs[1])
		}
	case tensor.Tensor:
		out = t
	default:
		return nil, nil, nil, errors.Errorf("output type is not supported, got %T", inputs[1])
	}

	switch t := inputs[2].(type) {
	case *dualValue:
		if grad, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, nil, errors.Errorf("grad should be a tensor, got %T", inputs[1])
		}
	case tensor.Tensor:
		grad = t
	default:
		return nil, nil, nil, errors.Errorf("grad type is not supported, got %T", inputs[1])
	}

	return in, out, grad, nil
}

func (op *softmaxDiffOp) Do(inputs ...Value) (Value, error) {
	x, y, grad, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SoftmaxDiff input: %w", err)
	}

	s := x.Shape()
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

		Step 6 can be done in the usual manner. However, the BLAS librarie contain `(D|S)gemm`, which allows you to set alpha and beta.
	*/
	/*
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
	*/
	ret := tensor.New(tensor.WithShape(x.Shape().Clone()...), tensor.Of(x.Dtype()))
	op.do(x.Shape(), axis, x.Data(), y.Data(), grad.Data(), ret.Data())
	return ret, nil
}

func (op *softmaxDiffOp) f64Kernel(x, y, dy, retVal []float64, inner, ostride, dimSize, dimStride int) {
	for i := 0; i < len(x); i++ {
		oi := i / inner
		ii := i % inner
		xidx := oi*ostride + ii
		yidx := oi*ostride + ii
		dyidx := oi*ostride + ii
		dxidx := oi*ostride + ii

		if xidx >= len(x) {
			continue
		}

		if yidx >= len(y) {
			continue
		}
		if dyidx >= len(dy) {
			continue
		}
		if dxidx >= len(retVal) {
			continue
		}

		for d := 0; d < dimSize; d++ {
			// calculate sum
			var sum float64
			if op.isLog {
				sum += dy[d*dimStride]
			} else {
				sum += dy[d*dimStride] * y[d*dimStride]
			}

			for d := 0; d < dimSize; d++ {
				if op.isLog {
					retVal[d*dimStride] = dy[d*dimStride] - math.Exp(y[d*dimStride]*sum)
				} else {
					retVal[d*dimStride] = y[d*dimStride] * (dy[d*dimStride] - sum)
				}
			}

		}

	}
}

func (op *softmaxDiffOp) do(shp tensor.Shape, axis int, x, y, dy, retVal interface{}) {
	dimSize := shp[axis]
	outer := tensor.ProdInts([]int(shp[:axis]))
	inner := tensor.ProdInts([]int(shp[axis+1:]))
	if outer == 0 {
		outer = 1
	}
	if inner == 0 {
		inner = 1
	}
	dimStride := inner
	ostride := dimSize * dimStride
	switch x := x.(type) {
	case []float64:
		y := y.([]float64)
		dy := dy.([]float64)
		dydx := retVal.([]float64)
		op.f64Kernel(x, y, dy, dydx, inner, ostride, dimSize, dimStride)
	case []float32:
	}

}

// ensure it complies with the Op interface
var (
	_ Op   = &softmaxOp{}
	_ ADOp = &softmaxOp{}
	_ SDOp = &softmaxOp{}

	_ Op = &softmaxDiffOp{}
)
