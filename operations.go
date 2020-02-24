package gorgonia

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// contains all public operations that can be performed on nodes
// all the functions here have the signature:
// 		func (...) (*Node, error)

/* BINARY FUNCTIONS */
func binOpNode(op BinaryOp, a, b *Node) (retVal *Node, err error) {
	stabLogf("Creating node for %v, a: %p, b: %p", op, a, b)
	enterLogScope()
	defer leaveLogScope()
	// maybe make stabilization a build tag?
	if stabilization {
		enterLogScope()
		if ebo, ok := op.(elemBinOp); ok {
			ot := ebo.binOpType()

			enterLogScope()
			for _, fn := range binOpStabilizationFns[ot] {
				if retVal, err = fn(a, b); err == nil {
					leaveLogScope()
					return
				}

				if _, ok := err.(errNoStabilization); !ok {
					leaveLogScope()
					return
				}
			}
			leaveLogScope()
		}
		leaveLogScope()
	}
	stabLogf("No bin op stabilization")

	return ApplyOp(op, a, b)
}

// Mul is the general handler for multiplication of nodes. It is extremely overloaded. Only use if you know what you're doing
//
// If any of the nodes are ScalarType, then it'll be redirected to HadamardProd() instead
// If the nodes are both vectors (that is, have a shape of (x, 1) or (1, x)), then the operator used will be a vectorDot
// If only one of the nodes is a vector, then the operator used will be a matrix-vector multiplication will be used, and most importantly,
// a transpose will be used (when necessary)
// If both nodes are matrices, then well, matrix multiplication will be done
func Mul(a, b *Node) (retVal *Node, err error) {
	if a.IsScalar() || b.IsScalar() {
		return HadamardProd(a, b)
	}

	var op BinaryOp
	switch {
	case a.IsVector() && b.IsVector():
		op = linAlgBinOp{āBinaryOperator: vecDotOperator}
		return binOpNode(op, a, b)
	case a.IsVector() && b.IsMatrix():
		op = linAlgBinOp{āBinaryOperator: matVecMulOperator, transA: true}
		return binOpNode(op, b, a)
	case a.IsMatrix() && b.IsVector():
		op = linAlgBinOp{āBinaryOperator: matVecMulOperator}
		return binOpNode(op, a, b)
	case a.IsMatrix() && b.IsMatrix():
		op = linAlgBinOp{āBinaryOperator: matMulOperator}
		return binOpNode(op, a, b)
	default:
		return nil, errors.Errorf(nyiFail, "Mul", fmt.Sprintf("a %v b %v", a.shape, b.shape))
	}
}

// BatchedMatMul returns a node representing the batched mat mul operation.
//
// A list of transpose options are allowed. The
func BatchedMatMul(a, b *Node, transes ...bool) (retVal *Node, err error) {
	op := linAlgBinOp{āBinaryOperator: batchedMatMulOperator}
	switch len(transes) {
	case 0:
		// noop
	case 1:
		// transA
		op.transA = transes[0]
	case 2:
		// transA and transB
		op.transA = transes[0]
		op.transB = transes[1]
	default:
		// unsupported
		op.transA = transes[0]
		op.transB = transes[1]
	}

	return binOpNode(op, a, b)
}

// OuterProd returns a Node representing the outer product of two vectors. This function will return an error if both input nodes are not vectors
func OuterProd(a, b *Node) (retVal *Node, err error) {
	if !a.IsVector() || !b.IsVector() {
		return nil, errors.Errorf("Expected only vectors to be able to do OuterProd. %v is %v. %v is %v", a, a.Shape(), b, b.Shape()) //for now
	}

	// TODO: maybe align shapes?
	op := linAlgBinOp{āBinaryOperator: outerProdOperator}
	return binOpNode(op, a, b)
}

// Div is a shortcut function for HadamardDiv for scalar values. For matrix/tensor values, the matrix division operation is not yet handled, and will panic.
func Div(a, b *Node) (retVal *Node, err error) {
	if a.IsScalar() || b.IsScalar() || a.Shape().Eq(b.Shape()) {
		return HadamardDiv(a, b)
	}

	// otherwise, matrix division
	panic("Unhandled")
}

/* UNARY STUFF */

func unaryOpNode(op Op, a *Node) (retVal *Node, err error) {
	stabLogf("Creating node for %v, a: %p %v", op, a, a)
	enterLogScope()
	defer leaveLogScope()
	if stabilization {

		// do optimization/stabilization
		// TODO: maybe recursively stabilize?
		enterLogScope()
		ot := op.(elemUnaryOp).unaryOpType()
		for _, fn := range unaryOpStabilizationFns[ot] {
			if retVal, err = fn(a); err == nil {
				stabLogf("stabilized")
				leaveLogScope()
				return
			}

			if _, ok := err.(errNoStabilization); !ok {
				stabLogf("Actual error")
				leaveLogScope()
				return
			}
			stabLogf("No stabilization found")
		}
		leaveLogScope()
		stabLogf("No stabilizations - retVal: %v", retVal)
	}

	return ApplyOp(op, a)
}

// more complex unaries

// SoftMax performs softmax on the input. Specifically this is used:
//		e^(a[i]) / sum((e^(a[i])))
// For a more numerically stable SoftMax, use StableSoftMax.
// TODO: MULTI RANK SOFTMAX
func SoftMax(a *Node, axes ...int) (retVal *Node, err error) {
	aShape := a.Shape()
	axis := aShape.Dims() - 1 // default: last dim
	if a.IsColVec() || (a.IsVector() && !a.IsRowVec()) {
		axis = 0
	}

	if len(axes) > 0 {
		if axes[0] >= axis+1 || axes[0] < 0 {
			return nil, errors.Errorf("Cannot perform SoftMax on axis %d. Input has shape %v", axes[0], a.Shape())
		}
		axis = axes[0]
	}

	var exp, sum *Node
	if exp, err = Exp(a); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	if sum, err = Sum(exp, axis); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if sum.IsScalar() {
		return HadamardDiv(exp, sum)
	}

	// reshape if necessary
	ss := sum.Shape()
	diff := exp.Shape().Dims() - ss.Dims()

	// TODO: multirank softmax
	if diff > 0 {
		newShape := tensor.Shape(tensor.BorrowInts(ss.Dims() + diff))
		copy(newShape, ss)
		copy(newShape[axis+1:], newShape[axis:])
		newShape[axis] = 1

		if sum, err = Reshape(sum, newShape); err != nil {
			return nil, errors.Wrap(err, "Failed to reshape")
		}
	}

	return BroadcastHadamardDiv(exp, sum, nil, []byte{byte(axis)})

}

// StableSoftMax performs a numerically stable softmax on the input. Specifically this is the formula used:
//		e^(a - max(a)) / sum(e^(a - max(a)))
func StableSoftMax(a *Node) (retVal *Node, err error) {
	var max, exp, sum *Node
	if max, err = Max(a); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	if retVal, err = Sub(a, max); err == nil {
		if exp, err = Exp(retVal); err == nil {
			if sum, err = Sum(exp, 1); err == nil {
				return HadamardDiv(exp, sum)
			}
			return nil, errors.Wrap(err, operationError)
		}
		return nil, errors.Wrap(err, operationError)
	}
	return nil, errors.Wrap(err, operationError)
}

// LogSumExp performs addition in the log domain
func LogSumExp(a *Node, axis int) (retVal *Node, err error) {
	var max, exp, sum, logSum *Node
	if max, err = Max(a, axis); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	if retVal, err = Sub(a, max); err == nil {
		if exp, err = Exp(retVal); err == nil {
			if sum, err = Sum(exp, axis); err == nil {
				if sum, err = Add(sum, max); err == nil {
					if logSum, err = Log(sum); err == nil {
						return Sum(logSum, axis)
					}
				}
			}
		}
	}
	return nil, errors.Wrap(err, operationError)
}

/* Aggregate Functions */

// At is a symbolic operation for getting a value at the provided coordinates.
// If the input is a scalar, all the coordinates MUST be 0, or else an error will be returned.
func At(a *Node, coords ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		for _, c := range coords {
			if c != 0 {
				return nil, errors.Errorf("At() only works with scalars when the coordinates are (0...0). Got %v instead", coords)
			}
		}
		return a, nil
	}

	dims := a.Dims()
	op := atOp{
		coordinates: coords,
		d:           dims,
	}

	return ApplyOp(op, a)
}

// Max performs a max() on the input and the provided axes.
func Max(a *Node, along ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		// can't max a scalar. Should return error
		return a, nil
	}

	dims := a.Dims()
	if len(along) == 0 {
		along = intRange(0, dims)
	}

	op := newMaxOp(along, dims)

	return ApplyOp(op, a)
}

// Mean performs a mean() on the input and the provided axes.
func Mean(a *Node, along ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		// can't mean a scalar... return error
		return a, nil
	}

	dims := a.Dims()

	if len(along) == 0 {
		along = intRange(0, dims)
	}

	var s *Node
	if s, err = Sum(a, along...); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	sizes := make(Nodes, len(along))
	for i, axis := range along {
		if sizes[i], err = SizeOf(axis, a); err != nil {
			return nil, errors.Wrap(err, operationError)
		}
	}

	var counts *Node
	if counts, err = ReduceMul(sizes); err == nil {
		return HadamardDiv(s, counts)
	}
	return nil, errors.Wrap(err, operationError)
}

// Sum performs a sum() on the input and the provided axes.
func Sum(a *Node, along ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		retVal = a // or error?
		return
	}

	dims := a.Dims()
	if len(along) == 0 {
		switch {
		case a.IsRowVec():
			along = []int{1}
			dims = 1
		case a.IsColVec(), a.IsVector():
			along = []int{0}
			dims = 1
		default:
			along = intRange(0, dims)
		}
	}

	op := newSumOp(along, a.shape, dims)
	return ApplyOp(op, a)
}

// Norm returns the p-norm of a Value. Use p=2 if you want to use unordered norms.
//
// This is a simpler version of the norms found in the Tensor package, which specializes and optimizes even more
// (well, given it's adapted from Numpy, it is clearly way more optimized)
func Norm(a *Node, axis, p int) (retVal *Node, err error) {
	if p == 2 {
		if retVal, err = Square(a); err == nil {
			if retVal, err = Sum(retVal, axis); err == nil {
				if retVal, err = Sqrt(retVal); err != nil {
					return nil, errors.Wrap(err, operationError)
				}
			} else {
				return nil, errors.Wrap(err, operationError)
			}
		} else {
			return nil, errors.Wrap(err, operationError)
		}
		return
	}

	var dt tensor.Dtype
	if dt, err = dtypeOf(a.t); err != nil {
		return nil, errors.Wrapf(err, "Failed to determine the dtype of %T", a.t)
	}

	var b, inv *Node
	switch dt {
	case Float32:
		b = NewConstant(float32(p))
		inv = NewConstant(float32(1) / float32(p))
	case Float64:
		b = NewConstant(float64(p))
		inv = NewConstant(float64(1) / float64(p))
	default:
		return nil, errors.New("Cannot norm a non-floating point type")
	}

	if retVal, err = Pow(a, b); err == nil {
		if retVal, err = Sum(retVal, axis); err == nil {
			if retVal, err = Pow(retVal, inv); err != nil {
				return nil, errors.Wrap(err, operationError)
			}
		} else {
			return nil, errors.Wrap(err, operationError)
		}
	} else {
		return nil, errors.Wrap(err, operationError)
	}
	return
}

// Reduction

// ReduceAdd takes a slice of *Nodes, and folds them into one by adding
func ReduceAdd(nodes Nodes, opts ...NodeConsOpt) (retVal *Node, err error) {
	switch len(nodes) {
	case 0:
		return nil, nil // or error?
	case 1:
		return nodes[0], nil
	case 2:
		if retVal, err = Add(nodes[0], nodes[1]); err == nil {
			for _, opt := range opts {
				opt(retVal)
			}
		} else {
			return nil, errors.Wrap(err, operationError)
		}
		return
	}

	retVal = nodes[0]
	for i, n := range nodes {
		if i == 0 {
			continue
		}

		if retVal, err = Add(retVal, n); err != nil {
			err = errors.Wrap(err, operationError)
			return
		}
		for _, opt := range opts {
			opt(retVal)
		}
	}
	return
}

// ReduceMul is like foldl(*, nodes)
func ReduceMul(nodes Nodes, opts ...NodeConsOpt) (retVal *Node, err error) {
	switch len(nodes) {
	case 0:
		return nil, nil // or error?
	case 1:
		return nodes[0], nil
	case 2:
		if retVal, err = Mul(nodes[0], nodes[1]); err == nil {
			for _, opt := range opts {
				opt(retVal)
			}
		} else {
			return nil, errors.Wrap(err, operationError)
		}
		return
	}

	retVal = nodes[0]
	for i, n := range nodes {
		if i == 0 {
			continue
		}

		if retVal, err = Mul(retVal, n); err != nil {
			return nil, errors.Wrap(err, operationError)
		}
		for _, opt := range opts {
			opt(retVal)
		}
	}
	return
}

/* Shape related operations */

// SizeOf returns the size of a value along an axis
func SizeOf(axis int, x *Node) (retVal *Node, err error) {
	op := sizeOp{
		axis: axis,
		d:    x.Dims(),
	}

	// if the shape is known
	if x.shape != nil {
		op.val = x.shape[axis]
	}

	return ApplyOp(op, x)
}

// Slice slices a *Node. For T[:] slices, pass in nil. Will error out if node's type is not a Tensor
func Slice(n *Node, slices ...tensor.Slice) (retVal *Node, err error) {
	if _, ok := n.t.(TensorType); !ok {
		return nil, errors.Errorf("Cannot slice on non Tensor tensor. Got %T", n.t)
	}

	if len(slices) > n.shape.Dims() {
		return nil, errors.Errorf("Cannot slice %v. Shape: %v. Slices: %d", n, n.shape, len(slices))
	}

	retVal = n
	var dimsChanged int
	for i, s := range slices {
		var along int
		if i > 0 {
			if prev := slices[i-1]; prev != nil {
				if prev.End()-prev.Start() == 1 {
					dimsChanged++
				}
			}
		}
		along = i - dimsChanged

		op := newSliceOp(s, along, retVal.Dims())
		if retVal, err = ApplyOp(op, retVal); err != nil {
			return
		}
	}
	return
}

// Transpose performs a transpose on the input and provided permutation axes.
func Transpose(n *Node, axes ...int) (retVal *Node, err error) {
	// prep axes
	if len(axes) > 0 && len(axes) != n.Dims() {
		return nil, errors.Errorf("n has %d dims, while requested transposes is %d", n.Dims(), len(axes))
	}
	dims := len(n.shape)
	if len(axes) == 0 || axes == nil {
		axes = make([]int, dims)
		for i := 0; i < dims; i++ {
			axes[i] = dims - 1 - i
		}
	}

	// if axes is 0, 1, 2, 3... then no op
	if monotonic, incr1 := tensor.IsMonotonicInts(axes); monotonic && incr1 && axes[0] == 0 {
		retVal = n
		return
	}
	op := transposeOp{
		pattern: axes,
		d:       len(axes),
	}

	return ApplyOp(op, n)
}

// Concat performs a concatenate on the provided axis and inputs.
func Concat(axis int, ns ...*Node) (retVal *Node, err error) {
	// check that all the nodes have the same number of dimensions
	var d int
	for i, n := range ns {
		if i == 0 {
			d = n.shape.Dims()
			continue
		}

		if n.shape.Dims() != d {
			err = errors.Errorf("Dimension mismatch. Expected all the nodes to be concatenated to have %d dimensions. Got %d instead", d, n.shape.Dims())
			return
		}
	}

	if d == 0 {
		err = errors.Errorf("Concat only works on Tensor nodes")
		return
	}

	if axis >= d {
		err = errors.Errorf("Invalid axis. Nodes have %d dimensions. Axis is %d", d, axis)
		return
	}

	op := concatOp{axis: axis, d: d, children: len(ns)}
	return ApplyOp(op, ns...)
}

// Unconcat is the opposite of the built in concat function
// TODO: port this back to Gorgonia and use Gorgonia's sli instead
func Unconcat(a *Node, along int, n int) (Nodes, error) {
	aShape := a.Shape()
	if along < 0 || along > aShape.Dims() {
		return nil, errors.Errorf("Unable to Unconcat a of shape %v along axis %d", aShape, along)
	}

	if aShape[along]%n != 0 {
		return nil, errors.Errorf("Axis %d of %v cannot be nicely split into %d parts", along, aShape, n)
	}

	newShapeAlong := aShape[along] / n
	batches := aShape[along] / newShapeAlong

	var start int
	var retVal Nodes
	for i := 0; i < batches; i++ {
		ss := make([]tensor.Slice, len(aShape))
		for i := range ss {
			if i == along {
				ss[i] = S(start, start+newShapeAlong)
			} else {
				ss[i] = S(0, aShape[i])
			}
		}

		a2, err := Slice(a, ss...)
		if err != nil {
			return nil, errors.Wrapf(err, "Unable to slice a of shape %v along %d on batch %d. Slices were: %v", aShape, along, i, ss)
		}
		retVal = append(retVal, a2)
		start += newShapeAlong
	}
	return retVal, nil
}

// Reshape reshapes a node and returns a new node with the new shape
func Reshape(n *Node, to tensor.Shape) (retVal *Node, err error) {
	// check shape
	var negs int
	var infer int
	for i, s := range to {
		if s < 0 {
			negs++
			infer = i
		}
	}
	if negs > 1 {
		return nil, errors.Errorf("Unfortunately, inference of reshape parameters only allow for one variable (a negative number). Got %v instead", to)
	}

	if negs == 1 {
		prod := 1
		for i, s := range to {
			if i == infer {
				continue
			}
			prod *= s
		}
		inferred, rem := divmod(n.Shape().TotalSize(), prod)
		if rem != 0 {
			return nil, errors.Errorf("Cannot reshape %v to %v", n.Shape(), to)
		}
		to[infer] = inferred
	}

	op := reshapeOp{
		from: n.Shape(),
		to:   to,
	}
	return ApplyOp(op, n)
}

/* Contraction related operations */

// Tensordot performs a tensor contraction of a and b along specified axes.
func Tensordot(aAxes []int, bAxes []int, a, b *Node) (retVal *Node, err error) {

	// Check if input tensors actually have dim ⩾ 1
	if (len(a.Shape()) < 1) || (len(b.Shape()) < 1) || (a.Dims() < 1) || (b.Dims() < 1) {
		return nil, errors.New("Input Node's shape should have length at least 1")
	}

	// Check if number of specified axes for a and b matches
	if len(aAxes) != len(bAxes) {
		return nil, errors.New("Number of Axes supplied along which to contract tensors does not match")
	}

	// Check for duplicate indices
	if containsDuplicate(aAxes) || containsDuplicate(bAxes) {
		return nil, errors.New("Supplied axes to contract along contain duplicates")
	}

	// Check for more compatibility

	aShape := a.Shape()
	bShape := b.Shape()

	for _, aAxis := range aAxes {
		if aAxis >= len(aShape) {
			return nil, errors.New("Supplied higher higher axes number to contract along than Tensor's actual number of axes")
		}
	}

	for _, bAxis := range bAxes {
		if bAxis >= len(bShape) {
			return nil, errors.New("Supplied higher higher axes number to contract along than Tensor's actual number of axes")
		}
	}

	for aAxis, aDim := range aAxes {
		if aShape[aDim] != bShape[bAxes[aAxis]] {
			return nil, errors.New("Dimension mismatch: Can't contract tensors along supplied axes")
		}
	}

	// Otherwise, apply contraction
	aDims := len(aShape)
	bDims := len(bShape)
	retDims := len(aShape) + len(bShape) - 2*len(aAxes)

	op := tensordotOp{aAxes: aAxes, bAxes: bAxes, aDims: aDims, bDims: bDims, retDims: retDims}

	return ApplyOp(op, a, b)
}

// Mish is a novel activation function that is self regularizing.
//
// https://arxiv.org/abs/1908.08681
func Mish(a *Node) (retVal *Node, err error) {
	var sp, tsp *Node
	if sp, err = Softplus(a); err != nil {
		return nil, errors.Wrap(err, "Mish() - SoftPlus failed")
	}
	if tsp, err = Tanh(sp); err != nil {
		return nil, errors.Wrap(err, "Mish() - Tanh failed")
	}
	return HadamardProd(a, tsp)
}

// Private functions

func containsDuplicate(slice []int) bool {
	if nil == slice {
		return false
	}

	for index1, value1 := range slice {
		for index2, value2 := range slice {
			if (value1 == value2) && (index1 != index2) {
				return true
			}
		}
	}

	return false
}
