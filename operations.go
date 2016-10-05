package gorgonia

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

// contains all public operations that can be performed on nodes
// all the functions here have the signature:
// 		func (...) (*Node, error)

func applyOpWithName(op Op, name string, children ...*Node) (retVal *Node, err error) {
	if retVal, err = applyOp(op, children...); err == nil {
		WithName(name)(retVal)
	}
	return
}

// Generic apply function... for when you don't need to specialize
func applyOp(op Op, children ...*Node) (retVal *Node, err error) {
	var g *ExprGraph

	for _, child := range children {
		if child.g != nil {
			g = child.g
			break
		}
	}

	if g == nil {
		err = NewError(GraphError, "No Graph Supplied")
		return
	}

	if !Nodes(children).AllSameGraph() {
		err = NewError(GraphError, "Not all children have the same graph")
		return
	}

	// typecheck  before creating
	typeSysLogf("Inferring node type of %v with children: %#Y", op, Nodes(children))
	var retType Type
	if retType, err = inferNodeType(op, children...); err != nil {
		err = errors.Wrapf(err, "Type inference error. Op: %v. Children: %#Y, OpType:%v", op, Nodes(children), op.Type())
		return
	}
	// retType = pruneCompletely(retType)
	typeSysLogf("Done inferring. Return type is: %v %#v", retType, retType)

	// infer shapes, but print errors instead of returning
	shapeLogf("op: %v(%T) inferring shape", op, op)
	var s types.Shape
	if s, err = op.inferShape(retType, children...); err == nil {
		typeSysLogf("inferred type: %v", retType)
		shapeLogf("inferred shape %v", s)
		retVal = newUniqueNode(withType(retType), withOp(op), withChildren(children), withGraph(g), WithShape(s...))
	} else {
		retVal = newUniqueNode(withType(retType), withOp(op), withChildren(children), withGraph(g))
	}
	return
}

/* BINARY FUNCTIONS */
func binOpNode(op BinaryOp, a, b *Node) (retVal *Node, err error) {
	stabLogf("Creating node for %v, a: %p, b: %p", op, a, b)
	enterLoggingContext()
	defer leaveLoggingContext()
	// maybe make stabilization a build tag?
	if stabilization {
		enterLoggingContext()
		if ebo, ok := op.(elemBinOp); ok {
			ot := ebo.binOpType()

			enterLoggingContext()
			for _, fn := range binOpStabilizationFns[ot] {
				if retVal, err = fn(a, b); err == nil {
					leaveLoggingContext()
					return
				}

				if _, ok := err.(errNoStabilization); !ok {
					leaveLoggingContext()
					return
				}
				err = nil // reset err
			}
			leaveLoggingContext()
		}
		leaveLoggingContext()
	}
	stabLogf("No bin op stabilization")

	return applyOp(op, a, b)
}

// Add: pointwise a + b
func Add(a, b *Node) (retVal *Node, err error) {
	op := newElemBinOp(addOpType, a, b)
	return binOpNode(op, a, b)
}

// Sub: pointwise a - b
func Sub(a, b *Node) (retVal *Node, err error) {
	op := newElemBinOp(subOpType, a, b)
	return binOpNode(op, a, b)
}

// HadamardProd: pointwise a * b
func HadamardProd(a, b *Node) (retVal *Node, err error) {
	op := newElemBinOp(mulOpType, a, b)
	return binOpNode(op, a, b)
}

// Mul is the general handler for multiplication of nodes. It is extremely overloaded. Only use if you know what you're doing
//
// If any of the nodes are ScalarType, then it'll be redirected to HadamardMul() instead
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
	case a.IsVector() || b.IsVector():
		if a.IsVector() {
			// b is matrix
			op = linAlgBinOp{
				āBinaryOperator: matVecMulOperator,
				transA:          true,
			}

			// we return early because b is a matrix
			// the inputs are swapped
			return binOpNode(op, b, a)
		} else {
			// a is matrix
			op = linAlgBinOp{āBinaryOperator: matVecMulOperator}
		}
	default:
		// TODO: maybe align shapes?
		op = linAlgBinOp{āBinaryOperator: matMulOperator}
	}

	return binOpNode(op, a, b)
}

func OuterProd(a, b *Node) (retVal *Node, err error) {
	if !a.IsVector() || !b.IsVector() {
		err = NewError(GraphError, "Expected only vectors to be able to do OuterProd") //for now
		return
	}

	// TODO: maybe align shapes?
	op := linAlgBinOp{āBinaryOperator: outerProdOperator}
	return binOpNode(op, a, b)
}

// HadamardDiv: pointwise a / b
func HadamardDiv(a, b *Node) (retVal *Node, err error) {
	op := newElemBinOp(divOpType, a, b)
	return binOpNode(op, a, b)
}

func Div(a, b *Node) (retVal *Node, err error) {
	if a.IsScalar() || b.IsScalar() {
		return HadamardDiv(a, b)
	}

	// otherwise, matrix division
	panic("Unhandled")
}

// Gt: pointwise a > b. retSame indicates if the return value should be the same type as the input values
func Gt(a, b *Node, retSame bool) (retVal *Node, err error) {
	op := newElemBinOp(gtOpType, a, b)
	op.retSame = retSame
	return binOpNode(op, a, b)
}

// Gte: pointwise a >= b. retSame indicates if the return value should be the same type as the input values
func Gte(a, b *Node, retSame bool) (retVal *Node, err error) {
	op := newElemBinOp(gteOpType, a, b)
	op.retSame = retSame
	return binOpNode(op, a, b)
}

/* UNARY STUFF */

func unaryOpNode(op Op, a *Node) (retVal *Node, err error) {
	stabLogf("Creating node for %v, a: %p %v", op, a, a)
	enterLoggingContext()
	defer leaveLoggingContext()
	if stabilization {

		// do optimization/stabilization
		// TODO: maybe recursively stabilize?
		enterLoggingContext()
		ot := op.(elemUnaryOp).unaryOpType()
		for _, fn := range unaryOpStabilizationFns[ot] {
			if retVal, err = fn(a); err == nil {
				stabLogf("stabilized")
				leaveLoggingContext()
				return
			}

			if _, ok := err.(errNoStabilization); !ok {
				stabLogf("Actual error")
				leaveLoggingContext()
				return
			} else {
				stabLogf("No stabilization found")
				err = nil // reset err
			}
		}
		leaveLoggingContext()
		stabLogf("No stabilizations - retVal: %v", retVal)
	}

	return applyOp(op, a)
}

// Abs: |a|
func Abs(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(absOpType, a)
	return unaryOpNode(op, a)
}

// Sign: pointwise sign function. -1 for a negative, +1 for positive
func Sign(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(signOpType, a)
	return unaryOpNode(op, a)
}

func Ceil(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(ceilOpType, a)
	return unaryOpNode(op, a)
}

func Floor(a *Node) (retval *Node, err error) {
	op := newElemUnaryOp(floorOpType, a)
	return unaryOpNode(op, a)
}

func Sin(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(sinOpType, a)
	return unaryOpNode(op, a)
}

func Cos(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(cosOpType, a)
	return unaryOpNode(op, a)
}

func Exp(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(expOpType, a)
	return unaryOpNode(op, a)
}

func Log(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(lnOpType, a)
	return unaryOpNode(op, a)
}

func Log2(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(log2OpType, a)
	return unaryOpNode(op, a)
}

func Neg(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(negOpType, a)
	return unaryOpNode(op, a)
}

func Square(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(squareOpType, a)
	return unaryOpNode(op, a)
}

func Sqrt(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(sqrtOpType, a)
	return unaryOpNode(op, a)
}

func Inverse(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(inverseOpType, a)
	return unaryOpNode(op, a)
}

func Cube(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(cubeOpType, a)
	return unaryOpNode(op, a)
}

func Sigmoid(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(sigmoidOpType, a)
	return unaryOpNode(op, a)
}

func Tanh(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(tanhOpType, a)
	return unaryOpNode(op, a)
}

func Log1p(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(log1pOpType, a)
	return unaryOpNode(op, a)
}

// more complex unaries

func SoftMax(a *Node) (retVal *Node, err error) {
	var exp, sum *Node
	if exp, err = Exp(a); err == nil {
		axis := 1 // default
		if exp.IsColVec() || (exp.IsVector() && !exp.IsRowVec()) {
			axis = 0
		}

		if sum, err = Sum(exp, axis); err == nil {
			if sum.IsScalar() {
				return HadamardDiv(exp, sum)
			}
			return Broadcast(divOpType, exp, sum, NewBroadcastPattern(nil, []byte{1}))
		}
	}
	return
}

func StableSoftMax(a *Node) (retVal *Node, err error) {
	var max, exp, sum *Node
	if max, err = Max(a); err != nil {
		err = errors.Wrap(err, operationError)
		return
	}
	if retVal, err = Sub(a, max); err == nil {
		if exp, err = Exp(retVal); err == nil {
			if sum, err = Sum(exp, 1); err == nil {
				return HadamardDiv(exp, sum)
			}
		}
	}
	return
}

func Softplus(a *Node) (retVal *Node, err error) {
	op := newElemUnaryOp(softplusOpType, a)
	return unaryOpNode(op, a)
}

/* Aggregate Functions */

func At(a *Node, coords ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		for _, c := range coords {
			if c != 0 {
				err = NewError(GraphError, "At() only works with scalars when the coordinates are (0...0). Got %v instead", coords)
				return
			}
		}
		return a, nil
	}

	dims := a.Dims()
	op := atOp{
		coordinates: coords,
		d:           dims,
	}

	return applyOp(op, a)
}

func Max(a *Node, along ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		// can't max a scalar. Should return error
		// err = NewError(TypeError, "Cannot Max a Scalar")
		return a, nil
	}

	dims := a.Dims()
	if len(along) == 0 {
		along = intRange(0, dims)
	}

	op := newMaxOp(along, dims)

	return applyOp(op, a)
}

func Mean(a *Node, along ...int) (retVal *Node, err error) {
	if a.IsScalar() {
		// can't mean a scalar... return error
		// err = NewError(TypeError, "Cannot mean a scalar value")
		// return
		return a, nil
	}

	dims := a.Dims()

	if len(along) == 0 {
		along = intRange(0, dims)
	}

	var s *Node
	if s, err = Sum(a, along...); err != nil {
		err = errors.Wrap(err, operationError)
		return
	}

	sizes := make(Nodes, len(along))
	for i, axis := range along {
		if sizes[i], err = SizeOf(axis, a); err != nil {
			err = errors.Wrap(err, operationError)
			return
		}
	}

	var counts *Node
	if counts, err = ReduceMul(sizes); err == nil {
		retVal, err = HadamardDiv(s, counts)
	}
	return
}

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
		case a.IsColVec(), a.IsVector():
			along = []int{0}
		default:
			along = intRange(0, dims)
		}
	}

	op := newSumOp(along, a.shape, dims)
	return applyOp(op, a)
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
		return
	case 1:
		return nodes[0], nil
	case 2:
		if retVal, err = Mul(nodes[0], nodes[1]); err == nil {
			for _, opt := range opts {
				opt(retVal)
			}
		}
		return
	}

	retVal = nodes[0]
	for i, n := range nodes {
		if i == 0 {
			continue
		}

		if retVal, err = Mul(retVal, n); err != nil {
			err = errors.Wrap(err, operationError)
			return
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

	return applyOp(op, x)
}

// Slice slices a *Node. For T[:] slices, pass in nil. Will error out if node's type is not a Tensor
func Slice(n *Node, slices ...types.Slice) (retVal *Node, err error) {
	if _, ok := n.t.(*TensorType); !ok {
		err = NewError(GraphError, "Cannot slice on non Tensor types. Got %T", n.t)
		return
	}

	retVal = n
	for i, slice := range slices {
		var op sliceOp

		// a nil slice represents ":"
		if slice == nil {
			// op = newSliceOp(0, -1, i, retVal.Dims())
			continue
		} else {
			op = newSliceOp(slice, i, retVal.Dims())
		}

		if retVal, err = applyOp(op, retVal); err != nil {
			err = errors.Wrap(err, operationError)
			return
		}
	}
	return
}

func Transpose(n *Node, axes ...int) (retVal *Node, err error) {
	// prep axes
	if len(axes) > 0 && len(axes) != n.Dims() {
		err = NewError(ShapeError, "n has %d dims, while requested transposes is %d", n.Dims(), len(axes))
		return
	}
	dims := len(n.shape)
	if len(axes) == 0 || axes == nil {
		axes = make([]int, dims)
		for i := 0; i < dims; i++ {
			axes[i] = dims - 1 - i
		}
	}

	// if axes is 0, 1, 2, 3... then no op
	if monotonic, incr1 := types.IsMonotonicInts(axes); monotonic && incr1 && axes[0] == 0 {
		retVal = n
		return
	}
	op := transposeOp{
		pattern: axes,
		d:       len(axes),
	}

	return applyOp(op, n)
}
