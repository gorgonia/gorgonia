package gorgonia

import (
	"fmt"
	"math"

	"gorgonia.org/tensor"
)

const (
	// graphviz name for a full graph
	fullGraphName = "fullGraph"

	// group names
	exprgraphClust = "expressionGraph"
	constantsClust = "constants"
	inputsClust    = "inputs"
	gradClust      = "gradients"
	strayClust     = "undifferentiated nodes"

	// subgraphs to rank the same
	outsideSubG = "outsides"
	inputConsts = "inputConsts"

	// special nodes for graphviz hacking
	outsideRoot   = "outsideRoot"
	outsideInputs = "outsideInputs"
	insideInputs  = "insideInputs"
	outsideConsts = "outsideConsts"
	insideConsts  = "insideConsts"
	outsideExprG  = "outsideExprG"
	insideExprG   = "insideExprG"
	outsideGrads  = "outsideGrads"
	insideGrads   = "insideGrads"

	// error messages
	sortFail            = "Failed to sort"
	cloneFail           = "Failed to carry clone(%v)"
	clone0Fail          = "Failed to carry clone0()"
	nyiTypeFail         = "%s not yet implemented for %T"
	nyiFail             = "%s not yet implemented for %v"
	dtypeOfFail         = "Failed to carry dtypeOf()"
	mulFail             = "Failed to carry Mul()"
	applyOpFail         = "Failed to carryApplyOp()"
	opDoFail            = "Failed to carry op.Do()"
	binOpDoFail         = "Failed to carry binOp.Do()"
	binOpNodeFail       = "Failed to carry binary operation %T"
	applyFail           = "Failed to carry Apply()"
	binOpFail           = "Binary operator received %d arguments"
	hadamardProdFail    = "Failed to carry hadamardProd()"
	hadamardDivFail     = "Failed to carry hadamardDiv()"
	cubeFail            = "Failed to carry cube()"
	negFail             = "Failed to carry Neg()"
	invFail             = "Failed to carry Inv()"
	pointWiseMulFail    = "Failed to carry PointWiseMul()"
	pointWiseSquareFail = "Failed to carry PointWiseSquare()"
	clampFail           = "Failed to carry Clamp()"
	invSqrtFail         = "Failed to carry InvSqrt()"
	subFail             = "Failed to carry Sub()"
	addFail             = "Failed to carry Add()"
	signFail            = "Failed to carry Sign()"
	softplusFail        = "Failed to carry Softplus()"
	incrErr             = "increment couldn't be done. Safe op was performed instead"
	bindFail            = "Failed to bind"
	anyToValueFail      = "Failed to convert %v(%T) into a value.Value"
	dtypeExtractionFail = "Failed to extract dtype from %v"
	operationError      = "Operation failed"
	doFail              = "Doing %v failed"
	unsafeDoFail        = "UnsafeDoing %v failed."
	tFail               = "Failed to transpose Tensor"
	repFail             = "Failed to repeat Tensor along %d %d times"
	reshapeFail         = "Failed to reshape Tensor into %v. DataSize was: %d"
	sliceFail           = "Failed to slice Tensor with %v"
	execFail            = "Failed to execute %v in node %v"
	autodiffFail        = "Failed to differentiate %v"
	undefinedOnShape    = "%v undefined on shape %v"
	unsupportedDtype    = "dtype %v is not yet supported"
	gradOnDeviceFail    = "Cannot get gradient of %v on %v"
	makeValueFail       = "Unable to make value of %v with shape %v"
	allocFail           = "Unable to allocate %v bytes on %v"
)

var empty struct{}

var (
	onef32   = func(g *ExprGraph) *Node { return g.NewConstant(float32(1.0)) }
	onef64   = func(g *ExprGraph) *Node { return g.NewConstant(float64(1.0)) }
	zerof32  = func(g *ExprGraph) *Node { return g.NewConstant(float32(0.0)) }
	zerof64  = func(g *ExprGraph) *Node { return g.NewConstant(float64(0.0)) }
	twof64   = func(g *ExprGraph) *Node { return g.NewConstant(float64(2.0)) }
	twof32   = func(g *ExprGraph) *Node { return g.NewConstant(float32(2.0)) }
	threef64 = func(g *ExprGraph) *Node { return g.NewConstant(float64(3.0)) }
	threef32 = func(g *ExprGraph) *Node { return g.NewConstant(float32(3.0)) }
	ln2f64   = func(g *ExprGraph) *Node { return g.NewConstant(math.Ln2) }
	ln2f32   = func(g *ExprGraph) *Node { return g.NewConstant(float32(math.Ln2)) }

	onef32ConstOp  = func(g *ExprGraph) constant { return onef32(g).op.(constant) }
	onef64ConstOp  = func(g *ExprGraph) constant { return onef64(g).op.(constant) }
	zerof32ConstOp = func(g *ExprGraph) constant { return zerof32(g).op.(constant) }
	zerof64ConstOp = func(g *ExprGraph) constant { return zerof64(g).op.(constant) }

	constmap map[string]map[tensor.Dtype]*Node
)

var oneone = tensor.Shape{1, 1}

func constants(g *ExprGraph, s string, dt tensor.Dtype) (*Node, error) {
	switch s {
	case "zero":
		switch dt {
		case Float32:
			return zerof32(g), nil
		case Float64:
			return zerof64(g), nil
		default:
			return nil, fmt.Errorf("Constant %v not provided for %v", s, dt)
		}
	case "one":
		switch dt {
		case Float32:
			return onef32(g), nil
		case Float64:
			return onef64(g), nil
		default:
			return nil, fmt.Errorf("Constant %v not provided for %v", s, dt)
		}
	case "two":
		switch dt {
		case Float32:
			return twof32(g), nil
		case Float64:
			return twof64(g), nil
		default:
			return nil, fmt.Errorf("Constant %v not provided for %v", s, dt)
		}
	case "three":
		switch dt {
		case Float32:
			return threef32(g), nil
		case Float64:
			return threef64(g), nil
		default:
			return nil, fmt.Errorf("Constant %v not provided for %v", s, dt)
		}
	case "log2":
		switch dt {
		case Float32:
			return ln2f32(g), nil
		case Float64:
			return ln2f64(g), nil
		default:
			return nil, fmt.Errorf("Constant %v not provided for %v", s, dt)
		}
	default:
		return nil, fmt.Errorf("Constant %v not provided for %v", s, dt)
	}
	return nil, nil
}

var (
	// Represents the types that Nodes can take in Gorgonia

	// Float64 ...
	Float64 = tensor.Float64
	// Float32 ...
	Float32 = tensor.Float32
	// Int ...
	Int = tensor.Int
	// Int64 ...
	Int64 = tensor.Int64
	// Int32 ...
	Int32 = tensor.Int32
	// Byte ...
	Byte = tensor.Uint8
	// Bool ...
	Bool = tensor.Bool

	// Ptr ...
	Ptr = tensor.UnsafePointer // equivalent to interface{}. Ugh Ugh Ugh

)

var acceptableDtypes = [...]tensor.Dtype{tensor.Float64, tensor.Float32, tensor.Int, tensor.Int64, tensor.Int32, tensor.Byte, tensor.Bool}
