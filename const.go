package gorgonia

import (
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
	anyToValueFail      = "Failed to convert %v(%T) into a Value"
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
	onef32   = NewConstant(float32(1.0))
	onef64   = NewConstant(float64(1.0))
	zerof32  = NewConstant(float32(0.0))
	zerof64  = NewConstant(float64(0.0))
	twof64   = NewConstant(float64(2.0))
	twof32   = NewConstant(float32(2.0))
	threef64 = NewConstant(float64(3.0))
	threef32 = NewConstant(float32(3.0))
	ln2f64   = NewConstant(math.Ln2)
	ln2f32   = NewConstant(float32(math.Ln2))

	onef32ConstOp  = onef32.op.(constant)
	onef64ConstOp  = onef64.op.(constant)
	zerof32ConstOp = zerof32.op.(constant)
	zerof64ConstOp = zerof64.op.(constant)

	constmap map[string]map[tensor.Dtype]*Node
)

var oneone = tensor.Shape{1, 1}

func init() {
	constmap = map[string]map[tensor.Dtype]*Node{
		"zero": {
			Float32: zerof32,
			Float64: zerof64,
		},
		"one": {
			Float32: onef32,
			Float64: onef64,
		},
		"two": {
			Float32: twof32,
			Float64: twof64,
		},
		"three": {
			Float32: threef32,
			Float64: threef64,
		},
		"log2": {
			Float32: ln2f32,
			Float64: ln2f64,
		},
	}

}
