package gorgonia

import (
	"math"

	"github.com/chewxy/gorgonia/tensor"
)

// maxInt is the maximum value of the machine-dependent int type.
const maxInt int = int(^uint(0) >> 1)

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
	cloneFail           = "Failed to carry clone()"
	clone0Fail          = "Failed to carry clone0()"
	nyiTypeFail         = "%s not yet implemented for %T"
	nyiFail             = "%s not yet implemented for %v"
	dtypeOfFail         = "Failed to carry dtypeOf()"
	mulFail             = "Failed to carry Mul()"
	applyOpFail         = "Failed to carry applyOp()"
	opDoFail            = "Failed to carry op.Do()"
	binOpDoFail         = "Failed to carry binOp.Do()"
	binOpNodeFail       = "Failed to carry binary operation %T"
	applyFail           = "Failed to carry Apply()"
	binOpFail           = "Binary operator received %d arguments"
	hadamardProdFail    = "Failed to carry hadamardProd()"
	hadamardDivFail     = "Failed to carry hadamardDiv()"
	negFail             = "Failed to carry Neg()"
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
)

var oneone = tensor.Shape{1, 1}
