package gorgonia

import (
	"fmt"
	"math"

	"github.com/chewxy/math32"
)

var (
	/* float64 */

	// non differentiable
	absf64   = sf64UnaryOperator(math.Abs)
	signf64  = sf64UnaryOperator(_signf64)
	ceilf64  = sf64UnaryOperator(math.Ceil)
	floorf64 = sf64UnaryOperator(math.Floor)

	// differentiable
	sinf64         = sf64UnaryOperator(math.Sin)
	cosf64         = sf64UnaryOperator(math.Cos)
	expf64         = sf64UnaryOperator(math.Exp)
	lnf64          = sf64UnaryOperator(math.Log)
	log2f64        = sf64UnaryOperator(math.Log2)
	negf64         = sf64UnaryOperator(_negf64)
	squaref64      = sf64UnaryOperator(_squaref64)
	sqrtf64        = sf64UnaryOperator(math.Sqrt)
	inversef64     = sf64UnaryOperator(_inversef64)
	inverseSqrtf64 = sf64UnaryOperator(_inverseSqrtf64)

	// activation functions
	cubef64    = sf64UnaryOperator(_cubef64)
	tanhf64    = sf64UnaryOperator(_tanhf64)
	sigmoidf64 = sf64UnaryOperator(_sigmoidf64)

	// numerical stabilization optimization
	log1pf64    = sf64UnaryOperator(math.Log1p)
	expm1f64    = sf64UnaryOperator(math.Expm1)
	softplusf64 = sf64UnaryOperator(_softplusf64)
	// softplus isn't necessarily only a numerical stabilization op
	// (you can use it elsewhere), but I included it under numerical optimization

	/* Float32 */

	// non differentiable
	absf32   = sf32UnaryOperator(math32.Abs)
	signf32  = sf32UnaryOperator(_signf32)
	ceilf32  = sf32UnaryOperator(math32.Ceil)
	floorf32 = sf32UnaryOperator(math32.Floor)

	// start differentiable
	sinf32         = sf32UnaryOperator(math32.Sin)
	cosf32         = sf32UnaryOperator(math32.Cos)
	expf32         = sf32UnaryOperator(math32.Exp)
	lnf32          = sf32UnaryOperator(math32.Log)
	log2f32        = sf32UnaryOperator(math32.Log2)
	negf32         = sf32UnaryOperator(_negf32)
	squaref32      = sf32UnaryOperator(_squaref32)
	sqrtf32        = sf32UnaryOperator(math32.Sqrt)
	inversef32     = sf32UnaryOperator(_inversef32)
	inverseSqrtf32 = sf32UnaryOperator(_inverseSqrtf32)

	// typically used in activation functions
	cubef32    = sf32UnaryOperator(_cubef32)
	tanhf32    = sf32UnaryOperator(_tanhf32)
	sigmoidf32 = sf32UnaryOperator(_sigmoidf32)

	// numerical stabilization optimization
	log1pf32    = sf32UnaryOperator(math32.Log1p)
	expm1f32    = sf32UnaryOperator(math32.Expm1)
	softplusf32 = sf32UnaryOperator(_softplusf32)
)

type ʘUnaryOperatorType byte

const (
	absOpType ʘUnaryOperatorType = iota
	signOpType
	ceilOpType
	floorOpType

	// start differentiable
	sinOpType
	cosOpType
	expOpType
	lnOpType
	log2OpType
	negOpType
	squareOpType
	sqrtOpType
	inverseOpType     // multiplicative inverse
	inverseSqrtOpType // 1/sqrt(x)

	// typically used in activation functions
	cubeOpType
	tanhOpType
	sigmoidOpType

	// optimization related
	log1pOpType
	expm1OpType
	softplusOpType

	maxʘUnaryOperator // delimits end of all possible unary ops
)

func (u ʘUnaryOperatorType) String() string {
	if u >= maxʘUnaryOperator {
		return fmt.Sprintf("UNSUPPORTED UNARY OPERATOR (%d); max: %d", u, maxʘUnaryOperator)
	}

	return ʘUnaryOpStrs[u]
}

// ʘUnaryOpStrs is the string representation for a unaryOpType
// It should be held constant.
var ʘUnaryOpStrs = [maxʘUnaryOperator]string{
	"abs", "sign", "ceil", "floor",
	"sin", "cos", "exp",
	"ln", "log2", "neg", "square", "sqrt",
	"inv", "invSqrt",
	"cube", "tanh", "sigmoid",

	"log1p", "expm1", "softplus",
}

// ʘUnaryOpDifferentiable is the array of whether a unary operator is differentiable
// It should be held constant
var ʘUnaryOpDifferentiable = [maxʘUnaryOperator]bool{
	true, false, false, false,
	true, true, true,
	true, true, true, true, true,
	true, true,
	true, true, true,

	true, true, true,
}

var ʘUnaryOpDiffExprs = [maxʘUnaryOperator]func(x, y, gradY *Node) (*Node, error){
	absDiffExpr, nondiffUnaryOpExpr, nondiffUnaryOpExpr, nondiffUnaryOpExpr,
	sinDiffExpr, cosDiffExpr, expDiffExpr,
	lnDiffExpr, log2DiffExpr, negDiffExpr, squareDiffExpr, sqrtDiffExpr,
	inverseDiffExpr, inverseSqrtDiffExpr, cubeDiffExpr, tanhDiffExpr, sigmoidDiffExpr,

	log1pDiffExpr, expm1DiffExpr, softplusDiffExpr,
}

var ʘUnaryOpDiffFns = [maxʘUnaryOperator]func(x, y *Node) error{
	absDiff, nondiffUnaryOp, nondiffUnaryOp, nondiffUnaryOp,
	sinDiff, cosDiff, expDiff,
	lnDiff, log2Diff, negDiff, squareDiff, sqrtDiff,
	inverseDiff, inverseSqrtDiff, cubeDiff, tanhDiff, sigmoidDiff,

	log1pDiff, expm1Diff, softplusDiff,
}

var sf64UnaryOperators = [maxʘUnaryOperator]*sf64UnaryOperator{
	&absf64,
	&signf64,
	&ceilf64,
	&floorf64,
	&sinf64,
	&cosf64,
	&expf64,
	&lnf64,
	&log2f64,
	&negf64,
	&squaref64,
	&sqrtf64,
	&inversef64,
	&inverseSqrtf64,
	&cubef64,
	&tanhf64,
	&sigmoidf64,

	&log1pf64,
	&expm1f64,
	&softplusf64,
}

var sf32UnaryOperators = [maxʘUnaryOperator]*sf32UnaryOperator{
	&absf32,
	&signf32,
	&ceilf32,
	&floorf32,
	&sinf32,
	&cosf32,
	&expf32,
	&lnf32,
	&log2f32,
	&negf32,
	&squaref32,
	&sqrtf32,
	&inversef32,
	&inverseSqrtf32,
	&cubef32,
	&tanhf32,
	&sigmoidf32,

	&log1pf32,
	&expm1f32,
	&softplusf32,
}
