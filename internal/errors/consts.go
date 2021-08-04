package gerrors

const (
	CloneFail   = "Failed to clone Value"
	OpDoFail    = "Failed to carry op.Do"
	NYITypeFail = "%s Not Yet Implemented for %T"
	NYIFail     = "%s Not Yet Implemented for %v"

	TypeMismatch  = "Type Mismatch: a %T and b %T"
	ShapeMismatch = "Shape Mismatch. Expected %v. Got %v"

	SymbolicOpFail = "Failed to perform %v symbolically"

	noopMsg = "NoOp"
)

type NoOp struct{}

func (err NoOp) Error() string { return noopMsg }
func (err NoOp) NoOp()         {}
