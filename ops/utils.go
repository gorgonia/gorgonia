package ops

import "github.com/pkg/errors"

// CheckArity returns an error if the input number does not correspond to the expected arity
func CheckArity(op Arityer, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return errors.Errorf("%v has an arity of %d. Got %d instead", op, op.Arity(), inputs)
	}
	return nil
}
