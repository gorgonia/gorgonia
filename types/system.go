package types

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type dependentType interface {
	ResolveDepends() hm.Type
}

// Infer infers the application of the children on the opType. Note that children should already match the arity of the opType.
//
// Example:
// 	opType: a → a → a
// 	children: [Float64, Float64, b] // note that `b` has to already been passed in.
func Infer(opType hm.Type, children ...hm.Type) (retVal hm.Type, err error) {

	last := children[len(children)-1]
	b, ok := last.(hm.TypeVariable) // check that the last thing is a variable
	if !ok {
		return nil, errors.Errorf("Expected the last child type to be a variable. Got %v of %T instead", last, last)
	}

	fn2 := hm.NewFnType(children...)
	defer hm.ReturnFnType(fn2)

	var sub hm.Subs
	if sub, err = hm.Unify(opType, fn2); err != nil {
		return nil, errors.Wrapf(err, "Unable to infer %v @ %v", opType, fn2)
	}

	if retVal, ok = sub.Get(b); !ok {
		return nil, errors.Errorf("Expected a replacement for the variable %v", b)
	}
	if dep, ok := retVal.(dependentType); ok {
		return dep.ResolveDepends(), nil

	}
	return retVal, nil
}
