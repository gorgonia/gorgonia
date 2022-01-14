package types

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type dependentType interface {
	ResolveDepends() hm.Type
}

type canonizable interface {
	Canonical() hm.Type
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

// NewFunc creates a new *hm.FunctionType.
// The reason for using NewFunc over hm.NewFnType is that NewFunc
// will handle the canonicalization of types
// e.g. TensorType{0, TypeVariable('a')} is defined to be equal to TypeVariable('a'),
// so we canonicalize the TensorType to be TypeVariable('a').
// The canonicalization is outside the remit of the hm package, so it's done here instead.
func NewFunc(ts ...hm.Type) *hm.FunctionType {
	for i := range ts {
		if c, ok := ts[i].(canonizable); ok {
			ts[i] = c.Canonical()
		}
	}
	return hm.NewFnType(ts...)
}
