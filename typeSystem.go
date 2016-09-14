package gorgonia

import "fmt"

// Type represents the type of data that a value, or a variable contains. There is a type-analysis
// phase that happens in the graph as the graph is compiled.
//
// Gorgonia graphs are strictly and strongly typed. This means
// every expression HAS to have a type. It happens all behind the scenes though. But it also means no funny
// type coercion kind of business.
//
// The type system is an extremely simplistic type system - it does basic type inference, and has a lot of hacks
// for atomic types. The concept of a abstract data type is halfbaked in here as well, but it works well enough.
type Type interface {
	dims() int
	isScalar() bool
	fmt.Stringer
}

// a typeVariable represents any possible type.
// The equality of a type variable is compared by its name and its name only
type typeVariable struct {
	name        string
	instance    Type
	constraints typeClassSet
}

type typeVarConsOpt func(tv *typeVariable)

func withTVConstraints(cs ...typeClass) typeVarConsOpt {
	set := newTypeClassSet(cs...)
	f := func(tv *typeVariable) {
		tv.constraints = set
	}
	return f
}

func newTypeVariable(name string, opts ...typeVarConsOpt) *typeVariable {
	// retVal := &typeVariable{
	// 	name: name,
	// }
	retVal := borrowTypeVar()
	retVal.name = name
	for _, opt := range opts {
		opt(retVal)
	}
	return retVal
}

func (t *typeVariable) isScalar() bool {
	if t.instance != nil {
		return t.instance.isScalar()
	}
	return false
}

func (t *typeVariable) dims() int {
	if t.instance != nil {
		return t.instance.dims()
	}
	return -1
}

// checks if a typeVariable occurs in a type expression
func (t *typeVariable) in(t0 Type) bool {
	pruned := prune(t0)
	if ptv, ok := pruned.(*typeVariable); ok && t.eq(ptv) {
		return true
	}

	if op, ok := pruned.(typeOp); ok {
		ts := op.types()
		if len(ts) == 1 {
			defer returnTypes1(ts)
		}
		return t.inTypes(ts)
	}
	return false
}

// checks if a type's variable occurs in an array of types
func (t *typeVariable) inTypes(ts Types) bool {
	for _, typ := range ts {
		if t.in(typ) {
			return true
		}
	}
	return false
}

// equality - checks by name and by instance. This simplifies a lot of the environment requirements.
// it's assumed that there is only one env
func (t *typeVariable) eq(other *typeVariable) bool {
	if t.name != other.name {
		return false
	}

	if t.instance != nil && other.instance != nil {
		if !typeEq(t.instance, other.instance) {
			panic(fmt.Sprintf("t %v != other %v: different instances, but same name: %q", t, other, t.name))
		}
	}

	return true
}

func (t *typeVariable) String() string {
	if t.instance != nil {
		return t.instance.String()
	}
	if t.name == "" {
		return "''"
	}
	return t.name
}

// a typeOp is a type constructor that builds new types out of old types
type typeOp interface {
	types() Types
	name() *typeVariable

	setName(*typeVariable)
	setTypes(...Type)

	Type
}

func unify(t1, t2 Type) error {
	typeSysLogf("unifying %v(%T|%p) and %v(%T|%p)", t1, t1, t1, t2, t2, t2)
	enterLoggingContext()
	defer leaveLoggingContext()

	a := prune(t1)
	b := prune(t2)

	switch at := a.(type) {
	case *typeVariable:
		return unifyVar(at, b)
	case typeOp:
		switch bt := b.(type) {
		case *typeVariable:
			return unifyVar(bt, at)
		case typeOp:
			typeSysLogf("typeop")
			aname := at.name()
			bname := bt.name()
			atypes := at.types()
			btypes := bt.types()

			if len(atypes) == 1 {
				defer returnTypes1(atypes)
			}
			if len(btypes) == 1 {
				defer returnTypes1(btypes)
			}

			if aname != nil && len(atypes) > 0 {
				at.setName(bname)
				at.setTypes(btypes...)
				if err := unify(at, bt); err != nil {
					return err
				}
			} else if bname != nil {
				if err := unify(bname, at); err != nil {
					return err
				}
			} else if (aname != nil && bname != nil && !aname.eq(bname)) || len(atypes) != len(btypes) {
				return NewError(TypeError, "Type mismatch: %v != %v", a, b)
			}

			for i, at := range atypes {
				bt := btypes[i]
				if err := unify(at, bt); err != nil {
					return err
				}
			}
			leaveLoggingContext()
		case Dtype:
			if err := unify(b, at); err != nil {
				typeSysLogf("b: %v, at: %v; a : %v", b, at, a)
				return err
			}
		default:
			return NewError(NotYetImplemented, "b of type %T", b)
		}
	case Dtype:
		switch bt := b.(type) {
		case *typeVariable:
			return unifyVar(bt, at)
		case Dtype:
			if at != bt {
				return NewError(TypeError, "Type mismatch: %v is not the same as %v", at, bt)
			}

		default:
			return NewError(TypeError, "Type mismatch: %v(%p) is not the same as %v(%p)", at, at, b, b)
		}
	default:
		return NewError(TypeError, "Types %v and %v are not unifiable", t1, t2)
	}
	return nil
}

func unifyVar(tv *typeVariable, t Type) error {
	if ttv, ok := t.(*typeVariable); ok && !tv.eq(ttv) {
		u := tv.constraints.Union(tv.constraints)
		ttv.constraints = u
		tv.constraints = u
	}

	if tv.in(t) {
		return NewError(TypeError, "type %v will cause a recursive unification with %v", tv, t)
	}

	tv.instance = t
	return nil
}

// prune returns the currently defining instance of type T
func prune(t Type) Type {
	if tv, ok := t.(*typeVariable); ok {
		if tv.instance != nil {
			tv.instance = prune(tv.instance)
			return tv.instance
		}
	}
	return t
}

func pruneCompletely(t Type) Type {
	switch tt := t.(type) {
	case Dtype:
		return tt
	case *TensorType:
		tt.of = pruneCompletely(tt.of)
		return tt
	case *typeVariable:
		if tt.instance != nil {
			defer returnTypeVar(tt)
			return pruneCompletely(tt.instance)
		}
		return tt
	}
	panic("Unreachable")
}

func typeEq(a, b Type) bool {
	switch at := a.(type) {
	case *typeVariable:
		if bt, ok := b.(*typeVariable); ok {
			return at.eq(bt)
		}
		return false
	case *functionType:
		return false // functionTypes are not comparable for now
	case *TensorType:
		var bt *TensorType
		var ok bool
		if bt, ok = b.(*TensorType); !ok {
			return false
		}

		if bt.d != at.d {
			return false
		}

		// shape is part of the type, but mostly only at runtime,
		// so we'll only do a customary check if both shapes are not nil
		if at.shape != nil && bt.shape != nil {
			if !at.shape.Eq(bt.shape) {
				return false
			}
		}

		if !typeEq(prune(at.of), prune(bt.of)) {
			return false
		}

		return true
	case Dtype:
		if bt, ok := b.(Dtype); ok {
			return at == bt
		}
		return false
	}
	// return false in case a or b is nil (which is possible when there is literally no type (i.e. statments))
	return false
}

func dtypeOf(t Type) (retVal Dtype, err error) {
	pruned := prune(t)
	switch p := pruned.(type) {
	case Dtype:
		retVal = p
	case *TensorType:
		return dtypeOf(p.of)
	case *typeVariable:
		if p.instance == nil {
			err = NewError(typeError, "instance %v does not have a dtype", p)
		}

		return dtypeOf(p.instance)
	default:
		err = NewError(NotYetImplemented, "dtypeOf of %v not yet implemented", t)
		return
	}

	return

}

func runtimeTypeCheck(expected, got Types) (of Dtype, err error) {
	if len(expected) != len(got) {
		err = NewError(RuntimeError, "Input length mismatch")
		return
	}

	if of, err = dtypeOf(expected[0]); err != nil {
		return
	}

	for i, e := range expected {
		g := got[i]
		if !typeEq(e, g) {
			err = NewError(RuntimeError, "Expected input[%d] to be %v. Got %v instead", i, e, got[i])
			return
		}

		if i > 0 {
			var gdt Dtype
			if gdt, err = dtypeOf(g); err == nil {
				if gdt != of {
					err = NewError(RuntimeError, "Different dtypes encountered... Expected %v. Got %v instead", of, gdt)
					return
				}
			} else {
				return
			}
		}
	}
	return
}
