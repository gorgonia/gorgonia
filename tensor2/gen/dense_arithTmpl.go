package main

import (
	"fmt"
	"io"
	"text/template"
)

type DenseBinOp struct {
	OpName    string
	VecVec    string
	VecScalar string
	ScalarVec string
}

var denseBinOps = []DenseBinOp{
	{"add", "Add", "Trans", "Trans"},
	{"sub", "Sub", "TransInv", "TransInvR"},
	{"mul", "Mul", "Scale", "Scale"},
	{"div", "Div", "ScaleInv", "ScaleInvR"},
}

const densBinOpPrep = `package tensor

import "github.com/pkg/errors"


func prepDDOp(a, b *Dense, opts ...FuncOpt) (an, bn, rn Number, reuse *Dense, safe, toReuse, incr bool, err error) {
	var ok bool
	if an, ok = a.data.(Number); !ok {
		err = noopError{}
		return
	}

	if bn, ok = b.data.(Number); !ok {
		err = noopError{}
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return
	}

	safe, incr, reuseT := parseSafeReuse(opts...)
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot add with reuse")
			return
		}

		if rn, ok = reuse.data.(Number); !ok {
			err = errors.Errorf("Reuse is not a number")
			return
		}
	}
	return
}

func prepSD(a *Dense, opts ...FuncOpt) (an, rn Number, reuse *Dense, safe, toReuse, incr bool, err error) {
	var ok bool
	if an, ok = a.data.(Number); !ok {
		err = noopError{}
		return
	}

	safe, incr, reuseT := parseSafeReuse(opts...)
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot add with reuse")
			return
		}

		if rn, ok = reuse.data.(Number); !ok {
			err = errors.Errorf("Reuse is not a number")
			return
		}
	}
	return

}
`

const denseBinOpRaw = `func {{.OpName}}DD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepDDOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if reuse == b {
			// b + b first, because b will get clobbered
			if err = rn.{{title .OpName}}(bn); err != nil {
				err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to {{.OpName}} Array b to Array reused")
				return
			}

			if err = rn.{{title .OpName}}(an); err != nil {
				err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to {{.OpName}} Array a to Array reused")
				return
			}

			return reuse, nil
		}

		if err = rn.{{title .OpName}}(an); err != nil {
			err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to {{.OpName}} Array a to Array reused")
			return
		}
		if err = rn.{{title .OpName}}(bn); err != nil {
			err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to {{.OpName}} Array b to Array reused")
			return
		}

		return reuse, nil
	case toReuse:
		if _, err = safe{{title .OpName}}(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to {{.OpName}} Array a and Array b to Array reused")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safe{{title .OpName}}(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to safely {{.OpName}} Array a and b to rn")
			return
		}
		return
	case !safe:
		if err = an.{{title .OpName}}(bn); err != nil {
			err = errors.Wrapf(err, opFail, "{{.OpName}}DD. Unable to safely {{.OpName}} Array a to Array reused")
			return
		}
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func {{.OpName}}DS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepSD(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.{{title .OpName}}(an); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}DS. Unable to {{.OpName}} Array a to the reuse")
			return
		}
		if err = rn.{{.VecScalar}}(b); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}DS. Unable to {{.VecScalar}} the Array reuse by b of %T", b)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safe{{.VecScalar}}(an, b, rn); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}DS")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safe{{.VecScalar}}(an, b, rn); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}DS. Unable to safely {{.OpName}} ")
			return
		}
		return
	case !safe:
		err = an.{{.VecScalar}}(b)
		retVal = a
		return
	}
	panic("Unreachable")
}

func {{.OpName}}SD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepSD(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.{{title .OpName}}(bn); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}SD. Unable to {{.OpName}} Array b to the reuse")
			return
		}
		if err = rn.{{.ScalarVec}}(a); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}SD. Unable to {{.VecScalar}} the Array reuse by a of %T", a)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safe{{.ScalarVec}}(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}SD")
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(b.t, b.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safe{{.ScalarVec}}(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "{{.OpName}}SD. Unable to safely {{.OpName}} ")
			return
		}
		return
	case !safe:
		err = bn.{{.ScalarVec}}(a)
		retVal = b
		return
	}
	panic("Unreachable")
}

`

func generateDenseArith(f io.WriteCloser) {
	fmt.Fprintln(f, densBinOpPrep)
	fmt.Fprintln(f, "\n")
	for _, bo := range denseBinOps {
		fmt.Fprintf(f, "/* %v */\n\n", bo.OpName)
		denseBinOpTmpl.Execute(f, bo)
		fmt.Fprintln(f, "\n")
	}
	f.Close()
}

var (
	denseBinOpTmpl *template.Template
)

func init() {
	denseBinOpTmpl = template.Must(template.New("DenseBinOp").Funcs(funcMap).Parse(denseBinOpRaw))
}
