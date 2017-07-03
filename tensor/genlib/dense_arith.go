package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

type ArithBinOp struct {
	Kind   reflect.Kind
	OpName string
	OpSymb string
	IsFunc bool
}

type BinOps struct {
	*ManyKinds
	OpName     string
	OpSymb     string
	CommonName string
	IsFunc     bool
}

type ArithBinOps struct {
	BinOps

	HasIdentity   bool
	Identity      int
	IsCommutative bool
	IsAssociative bool
	IsInv         bool
	InvOpName     string
	InvOpSymb     string
}

var binOps = []struct {
	OpName string
	OpSymb string

	IsFunc bool

	HasIdentity   bool
	Identity      int
	IsCommutative bool
	IsAssociative bool
	IsInv         bool
	InvOpName     string
	InvOpSymb     string
}{
	{"Add", "+", false, true, 0, true, true, false, "", ""},
	{"Sub", "-", false, true, 0, false, false, false, "Add", "+"},
	{"Mul", "*", false, true, 1, true, true, false, "", ""},
	{"Div", "/", false, true, 1, false, false, false, "Mul", "*"},
	{"Pow", "math.Pow", true, false, 1, false, false, false, "", ""},
}

var vecscalarOps = []struct {
	OpName     string
	CommonName string
	OpSymb     string

	IsFunc bool

	HasIdentity   bool
	Identity      int
	IsCommutative bool
	IsAssociative bool
	IsInv         bool
	InvOpName     string
	InvOpSymb     string
}{
	{"Trans", "addition", "+", false, true, 0, true, true, false, "", ""},
	{"TransInv", "subtraction", "-", false, true, 0, false, false, false, "Trans", "+"},
	{"TransInvR", "subtraction", "-", false, false, 0, false, false, false, "", ""},
	{"Scale", "multiplication", "*", false, true, 1, true, true, false, "", ""}, // no good way to test inverse w/o  implicit broadcast
	{"ScaleInv", "division", "/", false, true, 1, false, false, false, "Scale", "*"},
	{"ScaleInvR", "division", "/", false, false, 0, false, false, false, "", ""},          // identity is 0 on purpose
	{"PowOf", "exponentiation", "math.Pow", true, false, 0, false, false, false, "", ""},  // identity is 0 on purpose
	{"PowOfR", "exponentiation", "math.Pow", true, false, 0, false, false, false, "", ""}, // identity is 0 on purpose
}

const prepArithRaw = `func prepBinaryDense(a, b *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t) && !isNumber(b.t) {
		err = noopError{}
		return
	}
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(typeMismatch, a.t, b.t)
		return
	}
	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
		return
	}

	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != a.t.Kind(){
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return 
		}

		if reuse.len() != a.Shape().TotalSize() {
			err =  errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

func prepUnaryDense(a *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t){
		err = noopError{}
		return
	}

	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}
		

		if reuse.t.Kind() != a.t.Kind(){
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return 
		}

		if reuse.len() != a.Shape().TotalSize() {
			err =  errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}
`

const denseDenseArithRaw = `// {{.OpName}} performs the operation on another *Dense. It takes a list of FuncOpts.
func (t *Dense) {{.OpName}}(other *Dense, opts ...FuncOpt) (retVal *Dense, err error){
	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$opName := .OpName -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}

	if t.e != nil {
		if {{lower $opName}}, ok := t.e.({{title $opName}}er); ok {
			// if safe, then make a copy
			var ret Tensor
			if ret, err = {{lower $opName}}.{{title $opName}}(t, other, opts...); err != nil {
				goto attemptGo
			}
			retVal = ret.(*Dense)
			return
		}
	}

attemptGo:
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}


	{{if or $scaleInv $div -}}var errs errorIndices
	{{end -}}
	var it, ot *FlatMaskedIterator
	if t.IsMaterializable() {
		it = NewFlatMaskedIterator(t.AP, t.mask)
	}
	if other.IsMaterializable() {
		ot = NewFlatMaskedIterator(other.AP, other.mask)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		retVal = reuse
		retVal.MaskFromDense(t, other)
		if it != nil{
			it.mask = retVal.mask
		}
		if ot != nil{
			ot.mask = retVal.mask
		}
		isMasked := retVal.IsMasked()
		switch reuse.t.Kind() {
		{{range .Kinds -}}
			{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			data := reuse.{{sliceOf .}}
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}
							data[incrI] += {{if $isFunc -}}
								{{if eq $op "math.Pow" -}}
									{{if eq .String "complex64" -}}
										complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
									{{else if isFloat . -}}
										{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
									{{else -}}
										{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
									{{end -}}
								{{end -}}
							{{else -}}
								t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
							{{end -}}
					  
						i += iterStep
						j += iterStep
					}
					{{if or $div $scaleInv -}}
						if err != nil {
							return
						}
						err = errs
					{{end -}}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}
						data[incrI] += {{if $isFunc -}}
							{{if eq $op "math.Pow" -}}
								{{if eq .String "complex64" -}}
									complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
								{{else if isFloat . -}}
									{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
								{{else -}}
									{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
								{{end -}}
							{{end -}}
						{{else -}}
							 t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
						{{end -}}
						j += iterStep
					}
					{{if or $div $scaleInv -}}
						if err != nil {
							return
						}
						err = errs
					{{end -}}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}
						data[incrI] += {{if $isFunc -}}
							{{if eq $op "math.Pow" -}}
								{{if eq .String "complex64" -}}
									complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
								{{else if isFloat . -}}
									{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
								{{else -}}
									{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
								{{end -}}
							{{end -}}
						{{else -}}
							 t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
						{{end -}}
						i += iterStep
					}
					{{if or $div $scaleInv -}}
						if err != nil {
							return
						}
						err = errs
					{{end -}}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}

						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}

						data[incrI] += {{if $isFunc -}}
							{{if eq $op "math.Pow" -}}
								{{if eq .String "complex64" -}}
									complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
								{{else if isFloat . -}}
									{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
								{{else -}}
									{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
								{{end -}}
							{{end -}}
						{{else -}}
							 t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
						{{end -}}
					}
					{{if or $div $scaleInv -}}
						if err != nil {
							return
						}
						err = errs
					{{end -}}					
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
				if isMasked{
					err = incrVec{{$opName}}{{short .}}Masked(t.{{sliceOf .}}, other.{{sliceOf .}}, reuse.{{sliceOf .}}, reuse.mask)
				} else{
					err = incrVec{{$opName}}{{short .}}(t.{{sliceOf .}}, other.{{sliceOf .}}, reuse.{{sliceOf .}})
				}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}

						data[incrI] +={{if $isFunc -}}
							{{if eq $op "math.Pow" -}}
								{{if eq .String "complex64" -}}
									complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
								{{else if isFloat . -}}
									{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
								{{else -}}
									{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
								{{end -}}
							{{end -}}
						{{else -}}
							 t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
						{{end -}}
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}

						data[incrI] +={{if $isFunc -}}
							{{if eq $op "math.Pow" -}}
								{{if eq .String "complex64" -}}
									complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
								{{else if isFloat . -}}
									{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
								{{else -}}
									{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
								{{end -}}
							{{end -}}
						{{else -}}
							 t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
						{{end -}}
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil{
							err = handleNoOp(err)
							break
						}
						{{if or $div $scaleInv -}}
							{{if isntFloat . -}}
								if other.{{getOne .}}(j) == 0 {
									errs = append(errs, j)
									continue
								}
							{{end -}}
						{{end -}}
						data[incrI] +={{if $isFunc -}}
							{{if eq $op "math.Pow" -}}
								{{if eq .String "complex64" -}}
									complex64(cmplx.Pow(complex128(t.{{getOne .}}(i)), complex128(other.{{getOne .}}(j))))
								{{else if isFloat . -}}
									{{mathPkg .}}Pow(t.{{getOne .}}(i), other.{{getOne .}}(j))
								{{else -}}
									{{asType .}}({{$op}}(float64(t.{{getOne .}}(i)), float64(other.{{getOne .}}(j))))
								{{end -}}
							{{end -}}
						{{else -}}
							 t.{{getOne .}}(i) {{$op}} other.{{getOne .}}(j)
						{{end -}}

						incrI += iterStep
					}
					{{if or $div $scaleInv -}}
						if err != nil {
							return
						}
						err = errs
					{{end -}}
				}
			}
			{{end -}}
		{{end -}}
		}
		{{if or $scaleInv $div -}}
		if errs != nil {
			err = errs
		}
		{{end -}}
		
	case toReuse:
		if t.IsMaterializable(){
			copyDenseIter(reuse, t, nil, it.FlatIterator)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.{{lower .OpName}}(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable(){
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.{{lower .OpName}}(other, nil, ot)
	case !safe:
		err = t.{{lower .OpName}}(other, it, ot)
		retVal = t
	}
	if it != nil{
		it.mask = nil
		}
	if ot != nil{
		ot.mask = nil
	}
	return
}
`

const denseDenseArithSwitchTableRaw = `func (t *Dense) {{lower .OpName}}(other *Dense, itt, ott Iterator) (err error){
	var it, ot *FlatMaskedIterator
	if itt != nil {
		it = new(FlatMaskedIterator)	
		switch iter:= itt.(type) {
			case *FlatIterator:			
				it.FlatIterator = iter
			case *FlatMaskedIterator:
				it = iter
			case *MultIterator:
				it.FlatIterator = iter.fit0
				it.mask = iter.mask
		}
	}
	if ott != nil {
		ot = new(FlatMaskedIterator)
		switch iter:= ott.(type) {
			case *FlatIterator:
				ot.FlatIterator = iter
			case *FlatMaskedIterator:
				ot = iter
			case *MultIterator:
				ot.FlatIterator = iter.fit0
				ot.mask = iter.mask
		}
	}

	t.MaskFromDense(t, other)

	if it != nil{
		it.mask = t.mask
	}
	if ot != nil{
		ot.mask = t.mask
	}
		
	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$opName := .OpName -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}
	{{if or $scaleInv $div -}}var errs errorIndices
	{{end -}}
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		tdata := t.{{sliceOf .}}
		odata := other.{{sliceOf .}}
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if odata[j] == 0 {
						errs = append(errs, j)
						continue
					}
					{{end -}}
				{{end -}}
				tdata[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(tdata[i], odata[j])
						{{else -}}
							{{asType .}}({{$op}}(float64(tdata[i]), float64(odata[j])))
						{{end -}}
					{{end -}}
				{{else -}}
					tdata[i] {{$op}} odata[j]
				{{end -}}
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid(){
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if odata[j] == 0 {
						errs = append(errs, j)
						continue
					}
					{{end -}}
				{{end -}}
				tdata[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(tdata[i], odata[j])
						{{else -}}
							{{asType .}}({{$op}}(float64(tdata[i]), float64(odata[j])))
						{{end -}}
					{{end -}}
				{{else -}}
					tdata[i] {{$op}} odata[j]
				{{end -}}
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if odata[j] == 0 {
						errs = append(errs, j)
						continue
					}
					{{end -}}
				{{end -}}
				tdata[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(tdata[i], odata[j])
						{{else -}}
							{{asType .}}({{$op}}(float64(tdata[i]), float64(odata[j])))
						{{end -}}
					{{end -}}
				{{else -}}
					tdata[i] {{$op}} odata[j]
				{{end -}}
				i += iterStep
			}
		default:
			if t.IsMasked(){
				vec{{$opName}}{{short .}}Masked(tdata, odata, t.mask)
			} else {
				vec{{$opName}}{{short .}}(tdata, odata)	
			}
		}
		{{end -}}
	{{end -}}
	default:
		// TODO: Handle Number interface
	}

	{{if or $scaleInv $div -}}
		if err != nil {
			return
		}

		if errs != nil {
			err = errs
		}
	{{end -}}
	return
}
`

const denseScalarArithRaw = `// {{.OpName}} performs {{.CommonName}} on a *Dense and a scalar value. The scalar value has to be of the same 
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) {{.OpName}}(other interface{}, opts ...FuncOpt) (retVal *Dense, err error){
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
		}

	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$opName := .OpName}}
	switch {
	case incr:
		switch t.t.Kind(){
		{{range .Kinds -}}
		{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			if t.IsMasked() {				
				err = incr{{$opName}}{{short .}}Masked(t.{{sliceOf .}}, reuse.{{sliceOf .}}, other.({{asType .}}), t.mask)				
			} else {
				err = incr{{$opName}}{{short .}}(t.{{sliceOf .}}, reuse.{{sliceOf .}}, other.({{asType .}}))
			}
			retVal = reuse	
		{{end -}}
		{{end -}}
		}
	case toReuse:
		if t.IsMaterializable(){
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) 
		}
		err = reuse.{{lower .OpName}}(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable(){
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.{{lower .OpName}}(other)
	case !safe:
		err = t.{{lower .OpName}}(other)
		retVal = t
	}
	return
}
`

const denseScalarArithSwitchTableRaw = `func (t *Dense) {{lower .OpName}}(other interface{}) (err error){
	{{$isFunc := .IsFunc -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}
	{{$isReversed := hasSuffix .OpName "R" -}}
	switch t.t.Kind() {
	{{$opName := .OpName -}}
	{{$op := .OpSymb -}}
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		b := other.({{asType .}})
		if t.IsMaterializable() {
			{{if or $scaleInv $div -}}
				if b == 0 {
					err = t.zeroIter()
					if err != nil {
						err = errors.Wrapf(err, div0, -1)
						return
					}
					err = errors.Errorf(div0, -1)
					return
				}
			{{end -}}
			it := IteratorFromDense(t)
			var i int
			data := t.{{sliceOf .}}
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid(){
				data[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(data[i]), complex128(b)))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(data[i], b)
						{{else -}}
							{{asType .}}({{$op}}(float64(data[i]), float64(b)))
						{{end -}}
					{{end -}}
				{{else -}}
					{{if $isReversed -}}
						b {{$op}} data[i]
					{{else -}}
						data[i] {{$op}} b
					{{end -}}
				{{end -}}
			}
			return nil
		}
		return {{lower $opName}}{{short .}}(t.{{sliceOf .}}, b)
		{{end -}}
		
	{{end -}}
	}
	return nil
}
`

var (
	ddArith      *template.Template
	ddArithTable *template.Template
	dsArith      *template.Template
	dsArithTable *template.Template
)

func init() {
	ddArith = template.Must(template.New("dense arith").Funcs(funcs).Parse(denseDenseArithRaw))
	ddArithTable = template.Must(template.New("dense arith table").Funcs(funcs).Parse(denseDenseArithSwitchTableRaw))
	dsArith = template.Must(template.New("dense-scalar arith").Funcs(funcs).Parse(denseScalarArithRaw))
	dsArithTable = template.Must(template.New("dense-scalar arith table").Funcs(funcs).Parse(denseScalarArithSwitchTableRaw))
}
func arith(f io.Writer, generic *ManyKinds) {
	fmt.Fprint(f, prepArithRaw)
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{
			ManyKinds: generic,
			OpName:    bo.OpName,
			OpSymb:    bo.OpSymb,
			IsFunc:    bo.IsFunc,
		}
		ddArith.Execute(f, op)
		ddArithTable.Execute(f, op)
		fmt.Fprint(f, "\n")
	}
	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{
			ManyKinds:  generic,
			OpName:     bo.OpName,
			CommonName: bo.CommonName,
			OpSymb:     bo.OpSymb,
			IsFunc:     bo.IsFunc,
		}
		dsArith.Execute(f, op)
		dsArithTable.Execute(f, op)
		fmt.Fprint(f, "\n")
	}
}
