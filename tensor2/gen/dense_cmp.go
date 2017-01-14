package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"text/template"
)

type DenseCmpBinOp struct {
	OpName string
}

var denseCmpBinOps = []DenseCmpBinOp{
	{"elEq"}, {"gt"}, {"gte"}, {"lt"}, {"lte"},
}

const prepDenseCmpBinOpRaw = `package tensor

func prepDenseCmpBinOp(a, b *Dense, opts ...FuncOpt) (ao, bo ElOrd, reuse *Dense, safe, same, toReuse bool, err error){
	var ok bool
	if ao, ok = a.data.(ElOrd); !ok {
		err = noopError{}
		return
	}

	if bo, ok = b.data.(ElOrd); !ok {
		err = noopError{}
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		// coarse type checking. Actual type switching will happen in the op
		if !same{
			var b Boolser
			if b, ok = reuse.data.(Boolser); !ok{
				err = errors.Errorf(typeMismatch, b, reuse.data)
				return
			}
		} else{
			var kal ElOrd 
			if kal, ok = reuse.data.(ElOrd); !ok {
				err = errors.Errorf(typeMismatch, kal, reuse.data)
				return
			}
		}

		if  err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}

func prepOneDenseCmp(a *Dense, opts ...FuncOpt)(ao ElOrd, reuse *Dense, safe, same, toReuse bool, err error){
	var ok bool
	if ao, ok = a.data.(ElOrd); !ok {
		err = noopError{}
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		// coarse type checking. Actual type switching will happen in the op
		if !same{
			var b Boolser
			if b, ok = reuse.data.(Boolser); !ok{
				err = errors.Errorf(typeMismatch, b, reuse.data)
				return
			}
		} else{
			var kal ElOrd 
			if kal, ok = reuse.data.(ElOrd); !ok {
				err = errors.Errorf(typeMismatch, kal, reuse.data)
				return
			}
		}

		if  err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}
`
const denseCmpBinOpRaw = `func {{.OpName}}DD(a, b *Dense, opts ...FuncOpt)(retVal *Dense, err error){
	ao, bo, reuse, safe, same, toReuse, err := prepDenseCmpBinOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	var arr Array
	if arr, err = ao.{{title .OpName}}(bo, safe); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func {{.OpName}}DS(a *Dense, b interface{}, opts ...FuncOpt)(retVal *Dense, err error){
	ao, reuse, safe, same, toReuse, err := prepOneDenseCmp(a, opts...)
	if err != nil {
		return nil, err
	}
	
	tmp := cloneArray(ao).(ElOrd)
	if err = tmp.Memset(b); err != nil {
		return 
	}

	var arr Array
	if arr, err = ao.{{title .OpName}}(tmp, same);err != nil{
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func {{.OpName}}SD(a interface{}, b *Dense, opts ...FuncOpt)(retVal *Dense, err error){
	bo, reuse, safe, same, toReuse, err := prepOneDenseCmp(b, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(bo).(ElOrd)
	if err = tmp.Memset(a); err != nil {
		return 
	}

	var arr Array
	if arr, err = tmp.{{title .OpName}}(bo, same); err !=nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe:
		d := recycledDense(b.t, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(b.data, arr)
		retVal = b
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

`

var (
	denseCmpBinOpTmpl *template.Template
)

func init() {
	denseCmpBinOpTmpl = template.Must(template.New("DenseCmpBinOp").Funcs(funcMap).Parse(denseCmpBinOpRaw))
}

func generateDenseCmp(fileName string) {
	if err := os.Remove(fileName); err != nil {
		if !os.IsNotExist(err) {
			panic(err)
		}
	}

	f, err := os.Create(fileName)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Fprintln(f, prepDenseCmpBinOpRaw)
	fmt.Fprintln(f, "\n")
	for _, bo := range denseCmpBinOps {
		fmt.Fprintf(f, "/* %v */\n\n", bo.OpName)
		denseCmpBinOpTmpl.Execute(f, bo)
		fmt.Fprintln(f, "\n")
	}
	f.Close()

	// gofmt and goimports this shit
	cmd := exec.Command("goimports", "-w", fileName)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fileName)
	}
}
