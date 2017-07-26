package main

import (
	"fmt"
	"io"
)

type DenseBinOp struct {
	MethodName string
	Name   string
	Scalar bool
}

func (fn *DenseBinOp) Write(w io.Writer) {
	if fn.Scalar {
		fmt.Fprintf(w, "func (t *Dense) %sScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {\n", fn.MethodName)
		denseArithScalarBody.Execute(w, fn)
	} else {
		fmt.Fprintf(w, "func (t *Dense) %s(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {\n", fn.MethodName)
		denseArithBody.Execute(w, fn)
	}
	w.Write([]byte("}\n\n"))

}

func generateDenseArith(f io.Writer, ak Kinds) {
	var methods []*DenseBinOp
	for _, bo := range arithBinOps {
		meth := &DenseBinOp{
			MethodName: bo.Name(),
			Name: bo.Name(),
		}
		methods = append(methods, meth)
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.Scalar = true
	}
	for _, meth := range methods {
		meth.Write(f)
	}
}


func generateDenseCmp(f io.Writer, ak Kinds) {
	var methods []*DenseBinOp
	for _, cbo := range cmpBinOps {
		methName := cbo.Name()
		if methName == "Eq" || methName == "Ne" {
			methName = "El"+cbo.Name()
		}
		meth := &DenseBinOp {
			MethodName: methName,
			Name: cbo.Name(), 
		}
		methods = append(methods, meth)
	}
	for _, meth := range methods {
		meth.Write(f)
		meth.Scalar = true
	}
	for _, meth := range methods {
		meth.Write(f)
	}
}