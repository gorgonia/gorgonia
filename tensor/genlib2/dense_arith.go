package main

import (
	"fmt"
	"io"
)

type DenseArith struct {
	Name   string
	Scalar bool
}

func (fn *DenseArith) Write(w io.Writer) {
	if fn.Scalar {
		fmt.Fprintf(w, "func (t *Dense) %sScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {\n", fn.Name)
		denseArithScalarBody.Execute(w, fn)
	} else {
		fmt.Fprintf(w, "func (t *Dense) %s(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {\n", fn.Name)
		denseArithBody.Execute(w, fn)
	}
	w.Write([]byte("}\n\n"))

}

func generateDenseArith(f io.Writer, ak Kinds) {
	var methods []*DenseArith
	for _, bo := range arithBinOps {
		meth := &DenseArith{
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
