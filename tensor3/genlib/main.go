package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"reflect"
)

type ManyKinds struct {
	Kinds []reflect.Kind
}

func main() {
	const (
		getSetName          = "../dense_getset.go"
		getSetTestsName     = "../dense_getset_test.go"
		genUtilsName        = "../genericUtils.go"
		transposeName       = "../dense_transpose_specializations.go"
		viewStackName       = "../dense_viewstack_specializations.go"
		genericArithName    = "../genericArith.go"
		genericCmpName      = "../genericCmp.go"
		denseArithName      = "../dense_arith.go"
		denseArithTestsName = "../dense_arith_test.go"
		denseCmpName        = "../dense_cmp.go"

		testTestName = "../test_test.go"
	)
	mk := makeManyKinds()
	pipeline(getSetName, mk, getset)
	pipeline(getSetTestsName, mk, getsetTest)
	pipeline(genUtilsName, mk, utils)
	pipeline(transposeName, mk, transpose)
	pipeline(viewStackName, mk, viewstack)
	pipeline(genericArithName, mk, genericArith)
	pipeline(genericCmpName, mk, genericCmp)
	pipeline(denseArithName, mk, arith)
	pipeline(denseArithTestsName, mk, denseArithTests)
	pipeline(denseCmpName, mk, denseCmp)
	pipeline(testTestName, mk, testtest)
}

func makeManyKinds() *ManyKinds {
	mk := make([]reflect.Kind, 0)
	for k := reflect.Invalid + 1; k < reflect.UnsafePointer+1; k++ {
		mk = append(mk, k)
	}
	return &ManyKinds{mk}
}

func pipeline(filename string, generic *ManyKinds, fn func(io.Writer, *ManyKinds)) {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	fmt.Fprintf(f, "package tensor\n/*\nGENERATED FILE. DO NOT EDIT\n*/\n\n")
	fn(f, generic)

	// gofmt and goimports this shit
	cmd := exec.Command("goimports", "-w", filename)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, filename)
	}
}
