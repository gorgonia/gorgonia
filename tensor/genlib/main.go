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
		getSetName                    = "../dense_getset.go"
		getSetTestsName               = "../dense_getset_test.go"
		genUtilsName                  = "../genericUtils.go"
		transposeName                 = "../dense_transpose_specializations.go"
		viewStackName                 = "../dense_viewstack_specializations.go"
		genericArithName              = "../genericArith.go"
		genericCmpName                = "../genericCmp.go"
		denseArithName                = "../dense_arith.go"
		denseArithTestsName           = "../dense_arith_test.go"
		denseCmpName                  = "../dense_cmp.go"
		denseCmpTestsName             = "../dense_cmp_test.go"
		denseCompatName               = "../dense_compat.go"
		denseCompatTestsName          = "../dense_compat_test.go"
		genericArgMethodsName         = "../genericArgmethods.go"
		denseArgMethodsName           = "../dense_argmethods.go"
		denseArgMethodsTestsName      = "../dense_argmethods_test.go"
		genericReductionName          = "../genericReduction.go"
		denseReductionName            = "../dense_reduction.go"
		denseReductionTestsName       = "../dense_reduction_test.go"
		denseReductionMethodsName     = "../dense_reduction_methods.go"
		denseReductionMethodsTestName = "../dense_reduction_methods_test.go"
		denseMapperName               = "../dense_mapper.go"
		denseApplyTestsName           = "../dense_apply_test.go"
		apiUnaryGenName               = "../api_unary_generated.go"
		apiUnaryGenTestsName          = "../api_unary_generated_test.go"
		denseGenName                  = "../dense_generated.go"
		denseGenTestsName             = "../dense_generated_test.go"

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
	pipeline(denseCmpTestsName, mk, denseCmpTests)
	pipeline(denseCompatName, mk, compat)
	pipeline(denseCompatTestsName, mk, testCompat)
	pipeline(testTestName, mk, testtest)
	pipeline(genericArgMethodsName, mk, genericArgmethods)
	pipeline(denseArgMethodsName, mk, argmethods)
	pipeline(denseArgMethodsTestsName, mk, argmethodsTests)
	pipeline(genericReductionName, mk, genericReduction)
	pipeline(denseReductionName, mk, reduction)
	pipeline(denseReductionTestsName, mk, denseReductionTests)
	pipeline(denseReductionMethodsName, mk, generateDenseReductionMethods)
	pipeline(denseReductionMethodsTestName, mk, generateDenseReductionMethodsTests)
	pipeline(denseMapperName, mk, generateDenseMapper)
	pipeline(denseApplyTestsName, mk, generateDenseApplyTests)
	pipeline(apiUnaryGenName, mk, generateUnaryAPIFuncs)
	pipeline(apiUnaryGenTestsName, mk, generateUnaryTests)
	pipeline(denseGenName, mk, generateDenseConstructionFns)
	pipeline(denseGenTestsName, mk, generateDenseTests)
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
