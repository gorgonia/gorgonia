package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"strings"
)

type ManyKinds struct {
	Kinds []reflect.Kind
}

var (
	gopath       string
	tensorPkgLoc string
	stdEngPkgLoc string
)

const (
	genmsg = "GENERATED FILE. DO NOT EDIT"
)

const (
	getSetName                    = "dense_getset.go"
	getSetTestsName               = "dense_getset_test.go"
	genUtilsName                  = "genericUtils.go"
	transposeName                 = "dense_transpose_specializations.go"
	viewStackName                 = "dense_viewstack_specializations.go"
	arrayGetSetName               = "arrayGetSet.go"
	genericArithName              = "genericArith.go"
	genericCmpName                = "genericCmp.go"
	denseArithName                = "dense_arith.go"
	denseArithTestsName           = "dense_arith_test.go"
	denseCmpName                  = "dense_cmp.go"
	denseCmpTestsName             = "dense_cmp_test.go"
	denseCompatName               = "dense_compat.go"
	denseCompatTestsName          = "dense_compat_test.go"
	genericArgMethodsName         = "genericArgmethods.go"
	denseArgMethodsName           = "dense_argmethods.go"
	denseArgMethodsTestsName      = "dense_argmethods_test.go"
	denseMaskCmpMethodsName       = "dense_maskcmp_methods.go"
	denseMaskCmpMethodsTestsName  = "dense_maskcmp_methods_test.go"
	genericReductionName          = "genericReduction.go"
	denseReductionName            = "dense_reduction.go"
	denseReductionTestsName       = "dense_reduction_test.go"
	denseReductionMethodsName     = "dense_reduction_methods.go"
	denseReductionMethodsTestName = "dense_reduction_methods_test.go"
	denseMapperName               = "dense_mapper.go"
	denseApplyTestsName           = "dense_apply_test.go"
	apiUnaryGenName               = "api_unary_generated.go"
	apiUnaryGenTestsName          = "api_unary_generated_test.go"
	denseGenName                  = "dense_generated.go"
	denseGenTestsName             = "dense_generated_test.go"
	denseIOName                   = "dense_io.go"
	denseIOTestsName              = "dense_io_test.go"

	testTestName = "test_test.go"

	stdengAdd  = "add.go"
	stdengTest = "test_test.go"
)

func init() {
	gopath = os.Getenv("GOPATH")
	tensorPkgLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor")
	stdEngPkgLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor/internal/stdeng")
}

func main() {
	pregenerate()
	mk := makeManyKinds()
	pipeline(tensorPkgLoc, getSetName, mk, getset)
	pipeline(tensorPkgLoc, getSetTestsName, mk, getsetTest)
	pipeline(tensorPkgLoc, genUtilsName, mk, utils)
	pipeline(tensorPkgLoc, transposeName, mk, transpose)
	pipeline(tensorPkgLoc, viewStackName, mk, viewstack)
	pipeline(tensorPkgLoc, arrayGetSetName, mk, arrayGetSet)
	pipeline(tensorPkgLoc, genericArithName, mk, genericArith)
	pipeline(tensorPkgLoc, genericCmpName, mk, genericCmp)
	pipeline(tensorPkgLoc, denseArithName, mk, arith)
	pipeline(tensorPkgLoc, denseArithTestsName, mk, denseArithTests)
	pipeline(tensorPkgLoc, denseCmpName, mk, denseCmp)
	pipeline(tensorPkgLoc, denseCmpTestsName, mk, denseCmpTests)
	pipeline(tensorPkgLoc, denseCompatName, mk, compat)
	pipeline(tensorPkgLoc, denseCompatTestsName, mk, testCompat)
	pipeline(tensorPkgLoc, testTestName, mk, testtest)
	pipeline(tensorPkgLoc, genericArgMethodsName, mk, genericArgmethods)
	pipeline(tensorPkgLoc, denseArgMethodsName, mk, argmethods)
	pipeline(tensorPkgLoc, denseArgMethodsTestsName, mk, argmethodsTests)
	pipeline(tensorPkgLoc, denseMaskCmpMethodsName, mk, maskcmpmethods)
	pipeline(tensorPkgLoc, denseMaskCmpMethodsTestsName, mk, maskcmpmethodsTests)
	pipeline(tensorPkgLoc, genericReductionName, mk, genericReduction)
	pipeline(tensorPkgLoc, denseReductionName, mk, reduction)
	pipeline(tensorPkgLoc, denseReductionTestsName, mk, denseReductionTests)
	pipeline(tensorPkgLoc, denseReductionMethodsName, mk, generateDenseReductionMethods)
	pipeline(tensorPkgLoc, denseReductionMethodsTestName, mk, generateDenseReductionMethodsTests)
	pipeline(tensorPkgLoc, denseMapperName, mk, generateDenseMapper)
	pipeline(tensorPkgLoc, denseApplyTestsName, mk, generateDenseApplyTests)
	pipeline(tensorPkgLoc, apiUnaryGenName, mk, generateUnaryAPIFuncs)
	pipeline(tensorPkgLoc, apiUnaryGenTestsName, mk, generateUnaryTests)
	pipeline(tensorPkgLoc, denseGenName, mk, generateDenseConstructionFns)
	pipeline(tensorPkgLoc, denseGenTestsName, mk, generateDenseTests)
	pipeline(tensorPkgLoc, denseIOName, mk, generateDenseIO)
	pipeline(tensorPkgLoc, denseIOTestsName, mk, generateDenseIOTests)
	// pipeline(blah, mk, maskcmpmethods)

	pipeline(stdEngPkgLoc, stdengAdd, mk, generateStdEngAdd)
	pipeline(stdEngPkgLoc, stdengTest, mk, arrayHeaderGetSet)
}

func writePkgName(f io.Writer, pkg string) {
	switch pkg {
	case tensorPkgLoc:
		fmt.Fprintf(f, "package tensor\n/*\n%v\n*/\n\n", genmsg)
	case stdEngPkgLoc:
		fmt.Fprintf(f, "package stdeng\n/*\n%v\n*/\n\n", genmsg)
	default:
		panic("UNKNOWN")
	}
}

func makeManyKinds() *ManyKinds {
	mk := make([]reflect.Kind, 0)
	for k := reflect.Invalid + 1; k < reflect.UnsafePointer+1; k++ {
		mk = append(mk, k)
	}
	return &ManyKinds{mk}
}

func pipeline(pkg, filename string, generic *ManyKinds, fn func(io.Writer, *ManyKinds)) {
	fullpath := path.Join(pkg, filename)
	f, err := os.Create(fullpath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	writePkgName(f, pkg)
	fn(f, generic)

	// gofmt and goimports this stuff
	cmd := exec.Command("goimports", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fullpath)
	}

	// cmd = exec.Command("sed", "-i", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	// if err = cmd.Run(); err != nil {
	// 	log.Fatalf("sed failed with %v for %q", err, fullpath)
	// }

	cmd = exec.Command("gofmt", "-s", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Gofmt failed for %q", fullpath)
	}
}

// pregenerate cleans up all files that were previously generated.
func pregenerate() error {
	pattern1 := path.Join(tensorPkgLoc, "*.go")
	matches, err := filepath.Glob(pattern1)
	if err != nil {
		return err
	}
	for _, m := range matches {
		b, err := ioutil.ReadFile(m)
		if err != nil {
			return err
		}
		s := string(b)
		if strings.Contains(s, genmsg) {
			if err := os.Remove(m); err != nil {
				return err
			}
		}
	}

	pattern2 := path.Join(stdEngPkgLoc, "*.go")
	matches, err = filepath.Glob(pattern2)
	if err != nil {
		return err
	}
	for _, m := range matches {
		b, err := ioutil.ReadFile(m)
		if err != nil {
			return err
		}
		s := string(b)
		if strings.Contains(s, genmsg) {
			if err := os.Remove(m); err != nil {
				return err
			}
		}
	}
	return nil
}
