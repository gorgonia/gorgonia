package main

import (
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"reflect"
)

const genmsg = "GENERATED FILE. DO NOT EDIT"

var (
	gopath, tensorPkgLoc, execLoc, storageLoc string
)

type Kinds struct {
	Kinds []reflect.Kind
}

func init() {
	gopath = os.Getenv("GOPATH")
	tensorPkgLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor")
	execLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor/internal/execution")
	storageLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor/internal/storage")
}

func main() {
	// pipeline("test", "BLAH_1.go", Kinds{allKinds}, generateGenericVecVecArith)
	// pipeline("test", "BLAH_2.go", Kinds{allKinds}, generateGenericMixedArith)
	// pipeline("test", "BLAH_3.go", Kinds{allKinds}, generateEArith)
	// pipeline("test", "BLAH_4.go", Kinds{allKinds}, generateGenericMap)
	// pipeline("test", "BLAH_5.go", Kinds{allKinds}, generateMap)
	// pipeline("test", "BLAH_6.go", Kinds{allKinds}, generateGenericVecVecCmp)
	// pipeline("test", "BLAH_7.go", Kinds{allKinds}, generateGenericMixedCmp)
	// pipeline("test", "BLAH_8.go", Kinds{allKinds}, generateMinMax)
	// pipeline("test", "BLAH_9.go", Kinds{allKinds}, generateStdEngArith)
	// pipeline("test", "BLAH_10.go", Kinds{allKinds}, generateDenseArith)
	// pipeline("test", "BLAH_11.go", Kinds{allKinds}, generateGenericUncondUnary)
	// pipeline("test", "BLAH_12.go", Kinds{allKinds}, generateGenericArgMethods)
	// pipeline("test", "BLAH_14.go", Kinds{allKinds}, generateStdEngCmp)

	// storage
	pipeline(storageLoc, "getset.go", Kinds{allKinds}, generateHeaderGetSet)
	pipeline(tensorPkgLoc, "array_getset.go", Kinds{allKinds}, generateArrayMethods)

	// execution
	pipeline(execLoc, "generic_arith_vv.go", Kinds{allKinds}, generateGenericVecVecArith)
	pipeline(execLoc, "generic_arith_mixed.go", Kinds{allKinds}, generateGenericMixedArith)
	// pipeline(execLoc, "generic_arith.go", Kinds{allKinds}, generateGenericScalarScalarArith) // generate once and manually edit later
	pipeline(execLoc, "generic_cmp_vv.go", Kinds{allKinds}, generateGenericVecVecCmp)
	pipeline(execLoc, "generic_cmp_mixed.go", Kinds{allKinds}, generateGenericMixedCmp)
	pipeline(execLoc, "generic_minmax.go", Kinds{allKinds}, generateMinMax)
	pipeline(execLoc, "generic_map.go", Kinds{allKinds}, generateGenericMap)
	pipeline(execLoc, "generic_unary.go", Kinds{allKinds}, generateGenericUncondUnary, generateGenericCondUnary, generateSpecialGenericUnaries)
	pipeline(execLoc, "generic_reduce.go", Kinds{allKinds}, generateGenericReduce)
	pipeline(execLoc, "generic_argmethods.go", Kinds{allKinds}, generateGenericArgMethods)
	pipeline(tensorPkgLoc, "generic_utils.go", Kinds{allKinds}, generateUtils)

	// level 1 aggregation
	pipeline(execLoc, "eng_arith.go", Kinds{allKinds}, generateEArith)
	pipeline(execLoc, "eng_map.go", Kinds{allKinds}, generateEMap)
	pipeline(execLoc, "eng_cmp.go", Kinds{allKinds}, generateECmp)
	pipeline(execLoc, "eng_reduce.go", Kinds{allKinds}, generateEReduce)
	pipeline(execLoc, "eng_unary.go", Kinds{allKinds}, generateUncondEUnary, generateCondEUnary, generateSpecialEUnaries)
	pipeline(execLoc, "reduction_specialization.go", Kinds{allKinds}, generateReductionSpecialization)
	pipeline(execLoc, "eng_argmethods.go", Kinds{allKinds}, generateInternalEngArgmethods)

	// level 2 aggregation
	pipeline(tensorPkgLoc, "defaultengine_arith.go", Kinds{allKinds}, generateStdEngArith)
	pipeline(tensorPkgLoc, "defaultengine_cmp.go", Kinds{allKinds}, generateStdEngCmp)
	pipeline(tensorPkgLoc, "defaultengine_unary.go", Kinds{allKinds}, generateStdEngUncondUnary, generateStdEngCondUnary)

	// level 3 aggregation
	pipeline(tensorPkgLoc, "dense_arith.go", Kinds{allKinds}, generateDenseArith)
	pipeline(tensorPkgLoc, "dense_cmp.go", Kinds{allKinds}, generateDenseCmp) // generate once, manually edit later

	// level 4 aggregation
	pipeline(tensorPkgLoc, "api_unary.go", Kinds{allKinds}, generateUncondUnaryAPI, generateCondUnaryAPI, generateSpecialUnaryAPI)

	// dense methods
	pipeline(tensorPkgLoc, "dense_generated.go", Kinds{allKinds}, generateDenseConstructionFns)
	pipeline(tensorPkgLoc, "dense_io.go", Kinds{allKinds}, generateDenseIO)
	pipeline(tensorPkgLoc, "dense_compat.go", Kinds{allKinds}, generateDenseCompat)

	// tests
	pipeline(tensorPkgLoc, "test_test.go", Kinds{allKinds}, generateTestUtils)
	pipeline(tensorPkgLoc, "dense_reduction_test.go", Kinds{allKinds}, generateDenseReductionTests)
	pipeline(tensorPkgLoc, "dense_apply_test.go", Kinds{allKinds}, generateDenseApplyTests)
	pipeline(tensorPkgLoc, "dense_argmethods_test.go", Kinds{allKinds}, generateArgmethodsTests)
	pipeline(tensorPkgLoc, "dense_getset_test.go", Kinds{allKinds}, generateDenseGetSetTests)

	pipeline(tensorPkgLoc, "api_arith_generated_test.go", Kinds{allKinds}, generateAPIArithTests)

}

func pipeline(pkg, filename string, kinds Kinds, fns ...func(io.Writer, Kinds)) {
	fullpath := path.Join(pkg, filename)
	f, err := os.Create(fullpath)
	if err != nil {
		log.Printf("fullpath %q", fullpath)
		log.Fatal(err)
	}
	defer f.Close()
	writePkgName(f, pkg)

	for _, fn := range fns {
		fn(f, kinds)
	}

	// gofmt and goimports this stuff
	cmd := exec.Command("goimports", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fullpath)
	}

	cmd = exec.Command("sed", "-i", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	if err = cmd.Run(); err != nil {
		if err.Error() != "exit status 4" { // exit status 4 == not found
			log.Fatalf("sed failed with %v for %q", err.Error(), fullpath)
		}
	}

	cmd = exec.Command("gofmt", "-s", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Gofmt failed for %q", fullpath)
	}
}
