package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"text/template"
)

var funcMap = template.FuncMap{
	"title":     strings.Title,
	"hasSuffix": strings.HasSuffix,
	"hasPrefix": strings.HasPrefix,
}

type ArrayType struct {
	Of          string
	Name        string
	DefaultZero string
	DefaultOne  string
	Compatible  string

	VecPkg  string // for floats
	MathPkg string

	isNumber bool
	elOrd    bool

	// test
	TestData0    string
	TestData1    string
	IncrTestData string
}

func main() {
	m := []ArrayType{
		ArrayType{
			Of:          "float64",
			Name:        "f64s",
			DefaultZero: "float64(0)",
			DefaultOne:  "float64(1)",
			Compatible:  "Float64s",

			VecPkg:  "vecf64",
			MathPkg: "math",

			isNumber: true,
			elOrd:    true,

			TestData0: "0, 1, 2, 3, 4",
			TestData1: "1, 2, 2, 1, 100",
		},

		ArrayType{
			Of:          "float32",
			Name:        "f32s",
			DefaultZero: "float32(0)",
			DefaultOne:  "float32(1)",
			Compatible:  "Float32s",

			VecPkg:  "vecf32",
			MathPkg: "math32",

			isNumber: true,
			elOrd:    true,

			TestData0: "0, 1, 2, 3, 4",
			TestData1: "1, 2, 2, 1, 100",
		},

		ArrayType{
			Of:          "int",
			Name:        "ints",
			DefaultZero: "int(0)",
			DefaultOne:  "int(1)",
			Compatible:  "Ints",

			isNumber: true,
			elOrd:    true,

			TestData0: "0, 1, 2, 3, 4",
			TestData1: "1, 2, 2, 1, 100",
		},

		ArrayType{
			Of:          "int64",
			Name:        "i64s",
			DefaultZero: "int64(0)",
			DefaultOne:  "int64(1)",
			Compatible:  "Int64s",

			isNumber: true,
			elOrd:    true,

			TestData0: "0, 1, 2, 3, 4",
			TestData1: "1, 2, 2, 1, 100",
		},

		ArrayType{
			Of:          "int32",
			Name:        "i32s",
			DefaultZero: "int32(0)",
			DefaultOne:  "int32(1)",
			Compatible:  "Int32s",

			isNumber: true,
			elOrd:    true,

			TestData0: "0, 1, 2, 3, 4",
			TestData1: "1, 2, 2, 1, 100",
		},

		ArrayType{
			Of:          "byte",
			Name:        "u8s",
			DefaultZero: "byte(0)",
			DefaultOne:  "byte(1)",
			Compatible:  "Bytes",

			isNumber: true,
			elOrd:    true,

			TestData0: "0, 1, 2, 3, 4",
			TestData1: "1, 2, 2, 1, 100",
		},

		ArrayType{
			Of:          "bool",
			Name:        "bs",
			DefaultZero: "false",
			DefaultOne:  "true",
			Compatible:  "Bools",

			TestData0: "true, false, true, false, true",
			TestData1: "true, true, true, false, false",
		},
	}

	const (
		testtestName        = "../test_test.go"
		basicsName          = "../array_impl.go"
		numbersName         = "../array_number.go"
		numbersTestName     = "../array_number_test.go"
		incrNumbersName     = "../array_incr.go"
		incrNumbersTestName = "../array_incr_test.go"
		eleqordName         = "../array_eleqord.go"
		eleqordTestName     = "../array_eleqord_test.go"

		denseBinOpName     = "../dense_arith.go"
		denseBinOpTestName = "../api_arith_test.go"
	)

	if err := os.Remove(basicsName); err != nil {
		if !os.IsNotExist(err) {
			panic(err)
		}
	}

	pipeline(testtestName, m, testtestFn)

	pipeline(basicsName, m, generateBasics)
	pipeline(numbersName, m, generateNumbers)
	pipeline(numbersTestName, m, generateNumbersTests)
	pipeline(incrNumbersName, m, generateNumbersIncr)
	pipeline(incrNumbersTestName, m, generateNumbersIncrTests)
	pipeline(eleqordName, m, generateElEqOrds)
	pipeline(eleqordTestName, m, generateElEqOrdsTests)

	generateDenseArith(denseBinOpName)
	generateDenseArithTests(denseBinOpTestName, m)
}

func pipeline(fileName string, l []ArrayType, fn func(io.Writer, []ArrayType)) {
	if err := os.Remove(fileName); err != nil {
		if !os.IsNotExist(err) {
			panic(err)
		}
	}

	f, err := os.Create(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	fmt.Fprintf(f, "package tensor\n")
	fn(f, l)

	// gofmt and goimports this shit
	cmd := exec.Command("goimports", "-w", fileName)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fileName)
	}
}

func testtestFn(f io.Writer, m []ArrayType) {
	m2 := make([]ArrayType, len(m))
	for i, t := range m {
		m2[i] = t
		m2[i].Name += "Dummy"
	}

	// declare types
	for _, v := range m2 {
		fmt.Fprintf(f, "type %v []%v\n", v.Name, v.Of)
	}

	// generate basics
	generateBasics(f, m2)

	// generate compat
	fmt.Fprintf(f, "/* COMPAT */ \n\n")
	for _, v := range m2 {
		compatibleTmpl.Execute(f, v)
		fmt.Fprintf(f, "\n")
	}

	generateNumbersOpsOnly(f, m2)
	generateNumbersIncr(f, m2)
	generateElEqOrds(f, m2)

	// generate misc
	for _, v := range m2 {
		sliceTmpl.Execute(f, v)
		fmt.Fprint(f, "\n")

		dtypeTmpl.Execute(f, v)
		fmt.Fprint(f, "\n")

		if strings.HasPrefix(v.Of, "float") {
			floatArrTmpl.Execute(f, v)
		}
	}
}
