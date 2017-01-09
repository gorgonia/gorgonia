package main

import (
	"fmt"
	"html/template"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
)

var funcMap = template.FuncMap{
	"title": strings.Title,
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
	TestData0 string
	TestData1 string
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
		testtestName = "output/test_test.go"

		basicsName      = "output/array_impl.go"
		numbersName     = "output/array_number.go"
		numbersTestName = "output/array_number_test.go"
		eleqordName     = "output/array_eleqord.go"
		eleqordTestName = "output/array_eleqord_test.go"
	)

	if err := os.Remove(basicsName); err != nil {
		if !os.IsNotExist(err) {
			panic(err)
		}
	}

	pipeline(testtestName, m, testtestFn)

	pipeline(basicsName, m, basicsFn)
	pipeline(numbersName, m, numbersFn)
	pipeline(numbersTestName, m, numbersTestFn)
	pipeline(eleqordName, m, eleqordsFn)
	pipeline(eleqordTestName, m, eleqordsTestFn)
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

	fn(f, l)

	// gofmt and goimports this shit
	cmd := exec.Command("goimports", "-w", fileName)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fileName)
	}

}

func basicsFn(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "package tensor\n")
	for _, tmpl := range basics {
		fmt.Fprintf(f, "/* %s */\n\n", tmpl.Name())
		for _, v := range m {
			tmpl.Execute(f, v)
			fmt.Fprintf(f, "\n")
		}
		fmt.Fprint(f, "\n")
	}

	fmt.Fprintln(f, "/* Transpose Specialization */\n")
	for _, v := range m {
		transposeTmpl.Execute(f, v)
		fmt.Fprintf(f, "\n")
	}
}

func testtestFn(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "package tensor\n")

	for _, v := range m {
		fmt.Fprintf(f, "type %vDummy []%v\n", v.Name, v.Of)
	}

	for _, tmpl := range basics {
		fmt.Fprintf(f, "/* %s */\n\n", tmpl.Name())
		for _, v := range m {
			v.Name += "Dummy"
			tmpl.Execute(f, v)
			fmt.Fprintf(f, "\n")
		}
		fmt.Fprint(f, "\n")
	}

	fmt.Fprintf(f, "/* COMPAT */ \n\n")
	for _, v := range m {
		v.Name += "Dummy"
		compatibleTmpl.Execute(f, v)
		fmt.Fprintf(f, "\n")
	}

	// generate numbers
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb}
				op.Name += "Dummy"
				binOpTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}

	// generate eleqords
	for _, bo := range eleqordBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if bo.OpName == "ElEq" || (bo.OpName != "ElEq" && v.elOrd) {
				op := ElOrdBinOp{v, bo.OpName, bo.OpSymb, bo.TypeClass}
				op.Name += "Dummy"
				eleqordTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}

	// generate misc
	for _, v := range m {
		v.Name += "Dummy"
		sliceTmpl.Execute(f, v)
		fmt.Fprint(f, "\n")
		dtypeTmpl.Execute(f, v)
	}
}

func numbersFn(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "package tensor\n")

	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb}
				binOpTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}

	log.Println("NOTE: Manually fix Div for non-float types")
}

func numbersTestFn(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "package tensor\n")

	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb}
				binOpTestTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}
}

func eleqordsFn(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "package tensor\n")

	for _, bo := range eleqordBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if bo.OpName == "ElEq" || (bo.OpName != "ElEq" && v.elOrd) {
				op := ElOrdBinOp{v, bo.OpName, bo.OpSymb, bo.TypeClass}
				eleqordTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}
}

func eleqordsTestFn(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "package tensor\n")

	for _, bo := range eleqordBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if bo.OpName == "ElEq" || (bo.OpName != "ElEq" && v.elOrd) {
				op := ElOrdBinOp{v, bo.OpName, bo.OpSymb, bo.TypeClass}
				eleqordTestTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}
}
