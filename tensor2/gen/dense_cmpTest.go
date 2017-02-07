package main

import (
	"fmt"
	"io"
	"text/template"
)

type DenseCmpBinOpTest struct {
	ArrayType

	ScalarValue            int
	TestSlice0, TestSlice1 []int

	CorrectBool0, CorrectBool1, CorrectBool2 []bool
	CorrectSame0, CorrectSame1, CorrectSame2 []int
}

type DenseCmpBinOpTestData struct {
	DenseCmpBinOp
	Types []DenseCmpBinOpTest
}

const denseBinOpCmpTestRaw = `var {{.OpName}}Tests = []struct{
	a interface{}
	b interface{}
	reuse *Dense
	reuseSame *Dense

	correct []bool
	correctSame interface{}
}{
	{{range .Types}}
	// {{title .Of}}
	{a: {{.Of}}({{.ScalarValue}}),
	 b: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice0 -}}{{printf "%d" .}},{{end -}} })),
	 reuse: New(Of({{title .Of}}), WithBacking(make([]bool, {{len .TestSlice0}} ))),
	 reuseSame: New(Of({{title .Of}}), WithBacking(make([]{{.Of}}, {{len .TestSlice0}} ))),
	 correct: []bool{ {{range .CorrectBool0}} {{printf "%t" .}}, {{end -}} },
	 correctSame: []{{.Of}}{ {{range .CorrectSame0}} {{printf "%d" .}}, {{end -}} },
	},
	{a: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice0 -}}{{printf "%d" .}},{{end -}} })),
	 b: {{.Of}}({{.ScalarValue}}),
	 reuse: New(Of({{title .Of}}), WithBacking(make([]bool, 7))),
	 reuseSame: New(Of({{title .Of}}), WithBacking(make([]{{.Of}}, 7))),
	 correct: []bool{ {{range .CorrectBool1}} {{printf "%t" .}}, {{end -}} },
	 correctSame: []{{.Of}}{ {{range .CorrectSame1}} {{printf "%d" .}}, {{end -}} },
	},
	{a: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice0 -}}{{printf "%d" .}},{{end -}} })),
	 b: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice1 -}}{{printf "%d" .}},{{end -}} })),
	 reuse: New(Of({{title .Of}}), WithBacking(make([]bool, 7))),
	 reuseSame: New(Of({{title .Of}}), WithBacking(make([]{{.Of}}, 7))),
	 correct: []bool{ {{range .CorrectBool2}} {{printf "%t" .}}, {{end -}} },
	 correctSame: []{{.Of}}{ {{range .CorrectSame2}} {{printf "%d" .}}, {{end -}} },
	},


	{{end -}}
}

func Test{{title .OpName}}(t *testing.T) {
	assert := assert.New(t)
	for i, ats :=range {{.OpName}}Tests {
		// safe and not same 
		T, err := {{title .OpName}}(ats.a, ats.b)
		if err != nil{
			t.Errorf("Safe Test of {{title .OpName}} %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "SafeTest {{title .OpName}} %d", i)

		// safe and same
		T, err = {{title .OpName}}(ats.a, ats.b, AsSameType())
		if err != nil {
			t.Errorf("Same Test of {{title .OpName}} %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "SameType Test {{title .OpName}} %d", i)

		// reuse and not same
		T, err = {{title .OpName}}(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil{
			t.Errorf("Reuse Not Same Test of {{title .OpName}} %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "Reuse Not SameTest {{title .OpName}} %d", i)

		// reuse and same
		T, err = {{title .OpName}}(ats.a, ats.b, WithReuse(ats.reuseSame), AsSameType())
		if err != nil{
			t.Errorf("Reuse Same Test of {{title .OpName}} %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Reuse Same Test {{title .OpName}} %d", i)

		// unsafe and same
		T, err = {{title .OpName}}(ats.a, ats.b, UseUnsafe())
		if err != nil{
			t.Errorf("Unsafe Test of {{title .OpName}} %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Unsafe Test {{title .OpName}} %d", i)
	}

}
`

var (
	denseBinOpCmpTestTmpl *template.Template
)

func init() {
	denseBinOpCmpTestTmpl = template.Must(template.New("denseBinOpCmpTest").Funcs(funcMap).Parse(denseBinOpCmpTestRaw))
}

func generateDenseCmpTests(f io.Writer, ts []ArrayType) {
	for _, cbo := range denseCmpBinOps {
		fmt.Fprintf(f, "/* %v */\n\n", cbo.OpName)
		boTest := DenseCmpBinOpTestData{DenseCmpBinOp: cbo}
		for _, t := range ts {
			if !t.isNumber {
				continue
			}
			testSlice0 := []int{1, 2, 3, 4, 3, 2, 1}
			testSlice1 := []int{1, 2, 3, 4, 5, 6, 7}
			scalarValue := 4

			correctBool0 := make([]bool, len(testSlice0))
			correctBool1 := make([]bool, len(testSlice0))
			correctBool2 := make([]bool, len(testSlice0))

			correctSame0 := make([]int, len(testSlice0))
			correctSame1 := make([]int, len(testSlice0))
			correctSame2 := make([]int, len(testSlice0))

			for i, v := range testSlice0 {
				switch cbo.OpName {
				case "elEq":
					correctBool0[i] = scalarValue == v
					correctBool1[i] = v == scalarValue
					correctBool2[i] = v == testSlice1[i]
				case "gt":
					correctBool0[i] = scalarValue > v
					correctBool1[i] = v > scalarValue
					correctBool2[i] = v > testSlice1[i]
				case "gte":
					correctBool0[i] = scalarValue >= v
					correctBool1[i] = v >= scalarValue
					correctBool2[i] = v >= testSlice1[i]
				case "lt":
					correctBool0[i] = scalarValue < v
					correctBool1[i] = v < scalarValue
					correctBool2[i] = v < testSlice1[i]
				case "lte":
					correctBool0[i] = scalarValue <= v
					correctBool1[i] = v <= scalarValue
					correctBool2[i] = v <= testSlice1[i]
				}
			}

			for i, v := range correctBool0 {
				if v {
					correctSame0[i] = 1
				} else {
					correctSame0[i] = 0
				}
			}

			for i, v := range correctBool1 {
				if v {
					correctSame1[i] = 1
				} else {
					correctSame1[i] = 0
				}
			}

			for i, v := range correctBool2 {
				if v {
					correctSame2[i] = 1
				} else {
					correctSame2[i] = 0
				}
			}

			dcbot := DenseCmpBinOpTest{
				ArrayType:   t,
				ScalarValue: scalarValue,
				TestSlice0:  testSlice0,
				TestSlice1:  testSlice1,

				CorrectBool0: correctBool0,
				CorrectBool1: correctBool1,
				CorrectBool2: correctBool2,

				CorrectSame0: correctSame0,
				CorrectSame1: correctSame1,
				CorrectSame2: correctSame2,
			}
			boTest.Types = append(boTest.Types, dcbot)
		}
		denseBinOpCmpTestTmpl.Execute(f, boTest)
		fmt.Fprintln(f, "\n")
	}
}
