package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"text/template"
)

type DenseBinOpTest struct {
	ArrayType

	ScalarValue            int
	ScalarValue2           int
	TestSlice0, TestSlice1 []int

	CorrectSV0, CorrectSV1, CorrectSV2 []int
	CorrectVS0, CorrectVS1, CorrectVS2 []int
	CorrectVV0, CorrectVV1, CorrectVV2 []int
}

type DenseBinOpTestData struct {
	DenseBinOp
	Types []DenseBinOpTest
}

const denseBinOpTestRaw = `var {{.OpName}}Tests = []struct{
	a interface{}
	b interface{}

	reuse *Dense
	incr *Dense

	correct0, correct1, correct2 interface{}
}{
	{{range .Types -}}
	// {{title .Of}}
	{a: {{.Of}}({{.ScalarValue}}), 
	 b: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice0 -}}{{printf "%d" .}},{{end -}} })),
	 incr: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ 100, 100, 100, 100 })), 
	 reuse: New(Of({{title .Of}}), WithBacking([]{{.Of}}{200, 200, 200, 200})), 
	 correct0: []{{.Of}}{ {{range .CorrectSV0 -}} {{printf "%d" .}}, {{end -}} },
	 correct1: []{{.Of}}{ {{range .CorrectSV1 -}} {{printf "%d" .}}, {{end -}} },
	},
	{a: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice0 -}}{{printf "%d" .}},{{end -}} })),
	 b: {{.Of}}({{.ScalarValue2}}), 
	 incr: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ 100, 100, 100, 100 })), 
	 reuse: New(Of({{title .Of}}), WithBacking([]{{.Of}}{200, 200, 200, 200})), 
	 correct0: []{{.Of}}{ {{range .CorrectVS0 -}} {{printf "%d" .}}, {{end -}} },
	 correct1: []{{.Of}}{ {{range .CorrectVS1 -}} {{printf "%d" .}}, {{end -}} },	
	},
	{a: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice0 -}}{{printf "%d" .}},{{end -}} })),
	 b: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ {{range .TestSlice1 -}}{{printf "%d" .}},{{end -}} })),
	 incr: New(Of({{title .Of}}), WithBacking([]{{.Of}}{ 100, 100, 100, 100 })), 
	 reuse: New(Of({{title .Of}}), WithBacking([]{{.Of}}{200, 200, 200, 200})), 
	 correct0: []{{.Of}}{ {{range .CorrectVV0 -}} {{printf "%d" .}}, {{end -}} },
	 correct1: []{{.Of}}{ {{range .CorrectVV1 -}} {{printf "%d" .}}, {{end -}} },
	},
	{{end -}}
}

func Test{{title .OpName}}(t *testing.T){
	assert := assert.New(t)
	for i, ats := range {{.OpName}}Tests{
		// safe
		T, err := {{title .OpName}}(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Safe Test {{title .OpName}} %d", i)

		// incr
		T, err = {{title .OpName}}(ats.a, ats.b, WithIncr(ats.incr))
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct1, T.Data(), "Incr Test {{title .OpName}} %d", i)

		// reuse
		T, err = {{title .OpName}}(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil{
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Reuse Test {{title .OpName}} %d", i)

		// unsafe
		T, err = {{title .OpName}}(ats.a, ats.b, UseUnsafe())
		if err != nil{
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Unsafe Test {{title .OpName}} %d", i)
	}
}
`

func generateDenseArithTests(fileName string, ts []ArrayType) {
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

	fmt.Fprintln(f, "package tensor \n\n")

	for _, bo := range denseBinOps {
		fmt.Fprintf(f, "/* %v */\n\n", bo.OpName)
		boTest := DenseBinOpTestData{DenseBinOp: bo}
		for _, t := range ts {
			if !t.isNumber {
				continue
			}
			testSlice0 := []int{1, 2, 3, 4}
			testSlice1 := []int{1, 2, 3, 4}
			s := 1
			s2 := 1

			if bo.OpName == "div" {
				s = 24

				for i := range testSlice0 {
					testSlice0[i] *= 2
					testSlice1[i] *= 2
				}
				s2 = 2
			}

			correctSV0 := make([]int, len(testSlice0))
			correctSV1 := make([]int, len(testSlice0))

			correctVS0 := make([]int, len(testSlice0))
			correctVS1 := make([]int, len(testSlice0))

			correctVV0 := make([]int, len(testSlice0))
			correctVV1 := make([]int, len(testSlice0))

			for i, v := range testSlice0 {
				switch bo.OpName {
				case "add":
					correctSV0[i] = s + v       // safe, unsafe and reuse
					correctSV1[i] = s + v + 100 // incr

					correctVS0[i] = v + s
					correctVS1[i] = v + s + 100

					correctVV0[i] = v + testSlice1[i]
					correctVV1[i] = v + testSlice1[i] + 100
				case "sub":
					correctSV0[i] = s - v       // safe, unsafe and reuse
					correctSV1[i] = s - v + 100 // incr

					correctVS0[i] = v - s
					correctVS1[i] = v - s + 100

					correctVV0[i] = v - testSlice1[i]
					correctVV1[i] = v - testSlice1[i] + 100

				case "mul":
					correctSV0[i] = s * v     // safe, unsafe and reuse
					correctSV1[i] = s*v + 100 // incr

					correctVS0[i] = v * s
					correctVS1[i] = v*s + 100

					correctVV0[i] = v * testSlice1[i]
					correctVV1[i] = v*testSlice1[i] + 100
				case "div":
					correctSV0[i] = int(float64(s) / float64(v))     // safe, unsafe and reuse
					correctSV1[i] = int(float64(s)/float64(v)) + 100 // incr

					correctVS0[i] = int(float64(v) / float64(s2))
					correctVS1[i] = int(float64(v)/float64(s2)) + 100

					correctVV0[i] = int(float64(v) / float64(testSlice1[i]))
					correctVV1[i] = int(float64(v)/float64(testSlice1[i])) + 100
				case "pow":
					correctSV0[i] = int(math.Pow(float64(s), float64(v)))       // safe, unsafe and reuse
					correctSV1[i] = int(math.Pow(float64(s), float64(v))) + 100 // incr

					correctVS0[i] = int(math.Pow(float64(v), float64(s)))
					correctVS1[i] = int(math.Pow(float64(v), float64(s))) + 100

					correctVV0[i] = int(math.Pow(float64(v), float64(testSlice1[i])))
					correctVV1[i] = int(math.Pow(float64(v), float64(testSlice1[i]))) + 100
				}
			}

			// special overflow handling for byte
			if t.Of == "byte" {
				for i := range testSlice0 {
					correctSV0[i] = int(byte(correctSV0[i]))
					correctSV1[i] = int(byte(correctSV1[i]))

					correctVS0[i] = int(byte(correctVS0[i]))
					correctVS1[i] = int(byte(correctVS1[i]))

					correctVV0[i] = int(byte(correctVV0[i]))
					correctVV1[i] = int(byte(correctVV1[i]))
				}
			}

			dbot := DenseBinOpTest{
				ArrayType:    t,
				ScalarValue:  s,
				ScalarValue2: s2,
				TestSlice0:   testSlice0,
				TestSlice1:   testSlice1,

				CorrectSV0: correctSV0,
				CorrectSV1: correctSV1,

				CorrectVS0: correctVS0,
				CorrectVS1: correctVS1,

				CorrectVV0: correctVV0,
				CorrectVV1: correctVV1,
			}
			boTest.Types = append(boTest.Types, dbot)
		}
		denseBinOpTestTmpl.Execute(f, boTest)
		fmt.Fprintln(f, "\n")
	}

	// gofmt and goimports this shit
	cmd := exec.Command("goimports", "-w", fileName)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fileName)
	}
}

var (
	denseBinOpTestTmpl *template.Template
)

func init() {
	denseBinOpTestTmpl = template.Must(template.New("DenseBinOpTest").Funcs(funcMap).Parse(denseBinOpTestRaw))
}
