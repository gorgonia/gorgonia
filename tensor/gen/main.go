package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/sergi/go-diff/diffmatchpatch"
)

var diffOnly = flag.Bool("d", false, "display diffs instead of rewriting files")
var list = flag.Bool("l", false, "list files")
var verbose = flag.Bool("v", false, "verbose")

var arithables []string
var matop []string
var linalgs []string
var cmp []string
var asm []string

var generates = map[string]string{
	"bool":    "b",
	"int":     "i",
	"float32": "f32",
	// "byte": "u8",
}

var manualcopy = []string{
	"format.go",
	"format_test.go",
}

var allignores = []string{
	"compat.go",
	"compat_test.go",
	"io.go",
	"io_test.go",

	"matrix.go",
	"matrix_test.go",
}

var boolignores = []string{
	"argmethods.go",
	"argmethods_test.go",
	"blas.go",

	"matop_test.go",
	"tensor_test.go",
	"views_test.go",

	"utils.go",
	"utils_test.go",

	"test_test.go",

	"example_reduction_test.go",
}

var intignores = []string{
	"blas.go",
	"arith_floats.go",
	"arith_linalg_api.go",
	"arith_linalg_api_test.go",
	"arith_linalg_methods.go",
	"arith_linalg_methods_test.go",
	"arith_linalg_norms.go",
	"arith_linalg_norms_test.go",

	"test_test.go",
	"argmethods_test.go", // because argmethods test is solely float-related
}

var replacements = map[string]map[string]string{
	"bool": map[string]string{
		// matop.go - incr doesn't make sense for bool
		`	case t.viewOf != nil && incr:
		it := types.NewFlatIterator(t.AP)
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			res[next] += fn(t.data[next])
		}`: "",
		`	case t.viewOf == nil && incr:
		for i, v := range t.data {
			res[i] += fn(v)
		}`: "",
	},
	"int": map[string]string{
		// argmethods.go
		"ti.NewTensor":   "NewTensor",
		"ti.WithShape":   "WithShape",
		"ti.WithBacking": "WithBacking",
		"ti.AsScalar":    "AsScalar",
		"*ti.Tensor":     "*Tensor",

		// argmethods_test.go
		"var maxes *ti.Tensor": "var maxes *Tensor",

		// utils.go
		"r[i] = rand.NormInt()": "r[i] = rand.Int()",
		`		// TODO: Maybe error instead of this?
		if math.IsNaN(v) || math.IsInf(v, 1) {
			max = i
			f = v
			break
		}`: "",
		`		// TODO: Maybe error instead of this?
		if math.IsNaN(v) || math.IsInf(v, -1) {
			min = i
			f = v
			break
		}`: "",

		// arith.go
		"a[i] = math.Pow(v, b[i])": "a[i] = int(math.Pow(float64(v), float64(b[i])))",
		"a[i] = math.Pow(v, s)":    "a[i] = int(math.Pow(float64(v), float64(s)))",
		"a[i] = math.Pow(s, v)":    "a[i] = int(math.Pow(float64(s), float64(v)))",
		"math.Mod(a, b)":           "int(math.Mod(float64(a), float64(b)))",

		// arith_api_unary.go
		"if v < min || math.IsInf(v, -1)": "if v < min",
		"if v > max || math.IsInf(v, 1)":  "if v > max",

		// arith_api_unary_test.go
		"correct[i] = math.Sqrt(v)": "correct[i] = int(math.Sqrt(float64(v)))",
		`	backingC := []int{-1, -2, -3, -4}
	Tc := NewTensor(WithBacking(backingC))
	Sqrt(Tc, types.UseUnsafe())
	for _, v := range backingC {
		if !math.IsNaN(v) {
			t.Error("Expected NaN")
		}
	}`: "", // remove this test entirely

		// arith_go.go
		`		if b[i] == 0 {
			a[i] = math.Inf(0)
			continue
		}`: "",
		"a[i] = math.Sqrt(v)":          "a[i] = int(math.Sqrt(float64(v)))",
		"a[i] = int(1) / math.Sqrt(v)": "a[i] = 1 / int(math.Sqrt(float64(v)))",

		// arith_incr.go
		`		if c[i] == 0 {
			a[i] = math.Inf(0)
			continue
		}`: "",
		`		if v == 0 {
			a[i] = math.Inf(0)
			continue
		}`: "",
		"a[i] += math.Pow(v, c[i])": "a[i] += int(math.Pow(float64(v), float64(c[i])))",
		"a[i] += math.Pow(v, c)":    "a[i] += int(math.Pow(float64(v), float64(c)))",
		"a[i] += math.Pow(c, v)":    "a[i] += int(math.Pow(float64(c), float64(v)))",

		// arith_reduction_methods_test.go
		"expectedData = []int{0, 0.25, 0.4}": "expectedData = []int{0, 0, 0}",

		// arith_safe.go
		"retVal[i] = math.Pow(v, s)": "retVal[i] = int(math.Pow(float64(v), float64(s)))",
		"retVal[i] = math.Pow(s, v)": "retVal[i] = int(math.Pow(float64(s), float64(v)))",

		//tensor_test.go
		"AsScalar(3.1415)": "AsScalar(3)",

		// Flags stuff
		"AsTensorF64": "AsTensorInt",
		// General stuff
		"// +build !avx,!sse": "",
	},
	"float32": map[string]string{
		// general
		"math.IsNaN(": "math32.IsNaN(",
		"math.IsInf(": "math32.IsInf(",
		"math.Pow(":   "math32.Pow(",
		"math.Mod(":   "math32.Mod(",
		"math.Sqrt(":  "math32.Sqrt(",
		"math.Inf(":   "math32.Inf(",
		"math.NaN()":  "math32.NaN()",
		"math.Abs":    "math32.Abs",

		// utils.go
		"r[i] = rand.NormFloat32()": "r[i] = float32(rand.NormFloat64())",

		// BLAS stuff
		"Ddot":       "Sdot",
		"Dgemv":      "Sgemv",
		"Dgemm":      "Sgemm",
		"Dger":       "Sger",
		"blas64.Use": "blas32.Use",

		// Flags stuff
		"AsTensorF64": "AsTensorF32",
	},
}

func packageName(t string) string { return fmt.Sprintf("package tensor%s", generates[t]) }

func includes(t string) (retVal []string) {
	switch t {
	case "bool":
		return matop
	case "int":
		retVal = append(retVal, matop...)
		retVal = append(retVal, arithables...)
		retVal = append(retVal, cmp...)
	case "float32":
		retVal = append(retVal, matop...)
		retVal = append(retVal, arithables...)
		retVal = append(retVal, linalgs...)
		retVal = append(retVal, cmp...)
		// case "byte":
	}
	return
}

func ignores(t string) (retVal []string) {
	switch t {
	case "bool":
		retVal = append(retVal, allignores...)
		retVal = append(retVal, manualcopy...)
		retVal = append(retVal, boolignores...)
	case "int":
		retVal = append(retVal, allignores...)
		retVal = append(retVal, manualcopy...)
		retVal = append(retVal, intignores...)
		retVal = append(retVal, linalgs...)
	case "float32":
		retVal = append(retVal, allignores...)
		retVal = append(retVal, manualcopy...)
	}
	return
}

func towrite(t string) (retVal []string) {
	incl := includes(t)
	excl := ignores(t)

inclloop:
	for _, in := range incl {
		for _, ex := range excl {
			if in == ex {
				continue inclloop
			}
		}
		retVal = append(retVal, in)
	}
	return
}

func pasta(t string) {
	tw := towrite(t)
	capped := strings.Title(t)

	for _, fn := range tw {
		readFileName := "../f64/" + fn
		read, err := ioutil.ReadFile(readFileName)
		if err != nil {
			panic(err)
		}

		// replace package name
		replaced := strings.Replace(string(read), "package tensorf64", packageName(t), -1)

		if t == "bool" {
			// replace default0
			replaced = strings.Replace(replaced, "= float64(0) //@DEFAULTZERO", "= false //@DEFAULTZERO", -1)

			// replace default1
			replaced = strings.Replace(replaced, "= float64(1) //@DEFAULTONE", "= true //@DEFAULTONE", -1)
		}

		// replace types
		replaced = strings.Replace(replaced, "float64", t, -1)

		// replace named types
		replaced = strings.Replace(replaced, "Float64", capped, -1)

		//specific replacements
		repl := replacements[t]
		for tbr, r := range repl {
			replaced = strings.Replace(replaced, tbr, r, -1)
		}

		destFileName := fmt.Sprintf("../%s/%s", generates[t], fn)
		if *diffOnly {
			orig, err := ioutil.ReadFile(destFileName)
			if err != nil {
				log.Printf("Error while reading %v. Error: %v", destFileName, err)
				continue
			}

			dmp := diffmatchpatch.New()
			diffs := dmp.DiffMain(string(orig), replaced, false)
			patches := dmp.PatchMake(diffs)
			if len(patches) > 0 {
				fmt.Printf("%v Diff: \n%v\n", destFileName, dmp.PatchToText(patches))
			}
		} else {
			if err = ioutil.WriteFile(destFileName, []byte(replaced), 0664); err != nil {
				panic(err)
			}

			// gofmt and goimports this shit
			cmd := exec.Command("goimports", "-w", destFileName)
			if err = cmd.Run(); err != nil {
				log.Println(err)
			}
		}

	}
}

func walk(dir string, info os.FileInfo, _ error) (err error) {
	if info.IsDir() && info.Name() != "f64" {
		return filepath.SkipDir
	}

	fn := path.Base(dir)
	var matchedasm, matchedArith, matchedLinalg, matchedCmp, matchedGo bool
	if matchedasm, err = filepath.Match("*asm*", info.Name()); err == nil && matchedasm {
		asm = append(asm, fn)
		return nil
	}

	if matchedLinalg, err = filpath.Match("*linalg*.go", info, Name()); err == nil && matchedLinalg {
		linalgs = append(linalgs, fn)
		return nil
	}

	if matchedArith, err = filepath.Match("arith*.go", info.Name()); err == nil && matchedArith {
		arithables = append(arithables, fn)
		return nil
	}

	if matchedCmp, err = filepath.Match("cmp_*.go", info.Name()); err == nil && matchedCmp {
		cmp = append(cmp, fn)
		return nil
	}

	if matchedGo, err = filepath.Match("*.go", info.Name()); err == nil && matchedGo {
		matop = append(matop, fn)
		return nil
	}
	return
}

func vprintf(format string, attrs ...interface{}) {
	if *verbose {
		fmt.Printf(format, attrs...)
	}
}

func main() {
	flag.Parse()

	filepath.Walk("../f64", walk)
	vprintf("arithables: %v\n", arithables)
	vprintf("cmp: %v\n", cmp)
	vprintf("matop: %v\n", matop)

	for t, short := range generates {
		dirname := "../" + short

		if _, err := os.Stat(dirname); os.IsNotExist(err) {
			vprintf("Created %s\n", dirname)
			os.Mkdir(dirname, 0777)
		}

		vprintf("Working on %v. These files will be overwritten:\n", dirname)
		for _, w := range towrite(t) {
			vprintf("\t%v\n", w)
			filename := filepath.Join(dirname, w)
			if !*diffOnly {
				if err := os.Remove(filename); err != nil {
					if !os.IsNotExist(err) {
						panic(err)
					}
				}
			}
		}
		pasta(t)

	}
}
