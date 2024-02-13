package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path"
	"strings"
	"text/template"

	"github.com/pkg/errors"
)

const genmsg = "// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT."

const importStmt = `import (
gctx "gorgonia.org/gorgonia/internal/context"
"gorgonia.org/gorgonia/internal/datatypes"
"gorgonia.org/tensor"
)`

var (
	gopath, stdopsloc string

	stubsFilename string
	stubsFile     io.WriteCloser

	symdiffsFilename string
	symdiffsFile     io.Reader

	dodiffsFilename string
	dodiffsFile     io.Reader
)

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		usr, err := user.Current()
		if err != nil {
			log.Fatal(err)
		}
		gopath = path.Join(usr.HomeDir, "go")
		stat, err := os.Stat(gopath)
		if err != nil {
			log.Fatal(err)
		}
		if !stat.IsDir() {
			log.Fatal("You need to define a $GOPATH")
		}
	}
	stdopsloc = path.Join(gopath, "src/gorgonia.org/gorgonia/ops/std")
	stubsFilename = path.Join(stdopsloc, "stubs_generated.go")
	symdiffsFilename = path.Join(stdopsloc, "symdiffs.go")
	dodiffsFilename = path.Join(stdopsloc, "dodiffs.go")

	// handle stubsFile
	var err error
	if stubsFile, err = os.OpenFile(stubsFilename, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644); err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(stubsFile, "package stdops\n\n%v\n\n", genmsg)

	// handle symdiffsFile
	if symdiffsFile, err = os.OpenFile(symdiffsFilename, os.O_CREATE|os.O_RDONLY, 0644); err != nil {
		log.Fatal(err)
	}

	// handle dodiffsFile
	if dodiffsFile, err = os.OpenFile(dodiffsFilename, os.O_CREATE|os.O_RDONLY, 0644); err != nil {
		log.Fatal(err)
	}
}

func goimports(filename string) error {
	cmd := exec.Command("goimports", "-w", filename)
	err := cmd.Run()
	if err != nil {
		return errors.Wrapf(err, "Unable to goimports %v", filename)
	}
	return nil
}

func generateBinOp(ops []Op, tmpl *template.Template, unstubbedSymDiffs, unstubbedDoDiffs []string) error {
	for _, op := range ops {
		filename := strings.ToLower(op.Name) + "_generated.go"
		p := path.Join(stdopsloc, filename)
		f, err := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		fmt.Fprintf(f, "package stdops\n\n%v\n\n %v\n\n", genmsg, importStmt)
		if err := tmpl.Execute(f, op); err != nil {
			return errors.Wrapf(err, "Unable to execute binopTmpl for %v", op.Name)
		}
		if err := f.Close(); err != nil {
			return errors.Wrapf(err, "Unable to close %v", p)
		}
		if err := goimports(p); err != nil {
			return err
		}

		// extra: write symdiff to stubs
		if !in(unstubbedSymDiffs, op.Name) {
			if err := binSymDiffTmpl.Execute(stubsFile, op); err != nil {
				return errors.Wrapf(err, "Unable to add %v SymDiff stubs", op.Name)
			}
		}

		// extra: write dodiff to stubs
		if !in(unstubbedDoDiffs, op.Name) {
			if err := doDiffTmpl.Execute(stubsFile, op); err != nil {
				return errors.Wrapf(err, "Unable to add %d DoDiff stubs", op.Name)
			}
		}
	}
	return nil
}

func generateBinOpTest(ops []Op, input binopTestInput, results []binopTestResult, isCmp bool, tmpl *template.Template) error {
	for i, op := range ops {
		opTest := binopTest{Op: op, binopTestInput: input, binopTestResult: results[i], IsCmpRetTrue: true, IsCmp: isCmp}
		filename := strings.ToLower(op.Name) + "_generated_test.go"
		p := path.Join(stdopsloc, filename)
		f, err := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		fmt.Fprintf(f, "package stdops\n\n%v\n\n %v\n\n", genmsg, importStmt)

		if err := tmpl.Execute(f, opTest); err != nil {
			return errors.Wrapf(err, "Unable to execute binopTmpl for %v", op.Name)
		}
		// for cmp
		if isCmp {
			opTest.IsCmpRetTrue = false
			opTest.binopTestInput = cmpTestInputBool
			opTest.binopTestResult = cmpTestResultsBool[i]
			if err := tmpl.Execute(f, opTest); err != nil {
				return errors.Wrapf(err, "Unable to execute binopTmpl for %v", op.Name)
			}
		}
		if err := f.Close(); err != nil {
			return errors.Wrapf(err, "Unable to close %v", p)
		}
		if err := goimports(p); err != nil {
			return err
		}
	}
	return nil
}

func generateAriths(unstubbedSymDiffs, unstubbedDoDiffs []string) error {
	if err := generateBinOp(ariths, arithOpTmpl, unstubbedSymDiffs, unstubbedDoDiffs); err != nil {
		return errors.Wrap(err, "generateAriths.generateBinOp")
	}
	if err := generateBinOpTest(ariths, arithTestInput, arithTestResults, false, arithOpTestTmpl); err != nil {
		return errors.Wrap(err, "generateAriths.generateBinOpTests")
	}

	return nil
}

func generateCmps(unstubbedSymDiffs, unstubbedDoDiffs []string) error {
	if err := generateBinOp(cmps, cmpOpTmpl, unstubbedSymDiffs, unstubbedDoDiffs); err != nil {
		return errors.Wrap(err, "generateCmps.generateBinOp")
	}
	if err := generateBinOpTest(cmps, cmpTestInputSame, cmpTestResultsSame, true, arithOpTestTmpl); err != nil {
		return errors.Wrap(err, "generateCmps.generateBinOpTests")
	}
	return nil
}

func generateUnOps(unstubbedSymDiffs, unstubbedDoDiffs []string) error {
	tmpl := unopTmpl
	for _, op := range unops {
		filename := strings.ToLower(op.Name) + "_generated.go"
		p := path.Join(stdopsloc, filename)
		f, err := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		fmt.Fprintf(f, "package stdops\n\n%v\n\n %v\n\n", genmsg, importStmt)
		if err := tmpl.Execute(f, op); err != nil {
			return errors.Wrapf(err, "Unable to execute unopTmpl for %v", op.Name)
		}
		if err := f.Close(); err != nil {
			return errors.Wrapf(err, "Unable to close %v", p)
		}
		if err := goimports(p); err != nil {
			return err
		}

		// extra: write symdiff to stubs
		if !in(unstubbedSymDiffs, op.Name) {
			if err := binSymDiffTmpl.Execute(stubsFile, op); err != nil {
				return errors.Wrapf(err, "Unable to add %v SymDiff stubs", op.Name)
			}
		}

		// extra: write dodiff to stubs
		if !in(unstubbedDoDiffs, op.Name) {
			if err := doDiffTmpl.Execute(stubsFile, op); err != nil {
				return errors.Wrapf(err, "Unable to add %d DoDiff stubs", op.Name)
			}
		}
	}

	tmpl = unopTestTmpl
	for _, op := range unops {
		tests, ok := unopTests[op.Name]
		if !ok {
			continue
		}

		filename := strings.ToLower(op.Name) + "_generated_test.go"
		p := path.Join(stdopsloc, filename)
		f, err := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		fmt.Fprintf(f, "package stdops\n\n%v\n\n %v\n\n", genmsg, importStmt)

		o := unoptestWithOp{op, tests}
		if err := tmpl.Execute(f, o); err != nil {
			return errors.Wrapf(err, "Unable to execute unopTmpl for %v", op.Name)
		}
		if err := f.Close(); err != nil {
			return errors.Wrapf(err, "Unable to close %v", p)
		}
		if err := goimports(p); err != nil {
			return err
		}
	}
	return nil
}

func generateBinOpAPI() (err error) {
	type apiwrap struct {
		Op
		IsCmp bool
	}

	filename := "api_generated.go"
	filenameTest := "api_generated_test.go"
	p := path.Join(stdopsloc, filename)
	pt := path.Join(stdopsloc, filenameTest)
	f, err := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	g, err := os.OpenFile(pt, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	fmt.Fprintf(f, "package stdops\n\n%v\n\n", genmsg)
	fmt.Fprintf(g, "package stdops\n\n%v\n\n", genmsg)
	for _, o := range ariths {
		if err := binopAPITmpl.Execute(f, apiwrap{o, false}); err != nil {
			return errors.Wrapf(err, "Unable to execute binopAPITmpl for %v", o.Name)
		}

		if err := binopAPITestTmpl.Execute(g, apiwrap{o, false}); err != nil {
			return errors.Wrapf(err, "Unable to execute binopAPITestTmpl for %v", o.Name)
		}
	}
	for _, o := range cmps {
		if err := binopAPITmpl.Execute(f, apiwrap{o, true}); err != nil {
			return errors.Wrapf(err, "Unable to execute binopAPITmpl for %v", o.Name)
		}

		if err := binopAPITestTmpl.Execute(g, apiwrap{o, true}); err != nil {
			return errors.Wrapf(err, "Unable to execute binopAPITestTmpl for %v", o.Name)
		}
	}

	if err := f.Close(); err != nil {
		return errors.Wrapf(err, "Unable to close %v", p)
	}
	if err := g.Close(); err != nil {
		return errors.Wrapf(err, "Unable to close %v", pt)
	}

	if err := goimports(p); err != nil {
		return errors.Wrapf(err, "Unable to goimports %v", p)
	}
	return goimports(pt)
}

func generateInterfaces() error {
	filename := "interfaces_generated.go"
	p := path.Join(stdopsloc, filename)

	f, err := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	fmt.Fprintf(f, "package stdops\n\n%v\n\n", genmsg)

	for _, iface := range unopInterfaces {
		if err := unopInterfaceTempl.Execute(f, iface); err != nil {
			return errors.Wrapf(err, "Unable to execute template for %v", iface.InterfaceName)
		}
	}
	if err := f.Close(); err != nil {
		return errors.Wrapf(err, "Unable to close %v", filename)
	}
	return goimports(p)
}

func finishStubs() error {
	if err := stubsFile.Close(); err != nil {
		return err
	}
	return goimports(stubsFilename)
}

// unstubbed parses the generated package for anything that has been manually moved out from the stubs file and "unstubbed".
func unstubbed(file io.Reader, name string) []string {
	fs := token.NewFileSet()
	sdf, err := parser.ParseFile(fs, name, file, parser.SkipObjectResolution)
	if err != nil {
		log.Fatal(err)
	}
	var ignored []string
	ast.Inspect(sdf, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if !ok {
			return true
		}

		// should have one receiver
		if fn.Recv == nil || len(fn.Recv.List) != 1 {
			return true
		}
		var recv string
		switch r0 := fn.Recv.List[0].Type.(type) {
		case *ast.Ident:
			recv = r0.Name
		case *ast.StarExpr:
			switch x := r0.X.(type) {
			case *ast.Ident:
				recv = x.Name
			case *ast.IndexListExpr:
				recv = x.X.(*ast.Ident).Name
			default:
				log.Fatalf("ERROR: Unsupported StarExpr of %T - value %#v in %v", r0.X, r0.X, name)
			}
		case *ast.IndexListExpr:
			recv = r0.X.(*ast.Ident).Name
		default:
			log.Printf("ERROR: UNSUPPORTED TYPE %v(%T) in %v", fn.Recv.List[0].Type, fn.Recv.List[0].Type, name)
		}

		ignored = append(ignored, strings.TrimSuffix(recv, "Op"))
		return false
	})
	return ignored
}

func main() {
	defer finishStubs()
	unstubbedSymDiffs := unstubbed(symdiffsFile, "symdiffs.go")
	unstubbedDoDiffs := unstubbed(dodiffsFile, "dodiffs.go")

	if err := generateAriths(unstubbedSymDiffs, unstubbedDoDiffs); err != nil {
		log.Fatal(err)
	}
	if err := generateCmps(unstubbedSymDiffs, unstubbedDoDiffs); err != nil {
		log.Fatal(err)
	}
	if err := generateUnOps(unstubbedSymDiffs, unstubbedDoDiffs); err != nil {
		log.Fatal(err)
	}
	if err := generateBinOpAPI(); err != nil {
		log.Fatal(err)
	}

	if err := generateInterfaces(); err != nil {
		log.Fatal(err)
	}

}
