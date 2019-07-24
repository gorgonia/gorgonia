package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"path"
	"path/filepath"
	"strings"
)

type nametypePair struct {
	name string
	*ast.FuncType
}

func functions(decls []ast.Decl) (signatures []nametypePair) {
	for _, decl := range decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			signatures = append(signatures, nametypePair{d.Name.Name, d.Type})
		default:
		}
	}
	return
}

type strRepr struct {
	name     string
	inTypes  []string
	retTypes []string

	printName bool
}

func (s strRepr) String() string {
	buf := new(bytes.Buffer)
	buf.Write([]byte("func "))
	if s.printName {
		buf.Write([]byte(s.name))
	}

	buf.Write([]byte("("))
	for i, v := range s.inTypes {
		buf.Write([]byte(v))
		if i < len(s.inTypes)-1 {
			buf.Write([]byte(", "))
		}
	}
	buf.Write([]byte(") ("))
	for i, v := range s.retTypes {
		buf.Write([]byte(v))
		if i < len(s.retTypes)-1 {
			buf.Write([]byte(", "))
		}
	}
	buf.Write([]byte(")"))
	return buf.String()
}

func processSig(pair nametypePair) strRepr {
	a := pair.FuncType
	var inTypes, retTypes []string
	if a.Params == nil {
		goto next
	}
	for _, field := range a.Params.List {
		names := len(field.Names)
		typ := parseTypeExpr(field.Type)
		if names == 0 {
			inTypes = append(inTypes, typ)
			continue
		}
		for i := 0; i < names; i++ {
			inTypes = append(inTypes, typ)
		}
	}
next:
	if a.Results == nil {
		return strRepr{pair.name, inTypes, retTypes, true}
	}
	for _, field := range a.Results.List {
		names := len(field.Names)
		typ := parseTypeExpr(field.Type)
		if names == 0 {
			retTypes = append(retTypes, typ)
			continue
		}
		for i := 0; i < names; i++ {
			retTypes = append(retTypes, typ)
		}
	}
	return strRepr{pair.name, inTypes, retTypes, true}
}

func parseTypeExpr(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.StarExpr:
		x := parseTypeExpr(e.X)
		return "*" + x
	case *ast.SelectorExpr:
		return parseTypeExpr(e.X) + "." + e.Sel.Name
	case *ast.Ellipsis:
		return "..." + parseTypeExpr(e.Elt)
	case *ast.ArrayType:
		return "[]" + parseTypeExpr(e.Elt)
	default:
		return fmt.Sprintf("%T", expr)
	}
}

func filterSigs(xs []strRepr, fn func(strRepr) bool) (retVal []strRepr) {
	for _, x := range xs {
		if fn(x) {
			retVal = append(retVal, x)
		}
	}
	return
}

func functionSignatures() {
	files := path.Join(gorgonialoc, "*.go")
	matches, err := filepath.Glob(files)

	if err != nil {
		log.Fatal(err)
	}
	fset := token.NewFileSet()

	var allFns []strRepr
	for _, f := range matches {
		file, err := parser.ParseFile(fset, f, nil, parser.AllErrors)
		if err != nil {
			log.Fatal(err)

		}

		fns := functions(file.Decls)
		for _, fn := range fns {
			sig := processSig(fn)
			sig.printName = false
			if strings.Title(sig.name) == sig.name {
				allFns = append(allFns, sig)
			}
		}
	}
	f := func(a strRepr) bool {
		want := []string{"*Node", "error"}
		if len(a.retTypes) != len(want) {
			return false
		}
		for i, v := range a.retTypes {
			if v != want[i] {
				return false
			}
		}
		return true
	}

	signatures := make(map[string]int)
	interesting := filterSigs(allFns, f)
	for _, v := range interesting {
		signatures[fmt.Sprintf("%v", v)]++
	}

	for k, v := range signatures {
		fmt.Printf("%v\t%d\n", k, v)
	}
}
