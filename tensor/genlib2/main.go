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
	gopath, tensorPkgLoc, stdEngPkgLoc string
)

type Kinds struct {
	Kinds []reflect.Kind
}

func init() {
	gopath = os.Getenv("GOPATH")
	tensorPkgLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor")
	stdEngPkgLoc = path.Join(gopath, "src/github.com/chewxy/gorgonia/tensor/internal/stdeng")
}

func main() {
	log.Printf("allKinds %d", len(allKinds))
	log.Printf("arithBinOps %d", len(arithBinOps))
	gen := makeGenericVecVecs(typedBinOps)
	log.Printf("gen %d", len(gen))
	f, _ := os.Create("test/BLAH.go")
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = false
		g.RequiresIterator = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
	}

	f.Close()
}

func pipeline(pkg, filename string, kinds Kinds, fn func(io.Writer, Kinds)) {
	fullpath := path.Join(pkg, filename)
	f, err := os.Create(fullpath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	writePkgName(f, pkg)
	fn(f, kinds)

	// gofmt and goimports this stuff
	cmd := exec.Command("goimports", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fullpath)
	}

	cmd = exec.Command("sed", "-i", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("sed failed with %v for %q", err, fullpath)
	}

	cmd = exec.Command("gofmt", "-s", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Gofmt failed for %q", fullpath)
	}
}
