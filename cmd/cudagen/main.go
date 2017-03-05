package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/chewxy/cu"
)

var debug = flag.Bool("debug", false, "compile with debug mode (-linelinfo is added to nvcc call)")

func stripExt(fullpath string) string {
	_, filename := filepath.Split(fullpath)
	ext := path.Ext(filename)
	name := strings.TrimRight(filename, ext)
	return name
}

func compileCUDA(src, targetLoc string, maj, min int) {
	name := stripExt(src)
	arch := fmt.Sprintf("-arch=compute_%d%d", maj, min)
	output := fmt.Sprintf("-o=%v", path.Join(targetLoc, name+".ptx"))

	var stderr bytes.Buffer

	var slow *exec.Cmd
	if *debug {
		slow = exec.Command("nvcc", output, arch, "-lineinfo", "-ptx", "-Xptxas", "--allow-expensive-optimizations", "-fmad=false", "-ftz=false", "-prec-div=true", "-prec-sqrt=true", src)
	} else {
		slow = exec.Command("nvcc", output, arch, "-ptx", "-Xptxas", "--allow-expensive-optimizations", "-fmad=false", "-ftz=false", "-prec-div=true", "-prec-sqrt=true", src)
	}

	slow.Stderr = &stderr
	if err := slow.Run(); err != nil {
		log.Fatalf("Failed to compile with nvcc: %v.", stderr.String())
	}

	// fast := exec.Command("nvcc", output, arch, "-ptx", "-Xptxas", "-allow-expensive-optimizations", "-fmad=false", "-use_fast_math", src)
	// if err := fast.Run(); err != nil {
	// 	log.Fatal(err)
	// }
}

func main() {
	flag.Parse()
	var devices int
	var err error

	if devices, err = cu.NumDevices(); err != nil {
		log.Fatalf("error while finding number of devices: %+v", err)
	}

	if devices == 0 {
		log.Fatal("No CUDA-capable devices found")
	}

	// get the lowest possible compute capability
	major := int(^uint(0) >> 1)
	minor := int(^uint(0) >> 1)
	for d := 0; d < devices; d++ {
		var dev cu.Device
		if dev, err = cu.GetDevice(d); err != nil {
			log.Fatalf("Unable to get GPU%d - %+v", err)
		}

		maj, min, err := dev.ComputeCapability()
		if err != nil {
			log.Fatalf("Unable to get compute compatibility of GPU%d - %v", d, err)
		}
		if maj > 0 && maj < major {
			major = maj
			minor = min
			continue
		}

		if min > 0 && min < minor {
			minor = min
		}
	}

	gopath := os.Getenv("GOPATH")
	gorgoniaLoc := path.Join(gopath, "src/github.com/chewxy/gorgonia")
	cuLoc := path.Join(gorgoniaLoc, "cuda modules/src")
	ptxLoc := path.Join(gorgoniaLoc, "cuda modules/target")

	ptxname := "%v.ptx"

	matches, err := filepath.Glob(fmt.Sprintf("%v/*.cu", cuLoc))
	if err != nil {
		log.Fatal(err)
	}

	for _, match := range matches {
		compileCUDA(match, ptxLoc, major, minor)
	}

	m := make(map[string][]byte)
	for _, match := range matches {
		name := stripExt(match)
		ptxfile := path.Join(ptxLoc, fmt.Sprintf(ptxname, name))
		data, _ := ioutil.ReadFile(ptxfile)
		m[name] = data
	}

	cudamodules := path.Join(gorgoniaLoc, "cudamodules.go")
	f, err := os.OpenFile(cudamodules, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	defer f.Close()

	f.Write([]byte("// +build cuda\n\npackage gorgonia\n"))
	for name, data := range m {
		fmt.Fprintf(f, "const %vPTX = `", name)
		f.Write(data)
		fmt.Fprintf(f, "`\n")
	}

	initFn := []byte(`func init() {
		cudaStdLib = map[string]string {
	`)
	f.Write(initFn)
	for name := range m {
		fmt.Fprintf(f, "\"%v\": %vPTX,\n", name, name)
	}
	f.Write([]byte("}\n\t}\n"))

	// gofmt and goimports this shit
	cmd := exec.Command("gofmt", "-w", gorgoniaLoc+"/cudamodules.go")
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, gorgoniaLoc+"/cudamodules.go")
	}
}
