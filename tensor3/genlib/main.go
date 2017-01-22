package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"reflect"
)

func main() {
	const (
		getSetName = "../dense_getset.go"
	)
	f, err := os.Create(getSetName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	mk := make([]reflect.Kind, 0)
	fmt.Fprintf(f, "package tensor\n/*\nGENERATED FILE. DO NOT EDIT\n*/\n\n")
	for k := reflect.Invalid + 1; k < reflect.UnsafePointer+1; k++ {
		if !isParameterized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			AsSlice.Execute(f, k)
			SimpleSet.Execute(f, k)
			SimpleGet.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
		mk = append(mk, k)
	}
	generic := &ManyKinds{Kinds: mk}
	MakeData.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Set.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Get.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Copy.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	CopyIter.Execute(f, generic)

	// gofmt and goimports this shit
	cmd := exec.Command("goimports", "-w", getSetName)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, getSetName)
	}
}
