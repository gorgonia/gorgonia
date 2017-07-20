package main

import (
	"fmt"
	"io"
)

func writePkgName(f io.Writer, pkg string) {
	switch pkg {
	case tensorPkgLoc:
		fmt.Fprintf(f, "package tensor\n/*\n%v\n*/\n\n", genmsg)
	case stdEngPkgLoc:
		fmt.Fprintf(f, "package stdeng\n/*\n%v\n*/\n\n", genmsg)
	default:
		fmt.Fprintf(f, "package unknown\n/*\n%v\n*/\n\n")
	}
}
