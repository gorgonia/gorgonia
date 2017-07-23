package main

import (
	"fmt"
	"io"
)

func writePkgName(f io.Writer, pkg string) {
	switch pkg {
	case tensorPkgLoc:
		fmt.Fprintf(f, "package tensor\n/*\n%v\n*/\n\n", genmsg)
	case execLoc:
		fmt.Fprintf(f, "package execution\n/*\n%v\n*/\n\n", genmsg)
	case storageLoc:
		fmt.Fprintf(f, "package storage\n/*\n%v\n*/\n\n", genmsg)
	default:
		fmt.Fprintf(f, "package unknown\n/*\n%v\n*/\n\n", genmsg)
	}
}
