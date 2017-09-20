package main

import (
	"fmt"
	"io"
)

func writePkgName(f io.Writer, pkg string) {
	switch pkg {
	case tensorPkgLoc:
		fmt.Fprintf(f, "// %s\n\npackage tensor\n\n", genmsg)
	case execLoc:
		fmt.Fprintf(f, "// %s\n\npackage execution\n\n", genmsg)
	case storageLoc:
		fmt.Fprintf(f, "// %s\n\npackage storage\n\n", genmsg)
	default:
		fmt.Fprintf(f, "// %s\n\npackage unknown\n\n", genmsg)
	}
}
