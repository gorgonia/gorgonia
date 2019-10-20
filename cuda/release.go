// +build !debug

package cuda

import (
	"log"
	"os"
)

var logger = log.New(os.Stderr, "", 0)
var replacement = "\n"

func tabcount() int { return 0 }

func enterLogScope() {}

func leaveLogScope() {}

func logf(format string, others ...interface{}) {}

func allocatorLogf(format string, attrs ...interface{}) {}
