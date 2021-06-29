// +build !debug

package shapes

import (
	"io"
)

// logstate is a no-op.
func (p *parser) logstate(name ...interface{}) {}

// printTab is a no-op.
func (p *parser) printTab(w io.Writer) {}
