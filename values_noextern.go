// +build !cuda

package gorgonia

import "github.com/pkg/errors"

// convM2V converts Memory to Value
func convM2V(m External, dev Device, mem Memory, val *Value) (err error) {
	if v, ok := mem.(Value); ok {
		*val = v
		return nil
	}
	return errors.Errorf("Cannot convert Memory 0x%x (%T) to Value", mem, mem)
}
