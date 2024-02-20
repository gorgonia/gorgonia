package internal

import "gorgonia.org/gorgonia/internal/errors"

func HandleNoOp(err error) error {
	if _, ok := err.(errors.NoOpError); ok {
		return nil
	}
	if err == nil {
		return nil
	}
	return err
}
