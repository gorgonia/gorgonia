package stdeng 

// NoOpError is a useful for operations that have no op.
type NoOpError interface {
	NoOp() bool
}

func handleNoOp(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := err.(NoOpError); !ok {
		return err
	}
	return nil
}