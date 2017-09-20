package execution

// Iterator is the generic iterator interface
type Iterator interface {
	Start() (int, error)
	Next() (int, error)
	NextValidity() (int, bool, error)
	NextValid() (int, int, error)
	NextInvalid() (int, int, error)
	Reset()
	SetReverse()
	SetForward()
	Coord() []int
	Done() bool
}

// NoOpError is a useful for operations that have no op.
type NoOpError interface {
	NoOp() bool
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

func handleNoOp(err error) error {
	if err == nil {
		return nil
	}

	if _, ok := err.(NoOpError); !ok {
		return err
	}
	return nil
}
