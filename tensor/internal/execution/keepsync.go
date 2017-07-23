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
