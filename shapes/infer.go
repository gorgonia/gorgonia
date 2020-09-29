package shapes

// inferstate keeps track of the state while inferring the types.
type inferstate struct {
	constraints
	subjectsTo

	cur rune // start with 'a'
}

func (s *inferstate) fresh() Var { cur := s.cur; s.cur++; return Var(cur) }

// Infer infers shapes
