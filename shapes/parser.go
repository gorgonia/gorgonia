package shapes

import (
	"bytes"
	"fmt"
	"strconv"
	"unicode"

	"github.com/pkg/errors"
)

// Parse parses a string and returns a shape expression.
func Parse(a string) (retVal Expr, err error) {
	q, err := lex(a)
	if err != nil {
		return nil, err
	}

	p := newParser(true)
	err = p.parse(q)
	p.logstate()
	if len(p.stack) <= 0 {
		p.printTab(nil)
		return nil, errors.Errorf("WTF?")
	}

	p.printTab(nil)
	retVal = p.stack[0].(Expr)
	return
}

type parser struct {
	queue      []tok           // incoming string of tokens
	stack      []substitutable // "working" stack
	infixStack []tok           // stack of operators
	qptr       int             // queue pointer

	log *bytes.Buffer
}

func newParser(log bool) *parser {
	var l *bytes.Buffer
	if log {
		l = new(bytes.Buffer)
	}
	return &parser{
		log: l,
	}
}

func (p *parser) pop() substitutable {
	if len(p.stack) == 0 {
		panic("cannot pop")
	}

	retVal := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	return retVal
}

func (p *parser) popExpr() (Expr, error) {
	s := p.pop()
	e, ok := s.(Expr)
	if !ok {
		return nil, errors.Errorf("Expected an Expr. Got %v of %v instead", s, s)
	}
	return e, nil
}

func (p *parser) push(a substitutable) { p.stack = append(p.stack, a) }

func (p *parser) pushInfix(t tok) { p.infixStack = append(p.infixStack, t) }

func (p *parser) popInfix() tok {
	if len(p.infixStack) == 0 {
		panic("Cannot pop infix stack")
	}
	retVal := p.infixStack[len(p.infixStack)-1]
	p.infixStack = p.infixStack[:len(p.infixStack)-1]
	return retVal
}

func (p *parser) cur() (tok, error) {
	if p.qptr < 0 || p.qptr >= len(p.queue) {
		return tok{}, errors.Errorf("Cannot get current token. Pointer: %d. Queue: %v", p.qptr, len(p.queue))
	}
	return p.queue[p.qptr], nil
}

func (p *parser) compose(g, f func() error, fname, gname string) func() error {
	return func() error {
		if err := f(); err != nil {
			return errors.Wrapf(err, "In composed functions %v ∘ %v. %v failed", fname, gname, fname)
		}
		if err := g(); err != nil {
			return errors.Wrapf(err, "In composed functions %v ∘ %v. %v failed", fname, gname, gname)
		}
		return nil
	}
}

// compareCur compares the cur token with  the top of the infix stack. It returns a function that the parser should take
func (p *parser) compareCur() func() error {
	p.logstate()
	t, err := p.cur()
	if err != nil {
		return func() error { return errors.Wrap(err, "Cannot compareCur()") }
	}
	switch t.t {
	case digit:
		return p.pushNum
	case letter:
		return p.pushVar
	default:
		if len(p.infixStack) == 0 {
			// push the current to infixStack
			return p.pushCurTok
		}
		// check if current token has greater op prec than top of stack
		top := p.infixStack[len(p.infixStack)-1]
		topPrec := opprec[top.v]
		curPrec := opprec[t.v]

		// if current is negative, we need to resolve until the infixStack has len 0 then pushCurTok
		if curPrec < 0 {
			if err := p.pushCurTok(); err != nil {
				return func() error { return errors.Wrap(err, "curPrec < 0") }
			}
			return p.resolveAllInfix
		}

		if curPrec > topPrec {
			return p.pushCurTok
		}

		// check special case of arrows (which are right assoc)
		if top.t == arrow && t.t == arrow {
			return p.pushCurTok
		}

		// otherwise resolve first then pushcurtok
		return p.compose(p.pushCurTok, p.resolveInfix, "resolveInfix", "pushCurTok")
	}
}

func (p *parser) incrQPtr() error { p.qptr++; return nil }

// pushVar pushes a var on to the values stack.
func (p *parser) pushVar() error {
	t, err := p.cur()
	if err != nil {
		return err
	}
	p.push(Var(t.v))

	// special case: check for '['
	if p.qptr == len(p.queue)-1 {
		return nil
	}
	if p.queue[p.qptr+1].v == '[' {
		p.push(SliceOf{})
		p.pushInfix(p.queue[p.qptr+1])
		p.incrQPtr()
		return nil
	}
	return nil
}

// pushNum pushes a number (typed as a Size) onto the values stack.
func (p *parser) pushNum() error {
	t, err := p.cur()
	if err != nil {
		return err
	}
	p.push(Size(int(t.v)))
	return nil
}

// pushCurTok pushes the current token into the infixStack
func (p *parser) pushCurTok() error {
	t, err := p.cur()
	if err != nil {
		return err
	}
	p.pushInfix(t)
	return nil
}

func (p *parser) parse(q []tok) (err error) {
	p.queue = q
	for p.qptr < len(p.queue) {
		if err = p.parseOne(); err != nil {
			// p.printTab(nil)
			return err
		}
		p.incrQPtr()
	}
	if len(p.infixStack) > 0 {
		if err := p.resolveAllInfix(); err != nil {
			return errors.Wrap(err, "Unable to resolve all infixes while in .parse")
		}
	}
	//p.printTab(nil)
	return nil
}

func (p *parser) parseOne() error {
	t, err := p.cur()
	if err != nil {
		return errors.Wrap(err, "Unable to parseOne")
	}

	// special cases: ()
	if t.v == '(' {
		if p.qptr == len(p.queue)-1 {
			return errors.Errorf("Dangling open paren '(' at %v", t.l)
		}
		if p.queue[p.qptr+1].v == ')' {
			p.push(Shape{})
			p.incrQPtr()
			return nil
		}
	}

	fn := p.compareCur()
	if err := fn(); err != nil {
		return errors.Wrap(err, "Unable to parseOne")
	}
	return nil
}

// resolveAllInfix resolves all the infixes in the infixStack. The result will be an empty infix stack.
func (p *parser) resolveAllInfix() error {
	var count int
	for len(p.infixStack) > 0 {
		if err := p.resolveInfix(); err != nil {
			return errors.Wrapf(err, "Unable to resolve all infixes. %d processed.", count)
		}
		count++
	}
	return nil
}

func (p *parser) resolveInfix() error {
	last := p.popInfix()
	var err error
	switch last.t {
	case unop:
		if err = p.resolveUnOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve unop %v.", last)
		}
	case binop:
		if err = p.resolveBinOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve binop %v.", last)
		}
	case cmpop:
		if err = p.resolveCmpOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve cmpop %v.", last)
		}
	case logop:
		if err = p.resolveLogOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve logop %v.", last)
		}
	case arrow:
		if err := p.resolveArrow(last); err != nil {
			return errors.Wrapf(err, "Cannot resolve arrow %v.", last)
		}
	case parenR:
		if err := p.resolveA(); err != nil {
			return errors.Wrap(err, "Cannot resolve A.")
		}
	case comma:
		if err := p.resolveComma(last); err != nil {
			return errors.Wrapf(err, "Cannot resolve comma %v", last)
		}
	case braceR:
		if err := p.resolveCompound(); err != nil {
			return errors.Wrapf(err, "Cannot resolve compound %v", last)
		}
	case brackR:
		if err := p.resolveSlice(); err != nil {
			return errors.Wrapf(err, "Cannot resolve slice %v", last)
		}
	case colon:
		if err := p.resolveColon(); err != nil {
			return errors.Wrapf(err, "Cannot resolve colon %v", last)
		}
	default:
		// log.Printf("last {%v %c %v} is unhandled", last.t, last.v, last.l)
	}

	return nil
}

// resolveGroup resolves groupings of `(...)` and `[...]`
func (p *parser) resolveGroup(want rune) error {
	var bw []tok
	for i := len(p.infixStack) - 1; i >= 0; i-- {
		t := p.popInfix()
		// keep going until you find the first '[' or '(', whatever that was passed into `want`
		if t.v == want {
			break
		}
		bw = append(bw, t)
	}
	reverse(bw)
	backup := p.infixStack
	p.infixStack = bw
	if err := p.resolveAllInfix(); err != nil {
		return errors.Wrapf(err, "Unable to resolveGroup. Group: %q", want)
	}
	if len(p.infixStack) > 0 {
		// error? TODO
	}
	p.infixStack = backup
	return nil
}

func (p *parser) resolveA() error {
	if err := p.resolveGroup('('); err != nil {
		return errors.Wrap(err, "Unable to resolveA.")
	}

	last := p.pop()
	if abs, ok := last.(Abstract); ok {
		if shp, ok := abs.ToShape(); ok {
			p.push(shp)
			return nil
		}
	}
	// `last` is an Expr that is not a Shape or Abstract.
	p.push(last)
	return nil
}

func (p *parser) resolveArrow(t tok) error {
	snd, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve snd of arrow at %d as Expr.", t.l)
	}
	fst, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve fst of arrow at %d as Expr.", t.l)
	}
	arr := Arrow{
		A: fst,
		B: snd,
	}
	p.push(arr)
	return nil
}

func (p *parser) resolveComma(t tok) error {
	// comma is a binary option. However if it's a trailing comma, then there is no need.
	snd := p.pop()

	if len(p.stack) == 0 {
		switch s := snd.(type) {
		case Abstract:
			p.push(s)
		case Sizelike:
			p.push(Abstract{s})
		}
		return nil
	}

	fst := p.pop()
	switch f := fst.(type) {
	case Abstract:
		switch s := snd.(type) {
		case Sizelike:
			f = append(f, s)
			p.push(f)
			return nil
		case Conser:
			ret := f.Cons(s).(substitutable)
			p.push(ret)
			return nil
		}
	case Sizelike:
		switch s := snd.(type) {
		case Sizelike:
			p.push(Abstract{f, s})
			return nil
		case Shape:
			abs := Abstract{f}
			ret := abs.Cons(s).(substitutable)
			p.push(ret)
			return nil
		case Abstract:
			s = append(s, f)
			copy(s[1:], s[0:])
			s[0] = f
			p.push(s)
			return nil
		}
	}
	panic("Unreachable")
}

func (p *parser) resolveUnOp(t tok) error {
	expr, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve expr of unop %c at %d.", t.v, t.l)
	}
	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse UnOp OpType %v", t)
	}
	o := UnaryOp{
		Op: op,
		A:  expr,
	}
	p.push(o)
	return nil
}

func (p *parser) resolveBinOp(t tok) error {
	snd, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Unable to resolve snd of BinOp at %d as Expr.", t.l)
	}
	fst, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Unable to resolve fst of BinOp at %d as Expr.", t.l)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse BinOp OpType %v", t)
	}

	o := BinOp{
		Op: op,
		A:  fst,
		B:  snd,
	}
	p.push(o)
	return nil
}

func (p *parser) resolveCmpOp(t tok) error {
	snd := p.pop()
	sndOp, ok := snd.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve snd of CmpOp %c at %d as Operation. Got %T instead ", t.v, t.l, snd)
	}
	fst := p.pop()
	fstOp, ok := fst.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve fst of CmpOp %c at %d as Operation.", t.v, t.l)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse CmpOp OpType %v", t)
	}

	o := SubjectTo{
		OpType: op,
		A:      fstOp,
		B:      sndOp,
	}
	p.push(o)
	return nil

}

func (p *parser) resolveLogOp(t tok) error {
	snd := p.pop()
	sndOp, ok := snd.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve snd of LogOp %c at %d as Operation. Got %T instead ", t.v, t.l, snd)
	}
	fst := p.pop()
	fstOp, ok := fst.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve fst of LogOp %c at %d as Operation. Got %v of %T instead", t.v, t.l, fst, fst)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse LogOp OpType %v", t)
	}

	o := SubjectTo{
		OpType: op,
		A:      fstOp,
		B:      sndOp,
	}
	p.push(o)
	return nil
}

// resolveCompound expects the stack to look like this:
// 	[..., Expr, SubjectTo{...}]
// The result will look like this
// 	[..., Compound{...}] (the Compound{} now has data)
func (p *parser) resolveCompound() error {
	// first check
	var st SubjectTo
	var e Expr
	var ok bool

	top := p.pop() // SubjectTo
	if st, ok = top.(SubjectTo); !ok {
		return errors.Errorf("Expected Top of Stack to be a SubjectTo is %v of %T. Stack: %v", top, top, p.stack)
	}
	snd := p.pop() // Expr
	if e, ok = snd.(Expr); !ok {
		return errors.Errorf("Expected Second of Stack to be a Expr is %v of %T. Stack: %v", snd, snd, p.stack)
	}
	c := Compound{
		Expr:      e,
		SubjectTo: st,
	}

	p.push(c)
	return nil
}

func (p *parser) resolveSlice() error {
	// five cases:
	// 1. single slice (e.g. a[0])
	// 2. range (e.g. a[0:2])
	// 3. stepped range (e.g. a[0:2:2])
	// 4. open range (e.g. a[1:]) CURRENTLY UNSUPPORTED. TODO.
	// 5. limit range (e.g. a[:2]) CURRENTLY UNSUPPORTED. TODO.

	// pop infixStack - this will handle any of the cases with colons.
	if err := p.resolveGroup('['); err != nil {
		return errors.Wrap(err, "Unable to resolveSlice.")
	}

	// resolve any potential SliceOf{} or IndexOf{}
	top := p.pop()

	// check for case 1
	var idxof bool
	var idx int
	var slice Sli
	switch t := top.(type) {
	case Sli:
		slice = t
	case Size:
		idx = int(t)
		idxof = true
	default:
		return errors.Errorf("top can either be Sli or Size. Got %v of %T instead", top, top)
	}

	snd := p.pop()
	so, ok := snd.(SliceOf)
	if !ok {
		p.push(snd)
		// check if idxof is true
		if idxof {
			// top should no longer just be an int
			top = Sli{start: idx, end: idx + 1, step: 1}
		}
		p.push(top)
		return nil
	}

	// if it's ok, then the third from the stack would be an Expr
	thd := p.pop().(Expr)
	if idxof {
		// then use IndexOf instead of SliceOf
		iof := IndexOf{I: Size(idx), A: thd}
		p.push(iof)
		return nil
	}
	so.Slice = slice
	so.A = thd
	p.push(so)
	return nil
}

func (p *parser) resolveColon() error {
	// five cases:
	// 1. single slice (e.g. a[0])
	// 2. range (e.g. a[0:2])
	// 3. stepped range (e.g. a[0:2:2])
	// 4. open range (e.g. a[1:]) CURRENTLY UNSUPPORTED. TODO.
	// 5. limit range (e.g. a[:2]) CURRENTLY UNSUPPORTED. TODO.

	// a colon is a binop
	snd := p.pop()
	fst := p.pop()

	s, ok := substToInt(snd)
	if !ok {
		return errors.Errorf("Expected the top to be a Size. Got %v of %T instead", snd, snd)
	}
	switch f := fst.(type) {
	case Size:
		// case 2
		retVal := Sli{start: int(f), end: s, step: 1}
		p.push(retVal)
	case Sli:
		// case 3
		f.step = s
		p.push(f)
	default:
		// case 4
		// NOT REALLY SUPPORTED. TODO
		retVal := Sli{start: int(s), end: int(s) + 1, step: 1}
		p.push(fst) // put it back
		p.push(retVal)
	}

	return nil
}

// operator precedence table
var opprec = map[rune]int{
	'(': 80,
	')': -1,
	'[': 1,
	']': 70,
	'{': 70,
	'}': -1,
	',': 2,
	':': 75,
	'|': -1,
	'→': 0,

	// unop
	'K': 60,
	'D': 60,
	'Π': 60,
	'Σ': 60,
	'∀': 60,

	// binop
	'+': 40,
	'-': 40,
	'×': 50,
	'÷': 50,

	// cmpop
	'=': 30,
	'≠': 30,
	'<': 30,
	'>': 30,
	'≤': 30,
	'≥': 30,

	// logop
	'∧': 20,
	'∨': 10,
}

type tokentype int

const (
	eos tokentype = iota
	parenL
	parenR
	brackL
	brackR
	braceL
	braceR
	digit
	letter
	comma
	arrow
	colon
	pipe
	unop
	binop
	cmpop
	logop
)

type tok struct {
	t tokentype // type
	v rune      // value of the token
	l int       // location
}

func (t tok) Format(s fmt.State, c rune) {
	switch t.t {
	case letter:
		fmt.Fprintf(s, "{%c %d}", t.v, t.l)
	case digit:
		fmt.Fprintf(s, "{%d %d}", t.v, t.l)
	case eos:
		fmt.Fprintf(s, "{EOS %d}", t.l)
	default:
		fmt.Fprintf(s, "{%c %d}", t.v, t.l)
	}
}

const eoserr = "Lex Error: sudden end of string found in %q. Position %d"

// lex is a function that takes a string and returns a slice of tokens.
//
// it's fundamentally a giant state table in a for loop. Since the grammar of the language is very strict, it is a fairly straight forwards parse.
func lex(a string) (retVal []tok, err error) {
	rs := []rune(a)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '(':
			retVal = append(retVal, tok{parenL, r, i})
		case r == ')':
			retVal = append(retVal, tok{parenR, r, i})
		case r == '[':
			retVal = append(retVal, tok{brackL, r, i})
		case r == ']':
			retVal = append(retVal, tok{brackR, r, i})
		case r == '{':
			retVal = append(retVal, tok{braceL, r, i})
		case r == '}':
			retVal = append(retVal, tok{braceR, r, i})
		case r == '→':
			retVal = append(retVal, tok{arrow, r, i})
		case r == '-':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '>' {
				i++
				retVal = append(retVal, tok{arrow, '→', i})
				continue
			}
			retVal = append(retVal, tok{binop, r, i})
		case r == '+', r == '*', r == '×', r == '/', r == '∕', r == '÷':
			rr := r
			if r == '*' {
				rr = '×'
			}
			if r == '/' || r == '∕' {
				rr = '÷'
			}
			retVal = append(retVal, tok{binop, rr, i})
		case r == '=', r == '≠', r == '≥', r == '≤', r == '≥', r == '≤': // single symbol cmp op (note there are TWO acceptable unicode symbols for gte and lte)
			retVal = append(retVal, tok{cmpop, r, i})
		case r == '!':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '=' {
				i++
				retVal = append(retVal, tok{cmpop, '≠', i})
			}
		case r == '>', r == '<':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '=' {
				i++
				var rr rune
				switch r {
				case '>':
					rr = '≥'
				case '<':
					rr = '≤'
				}
				retVal = append(retVal, tok{cmpop, rr, i})
				continue
			}
			retVal = append(retVal, tok{cmpop, r, i})
		case r == '∧', r == '∨', r == '⋀', r == '⋁': // single symbol logical op
			retVal = append(retVal, tok{logop, r, i})
		case r == '&':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '&' { // for people who think AND is written "&&"
				i++
				retVal = append(retVal, tok{logop, '∧', i})
				continue
			}
		case r == '|':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '|' { // for people who think OR is written "||"
				i++
				retVal = append(retVal, tok{logop, '∨', i})
				continue
			}
			retVal = append(retVal, tok{pipe, r, i})
		case r == 'Π', r == 'Σ', r == '∀', r == 'D', r == 'P', r == 'S':
			rr := r
			if r == 'P' {
				rr = 'Π'
			}
			if r == 'S' {
				rr = 'Σ'
			}
			retVal = append(retVal, tok{unop, rr, i})
		case r == ',':
			retVal = append(retVal, tok{comma, r, i})
		case r == ':':
			retVal = append(retVal, tok{colon, r, i})
		case unicode.IsSpace(r):
			continue // we ignore spaces and delimiters
		case unicode.IsDigit(r):
			var rs2 = []rune{r}
			for j := i + 1; j < len(rs); j++ {
				r2 := rs[j]
				if !unicode.IsDigit(r2) {
					i = j - 1
					break
				}
				rs2 = append(rs2, r2)
			}
			s := string(rs2)
			num, err := strconv.Atoi(s)
			if err != nil {
				return nil, errors.Wrapf(err, "Unable to parse %q as a number", s)
			}

			retVal = append(retVal, tok{digit, rune(int32(num)), i})
		case unicode.IsLetter(r):
			retVal = append(retVal, tok{letter, r, i})
			// check if next is a letter. if it is  then error out
			if i+1 >= len(rs) {
				return retVal, nil
			}
			if unicode.IsLetter(rs[i+1]) {
				return nil, errors.Errorf("Only single letters are allowed as variables.")
			}

		}

	}
	return retVal, nil
}
