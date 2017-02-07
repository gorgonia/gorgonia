package main

import "strings"

const START rune = 0x02
const END rune = 0x03

// vocab related
var sentences []string
var vocab []rune
var vocabIndex map[rune]int

func initVocab(ss []string, thresh int) {
	s := strings.Join(ss, "")
	dict := make(map[rune]int)
	for _, r := range s {
		dict[r]++
	}

	vocab = append(vocab, START)
	vocabIndex = make(map[rune]int)

	for ch, c := range dict {
		if c >= thresh {
			// then add letter to vocab
			vocab = append(vocab, ch)
		}
	}

	vocab = append(vocab, END)

	for i, v := range vocab {
		vocabIndex[v] = i
	}

	inputSize = len(vocab)
	outputSize = len(vocab)
	epochSize = len(ss)
}

func init() {
	sentencesRaw := strings.Split(corpus, "\n")
	for _, s := range sentencesRaw {
		s2 := strings.TrimSpace(s)
		if s2 != "" {
			sentences = append(sentences, s2)
		}
	}

	initVocab(sentences, 1)
}
