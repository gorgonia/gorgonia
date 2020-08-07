package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// ParseConfiguration Parse darknet configuration file
func ParseConfiguration(fname string) ([]map[string]string, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	lines := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := scanner.Text()
		if len(text) < 1 {
			continue
		}
		if text[0] == '#' {
			continue
		}
		lines = append(lines, strings.TrimSpace(text))
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	block := make(map[string]string)
	blocks := []map[string]string{}
	for i := range lines {
		if lines[i][0] == '[' {
			if len(block) != 0 {
				blocks = append(blocks, block)
				block = make(map[string]string)
			}
			block["type"] = lines[i][1 : len(lines[i])-1]
		} else {
			kv := strings.Split(lines[i], "=")
			if len(kv) != 2 {
				return nil, fmt.Errorf("Wrong format of layer parameters: %s", lines[i])
			}
			key, value := strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1])
			block[key] = value
		}
	}
	blocks = append(blocks, block)
	return blocks, nil
}

// ParseWeights Parse darknet weights
func ParseWeights(fname string) ([]float32, error) {
	fp, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer fp.Close()
	summary := []byte{}
	data := make([]byte, 4096)
	for {
		data = data[:cap(data)]
		n, err := fp.Read(data)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		data = data[:n]
		summary = append(summary, data...)
	}
	dataF32 := []float32{}
	for i := 0; i < len(summary); i += 4 {
		tempSlice := summary[i : i+4]
		tempFloat32 := Float32frombytes(tempSlice)
		dataF32 = append(dataF32, tempFloat32)
	}
	return dataF32, nil
}
