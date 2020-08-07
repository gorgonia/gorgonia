package gorgonia

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {
	target, _ := prepareTrain32("./examples/tiny-yolo-v3-coco/data", 52)
	input := tensor.New(tensor.Of(tensor.Float32))

	r, err := os.Open("./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(10, 13), (16, 30), (33, 23)].npy")
	if err != nil {
		t.Error(err)
		return
	}
	input.ReadNpy(r)
	output := tensor.New(tensor.Of(tensor.Float32))
	r, err = os.Open("./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(10, 13), (16, 30), (33, 23)].npy")
	if err != nil {
		t.Error(err)
		return
	}
	output.ReadNpy(r)

	g := NewGraph()
	inp := NewTensor(g, tensor.Float32, 4, WithShape(input.Shape()...), WithName("inp"))
	inp2 := NewTensor(g, tensor.Float32, 4, WithShape(target.Shape()...), WithName("inp2"))
	inp3 := NewTensor(g, tensor.Float32, 3, WithShape(output.Shape()...), WithName("inp3"))

	yoloOut, err := YOLOv3(inp, []float32{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319}, []int{0, 1, 2}, 416, 80, 0.5, inp2)
	if err != nil {
		t.Error(err)
		return
	}

	vm := NewTapeMachine(g)
	err = Let(inp, input)
	if err != nil {
		t.Error(err)
		return
	}
	err = Let(inp2, target)
	if err != nil {
		t.Error(err)
		return
	}
	err = Let(inp3, output)
	if err != nil {
		t.Error(err)
		return
	}
	err = vm.RunAll()
	if err != nil {
		t.Error(err)
		return
	}

	vm.Close()
	if !assert.Equal(t, yoloOut.Shape(), output.Shape(), "Output shape is not equal to expected shape value") {
		t.Error(fmt.Sprintf("Got: %v\nExpected: %v", yoloOut.Shape(), output.Shape()))
	}

}

func prepareTrain32(pathToDir string, gridSize int) (*tensor.Dense, error) {
	files, err := ioutil.ReadDir(pathToDir)
	if err != nil {
		return &tensor.Dense{}, err
	}
	farr := [][]float32{}
	maxLen := gridSize * gridSize
	numTrainFiles := 0
	for _, file := range files {
		cfarr := []float32{}
		if file.IsDir() || filepath.Ext(file.Name()) != ".txt" {
			continue
		}
		numTrainFiles++
		f, err := ioutil.ReadFile(pathToDir + "/" + file.Name())
		if err != nil {
			return &tensor.Dense{}, err
		}
		str := string(f)
		fmt.Println(str)
		str = strings.ReplaceAll(str, "\n", " ")
		arr := strings.Split(str, " ")
		for i := 0; i < len(arr); i++ {
			if s, err := strconv.ParseFloat(arr[i], 32); err == nil {
				if float32(s) < 0 {
					return &tensor.Dense{}, fmt.Errorf("Incorrect training data")
				}
				cfarr = append(cfarr, float32(s))
			} else {
				return &tensor.Dense{}, err
			}
		}
		farr = append(farr, cfarr)
	}
	backArr := []float32{}
	for i := 0; i < len(farr); i++ {
		backArr = append(backArr, float32(len(farr[i])))
		backArr = append(backArr, farr[i]...)
		if len(farr[i]) < maxLen {
			zeroes := make([]float32, maxLen-len(farr[i])-1)
			backArr = append(backArr, zeroes...)
		}
	}
	return tensor.New(tensor.WithShape(numTrainFiles, 1, gridSize, gridSize), tensor.Of(tensor.Float32), tensor.WithBacking(backArr)), nil
}
