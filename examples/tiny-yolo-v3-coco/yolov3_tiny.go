package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"

	"gorgonia.org/tensor"
)

// YOLOv3 YOLOv3 architecture
type YOLOv3 struct {
	g                                 *gorgonia.ExprGraph
	classesNum, boxesPerCell, netSize int
	out                               []*gorgonia.Node
	layersInfo                        []string
}

// Print Print architecture of network
func (net *YOLOv3) Print() {
	for i := range net.layersInfo {
		fmt.Println(net.layersInfo[i])
	}
}

// GetOutput Get out YOLO layers (can be multiple of them)
func (net *YOLOv3) GetOutput() []*gorgonia.Node {
	return net.out
}

// NewYoloV3Tiny Create new tiny YOLO v3
func NewYoloV3Tiny(g *gorgonia.ExprGraph, input *gorgonia.Node, classesNumber, boxesPerCell int, leakyCoef float64, cfgFile, weightsFile string) (*YOLOv3, error) {
	inputS := input.Shape()
	if len(inputS) < 4 {
		return nil, fmt.Errorf("Input for YOLOv3 should contain infromation about 4 dimensions")
	}

	buildingBlocks, err := ParseConfiguration(cfgFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet configuration")
	}

	netParams := buildingBlocks[0]
	netWidthStr := netParams["width"]
	netWidth, err := strconv.Atoi(netWidthStr)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("Network's width must be integer, got value: '%s'", netWidthStr))
	}

	weightsData, err := ParseWeights(weightsFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet weights")
	}

	fmt.Println("Loading network...")
	layers := []*layerN{}
	outputFilters := []int{}
	prevFilters := 3

	networkNodes := []*gorgonia.Node{}

	blocks := buildingBlocks[1:]
	lastIdx := 5 // Skip first 5 values (header of weights file)
	epsilon := float32(0.000001)

	yoloNodes := []*gorgonia.Node{}
	for i := range blocks {
		block := blocks[i]
		filtersIdx := 0
		layerType, ok := block["type"]
		if ok {
			switch layerType {
			case "convolutional":
				filters := 0
				padding := 0
				kernelSize := 0
				stride := 0
				batchNormalize := 0
				bias := false
				activation := "activation"
				activation, ok := block["activation"]
				if !ok {
					fmt.Printf("No field 'activation' for convolution layer")
					continue
				}
				batchNormalizeStr, ok := block["batch_normalize"]
				batchNormalize, err := strconv.Atoi(batchNormalizeStr)
				if !ok || err != nil {
					batchNormalize = 0
					bias = true
				}
				filtersStr, ok := block["filters"]
				filters, err = strconv.Atoi(filtersStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'filters' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				paddingStr, ok := block["pad"]
				padding, err = strconv.Atoi(paddingStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'pad' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				kernelSizeStr, ok := block["size"]
				kernelSize, err = strconv.Atoi(kernelSizeStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'size' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				pad := 0
				if padding != 0 {
					pad = (kernelSize - 1) / 2
				}
				strideStr, ok := block["stride"]
				stride, err = strconv.Atoi(strideStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for convolution layer: %s\n", err.Error())
					continue
				}

				ll := &convLayer{
					filters:            filters,
					padding:            pad,
					kernelSize:         kernelSize,
					stride:             stride,
					activation:         activation,
					activationReLUCoef: leakyCoef,
					batchNormalize:     batchNormalize,
					bias:               bias,
				}

				shp := tensor.Shape{filters, prevFilters, kernelSize, kernelSize}
				kernels := []float32{}
				biases := []float32{}
				if ll.batchNormalize > 0 {
					nb := shp[0]
					nk := shp.TotalSize()

					biases = weightsData[lastIdx : lastIdx+nb]
					lastIdx += nb

					gammas := weightsData[lastIdx : lastIdx+nb]
					lastIdx += nb

					means := weightsData[lastIdx : lastIdx+nb]
					lastIdx += nb

					vars := weightsData[lastIdx : lastIdx+nb]
					lastIdx += nb

					kernels = weightsData[lastIdx : lastIdx+nk]
					lastIdx += nk

					// Denormalize weights
					for s := 0; s < shp[0]; s++ {
						scale := gammas[s] / math32.Sqrt(vars[s]+epsilon)
						biases[s] = biases[s] - means[s]*scale
						isize := shp[1] * shp[2] * shp[3]
						for j := 0; j < isize; j++ {
							kernels[isize*s+j] *= scale
						}
					}
				} else {
					if ll.bias {
						nb := shp[0]
						nk := shp.TotalSize()
						biases = weightsData[lastIdx : lastIdx+nb]
						lastIdx += nb
						kernels = weightsData[lastIdx : lastIdx+nk]
						lastIdx += nk
					}
				}

				convTensor := tensor.New(tensor.WithBacking(kernels), tensor.WithShape(shp...))
				convNode := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(shp...), gorgonia.WithName(fmt.Sprintf("conv_%d", i)), gorgonia.WithValue(convTensor))
				ll.convNode = convNode
				ll.biases = biases
				ll.layerIndex = i

				var l layerN = ll
				convBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Convolutional block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, convBlock)
				input = convBlock

				layers = append(layers, &l)

				filtersIdx = filters
				break
			case "upsample":
				scale := 0
				scaleStr, ok := block["stride"]
				scale, err = strconv.Atoi(scaleStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for upsampling layer: %s\n", err.Error())
					continue
				}

				var l layerN = &upsampleLayer{
					scale: scale,
				}

				upsampleBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Upsample block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, upsampleBlock)
				input = upsampleBlock

				layers = append(layers, &l)

				filtersIdx = prevFilters
				break
			case "route":
				routeLayersStr, ok := block["layers"]
				if !ok {
					fmt.Printf("No field 'layers' for route layer")
					continue
				}
				layersSplit := strings.Split(routeLayersStr, ",")
				if len(layersSplit) < 1 {
					fmt.Printf("Something wrong with route layer. Check if it has one array item atleast")
					continue
				}
				for l := range layersSplit {
					layersSplit[l] = strings.TrimSpace(layersSplit[l])
				}

				start := 0
				end := 0
				start, err := strconv.Atoi(layersSplit[0])
				if err != nil {
					fmt.Printf("Each first element of 'layers' parameter for route layer should be an integer: %s\n", err.Error())
					continue
				}
				if len(layersSplit) > 1 {
					end, err = strconv.Atoi(layersSplit[1])
					if err != nil {
						fmt.Printf("Each second element of 'layers' parameter for route layer should be an integer: %s\n", err.Error())
						continue
					}
				}

				if start > 0 {
					start = start - i
				}
				if end > 0 {
					end = end - i
				}

				l := routeLayer{
					firstLayerIdx:  i + start,
					secondLayerIdx: -1,
				}

				if end < 0 {
					l.secondLayerIdx = i + end
					filtersIdx = outputFilters[i+start] + outputFilters[i+end]
				} else {
					filtersIdx = outputFilters[i+start]
				}

				var ll layerN = &l

				routeBlock, err := l.ToNode(g, networkNodes...)
				if err != nil {
					fmt.Printf("\tError preparing Route block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, routeBlock)
				input = routeBlock

				layers = append(layers, &ll)

				break
			case "yolo":
				maskStr, ok := block["mask"]
				if !ok {
					fmt.Printf("No field 'mask' for YOLO layer")
					continue
				}
				maskSplit := strings.Split(maskStr, ",")
				if len(maskSplit) < 1 {
					fmt.Printf("Something wrong with yolo layer. Check if it has one item in 'mask' array atleast")
					continue
				}
				masks := make([]int, len(maskSplit))
				for l := range maskSplit {
					maskSplit[l] = strings.TrimSpace(maskSplit[l])
					masks[l], err = strconv.Atoi(maskSplit[l])
					if err != nil {
						fmt.Printf("Each element of 'mask' parameter for yolo layer should be an integer: %s\n", err.Error())
					}
				}
				anchorsStr, ok := block["anchors"]
				if !ok {
					fmt.Printf("No field 'anchors' for YOLO layer")
					continue
				}
				anchorsSplit := strings.Split(anchorsStr, ",")
				if len(anchorsSplit) < 1 {
					fmt.Printf("Something wrong with yolo layer. Check if it has one item in 'anchors' array atleast")
					continue
				}
				if len(anchorsSplit)%2 != 0 {
					fmt.Printf("Number of elemnts in 'anchors' parameter for yolo layer should be divided exactly by 2 (even number)")
					continue
				}
				anchors := make([]int, len(anchorsSplit))
				for l := range anchorsSplit {
					anchorsSplit[l] = strings.TrimSpace(anchorsSplit[l])
					anchors[l], err = strconv.Atoi(anchorsSplit[l])
					if err != nil {
						fmt.Printf("Each element of 'anchors' parameter for yolo layer should be an integer: %s\n", err.Error())
					}
				}
				anchorsPairs := [][2]int{}
				for a := 0; a < len(anchors); a += 2 {
					anchorsPairs = append(anchorsPairs, [2]int{anchors[a], anchors[a+1]})
				}
				selectedAnchors := [][2]int{}
				for m := range masks {
					selectedAnchors = append(selectedAnchors, anchorsPairs[masks[m]])
				}
				flatten := []int{}
				for a := range selectedAnchors {
					flatten = append(flatten, selectedAnchors[a][0])
					flatten = append(flatten, selectedAnchors[a][1])
				}

				ignoreThreshStr, ok := block["ignore_thresh"]
				if !ok {
					fmt.Printf("Warning: no field 'ignore_thresh' for YOLO layer")
				}
				ignoreThresh64, err := strconv.ParseFloat(ignoreThreshStr, 32)
				if !ok {
					fmt.Printf("Warning: can't cast 'ignore_thresh' to float32 for YOLO layer")
				}
				var l layerN = &yoloLayer{
					masks:          masks,
					anchors:        selectedAnchors,
					flattenAnchors: flatten,
					inputSize:      inputS[2],
					classesNum:     classesNumber,
					ignoreThresh:   float32(ignoreThresh64),
				}
				yoloBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing YOLO block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, yoloBlock)
				input = yoloBlock

				layers = append(layers, &l)
				yoloNodes = append(yoloNodes, yoloBlock)
				filtersIdx = prevFilters
				break
			case "maxpool":
				sizeStr, ok := block["size"]
				if !ok {
					fmt.Printf("No field 'size' for maxpooling layer")
					continue
				}
				size, err := strconv.Atoi(sizeStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}
				strideStr, ok := block["stride"]
				if !ok {
					fmt.Printf("No field 'stride' for maxpooling layer")
					continue
				}
				stride, err := strconv.Atoi(strideStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}

				var l layerN = &maxPoolingLayer{
					size:   size,
					stride: stride,
				}
				maxpoolingBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Max-Pooling block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, maxpoolingBlock)
				input = maxpoolingBlock
				layers = append(layers, &l)
				filtersIdx = prevFilters
				break
			default:
				fmt.Printf("Impossible layer: '%s'\n", layerType)
				break
			}
		}
		prevFilters = filtersIdx
		outputFilters = append(outputFilters, filtersIdx)
	}

	// Pretty print
	linfo := []string{}
	for i := range layers {
		linfo = append(linfo, fmt.Sprintf("%v", *layers[i]))
	}

	model := &YOLOv3{
		classesNum:   classesNumber,
		boxesPerCell: boxesPerCell,
		netSize:      netWidth,
		out:          yoloNodes,
		layersInfo:   linfo,
	}

	return model, nil
}
