package geocolor

import (
	"fmt"
	"log"
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"

	_ "github.com/gomlx/gomlx/backends/default"
)

type ColorSpace struct {
	ctx         *context.Context
	backend     backends.Backend
	colorsLCH   *tensors.Tensor
	colorsRGB   *tensors.Tensor
	neutralAxis *tensors.Tensor

	grid   *GridConfig
	cParam float64
}

func NewColorSpace(grid *GridConfig, c float64) *ColorSpace {
	return &ColorSpace{
		ctx:     context.New(),
		backend: backends.MustNew(),
		grid:    grid,
		cParam:  c,
	}
}

func (cs *ColorSpace) BuildPipeline(ctx *context.Context, g *graph.Graph) []*graph.Node {
	coords := BuildCoordinateGridGraph(g, cs.grid)
	localMetrics := ComputeLocalMetric(g, coords)
	distances := SolveGlobalDistances(g, localMetrics, cs.grid.OriginIndex)

	cParam := graph.Const(g, cs.cParam)
	dampened := ApplyDiminishingReturns(g, distances, cParam)

	neutralIndices := ExtractNeutralIndices(g, dampened)
	colorsLCH := ComputeGeometricColor(g, coords, neutralIndices, cs.grid.MinBounds, cs.grid.MaxBounds)
	colorsRGB := ConvertToRGB(g, colorsLCH)
	neutralAxis := OptimalNeutralAxis(g, colorsLCH)

	// Concatenate the two 4D tensors along the last dimension (the color channels)
	return []*graph.Node{colorsLCH, colorsRGB, neutralAxis}
}

func (cs *ColorSpace) Compute() (err error) {
	e, err := context.NewExec(cs.backend, cs.ctx, cs.BuildPipeline)
	if err != nil {
		return fmt.Errorf("Failed to compile GoMLX pipeline: %v\n", err)
	}

	results, err := e.Exec()
	if err != nil {
		return fmt.Errorf("Failed to execute GoMLX pipeline: %v\n", err)
	}

	// 4. Extract the 4D float64 tensor
	cs.colorsLCH = results[0]
	cs.colorsRGB = results[1]
	cs.neutralAxis = results[2]

	return nil
}

func (cs *ColorSpace) GetLCHColors() [][][][]float64 {
	return cs.colorsLCH.Value().([][][][]float64)
}

func (cs *ColorSpace) GetRGBColors() [][][][]float64 {
	return cs.colorsRGB.Value().([][][][]float64)
}

func (cs *ColorSpace) GetNeutralAxis() (neutral_L []float64, neutral_C []float64) {
	// Shape is [ResL][2]float64
	neutralVals := cs.neutralAxis.Value().([][]float64)

	neutral_L = make([]float64, cs.grid.ResL)
	neutral_C = make([]float64, cs.grid.ResL)

	for i := 0; i < cs.grid.ResL; i++ {
		neutral_L[i] = float64(neutralVals[i][0])
		neutral_C[i] = float64(neutralVals[i][1])
	}

	return neutral_L, neutral_C
}

func (cs *ColorSpace) TraceGeodesic(targetIndex [3]int) (path_L []float64, path_C []float64) {
	// Safely assert as a 4D slice
	distVals := cs.colorsLCH.Value().([][][][]float64)

	current := targetIndex
	origin := cs.grid.OriginIndex

	// Helper to convert discrete indices back to continuous coordinates
	getContinuous := func(idx [3]int) (float64, float64) {
		l := cs.grid.MinBounds[0] + float64(idx[0])*(cs.grid.MaxBounds[0]-cs.grid.MinBounds[0])/float64(cs.grid.ResL-1)
		a := cs.grid.MinBounds[1] + float64(idx[1])*(cs.grid.MaxBounds[1]-cs.grid.MinBounds[1])/float64(cs.grid.ResA-1)
		b := cs.grid.MinBounds[2] + float64(idx[2])*(cs.grid.MaxBounds[2]-cs.grid.MinBounds[2])/float64(cs.grid.ResB-1)
		c := math.Sqrt(a*a + b*b)
		return l, c
	}

	// Trace the steepest descent path
	for {
		l, c := getContinuous(current)
		path_L = append(path_L, l)
		path_C = append(path_C, c)

		// 1. Break if we explicitly hit the exact origin
		if current == origin {
			break
		}

		// 2. Break if we hit the L=0 floor (The Black Singularity)
		// current[0] is the Lightness index. If it hits 0, we are on the floor.
		if current[0] == 0 {
			// Manually append the true (0,0) origin to complete the visual curve
			path_L = append(path_L, 0.0)
			path_C = append(path_C, 0.0)
			break
		}
		minDist := float64(math.Inf(1))
		bestNeighbor := current

		// Search 3D neighborhood
		for dl := -1; dl <= 1; dl++ {
			for da := -1; da <= 1; da++ {
				for db := -1; db <= 1; db++ {
					if dl == 0 && da == 0 && db == 0 {
						continue
					}

					nl, na, nb := current[0]+dl, current[1]+da, current[2]+db

					if nl >= 0 && nl < cs.grid.ResL &&
						na >= 0 && na < cs.grid.ResA &&
						nb >= 0 && nb < cs.grid.ResB {

						// Grab the [0] index from the 4th dimension
						d := distVals[nl][na][nb][0]
						if d < minDist {
							minDist = d
							bestNeighbor = [3]int{nl, na, nb}
						}
					}
				}
			}
		}

		// Prevent infinite loops if caught in a local minimum well
		if distVals[bestNeighbor[0]][bestNeighbor[1]][bestNeighbor[2]][0] >= distVals[current[0]][current[1]][current[2]][0] {
			log.Println("Hit a local minimum, terminating trace early.")
			break
		}

		current = bestNeighbor
	}

	return path_L, path_C
}
