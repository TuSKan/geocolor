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
	ctx      *context.Context
	backend  backends.Backend
	pipeline func(ctx *context.Context, g *graph.Graph) *graph.Node
	results  *tensors.Tensor

	grid   *GridConfig
	cParam float32
}

func NewColorSpace(grid *GridConfig, c float32) *ColorSpace {
	return &ColorSpace{
		ctx:     context.New(),
		backend: backends.MustNew(),
		grid:    grid,
		cParam:  c,
		pipeline: func(ctx *context.Context, g *graph.Graph) *graph.Node {
			coords := BuildCoordinateGridGraph(g, grid)
			localMetrics := ComputeLocalMetric(g, coords)
			distances := SolveGlobalDistances(g, localMetrics, grid.OriginIndex)

			cParam := graph.Const(g, c)
			dampened := ApplyDiminishingReturns(g, distances, cParam)

			neutralIndices := ExtractNeutralAxis(g, dampened)
			return ComputeGeometricColor(g, coords, neutralIndices, grid.MinBounds, grid.MaxBounds)
		},
	}
}

func (cs *ColorSpace) Compute() (err error) {
	e, err := context.NewExec(cs.backend, cs.ctx, cs.pipeline)
	if err != nil {
		return fmt.Errorf("Failed to compile GoMLX pipeline: %v\n", err)
	}

	results, err := e.Exec()
	if err != nil {
		return fmt.Errorf("Failed to execute GoMLX pipeline: %v\n", err)
	}

	// 4. Extract the 4D float32 tensor
	cs.results = results[0]

	return nil
}

func (cs *ColorSpace) GetColors() [][][][]float32 {
	return cs.results.Value().([][][][]float32)
}

func (cs *ColorSpace) GetNeutralAxis() (neutral_L []float64, neutral_C []float64) {
	vals := cs.results.Value().([][][][]float32)

	// 5. Process the Neutral Axis
	neutral_L = make([]float64, cs.grid.ResL)
	neutral_C = make([]float64, cs.grid.ResL)

	for L := 0; L < cs.grid.ResL; L++ {
		minC := float32(math.Inf(1))
		var targetL float32

		for a := 0; a < cs.grid.ResA; a++ {
			for b := 0; b < cs.grid.ResB; b++ {
				lVal := vals[L][a][b][0]
				cVal := vals[L][a][b][1]

				if cVal < minC {
					minC = cVal
					targetL = lVal
				}
			}
		}

		neutral_L[L] = float64(targetL)
		neutral_C[L] = float64(minC)
	}

	return neutral_L, neutral_C
}

func (cs *ColorSpace) TraceGeodesic(targetIndex [3]int) (path_L []float64, path_C []float64) {
	// Safely assert as a 4D slice
	distVals := cs.results.Value().([][][][]float32)

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
		minDist := float32(math.Inf(1))
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
