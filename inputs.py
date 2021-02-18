import json

### Parameter declarations
import sys
inputfilePath = sys.argv[1]
f = open(inputfilePath)
parameters = json.load(f)
f.close()

for key in list(parameters["fixedDofs"].keys()):
    parameters["fixedDofs"][int(key)] = parameters["fixedDofs"].pop(key)

for key in list(parameters["externalForces"].keys()):
    parameters["externalForces"][int(key)] = parameters["externalForces"].pop(key)

# numerical parameters
alphaNM = parameters["alphaNM"]
betaNM =  parameters["betaNM"]
residualPrecision = parameters["residualPrecision"]

# material parameters
nu = parameters["nu"]                                        # Poisson's ratio
emod = parameters["emod"]                                 # Young's modulus
llambda = parameters["llambda"]       # 1st Lamé constant
lmu = parameters["lmu"]                       # 2nd Lamé constant
alphaS = parameters["alphaS"]                                 # thermal expansion coefficient
kappa = parameters["kappa"]               # compression modulus
alphaT = parameters["alphaT"]                                   # thermal conductivity
cp = parameters["cp"]                                # heat capacity, constant pressure
rho = parameters["rho"]                              # initial density
rT = parameters["rT"]                                        # external heat source

# problem instance
length = parameters["length"]                                  # dimensions of the cuboid
width = parameters["width"]
height = parameters["height"]
theta0 = parameters["theta0"]                                  # initial temperature
initValues = parameters["initValues"]            # initial displacements and temperature
numberOfElements = parameters["numberOfElements"]
numberOfNodes = parameters["numberOfNodes"]
dofPerNode = parameters["dofPerNode"]
nodesPerElement = parameters["nodesPerElement"]
numberOfGaussPoints = parameters["numberOfGaussPoints"]
simulationTime = parameters["simulationTime"]      # load steps
timeStep = parameters["timeStep"]              # step size
maxNewtonIterations = parameters["maxNewtonIterations"]
#load = -1000000.0

### boundary conditions

# node : [list, of, dofs]
fixedDofs = parameters["fixedDofs"]
# forces
# node: [list, of, forces per dof]
#heatFlux = parameters["heatFlux"]
#fz = -10000
externalForces = parameters["externalForces"]
externalGradient = parameters["externalGradient"]
endHeatFlux = parameters["endHeatFlux"]