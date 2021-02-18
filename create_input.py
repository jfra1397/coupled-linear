### Parameter declarations

# numerical parameters
alphaNM = 1.0 / 4.0 
betaNM =  1.0 / 2.0
residualPrecision = 1E-10

# material parameters
nu = 0.3                                        # Poisson's ratio
emod = 210000.0                                 # Young's modulus
llambda = emod*nu/((1.0-2.0*nu)*(1.0+nu))       # 1st Lamé constant
lmu = emod/(2.0 + 2.0*nu)                       # 2nd Lamé constant
alphaS = 0.0001                                 # thermal expansion coefficient
kappa = 2.0 / 3.0 * lmu + llambda               # compression modulus
alphaT = 35.0                                   # thermal conductivity
cp = 480000000.0                                # heat capacity, constant pressure
rho = 0.0000000075                              # initial density
rT = 0.0                                        # external heat source

# problem instance
length = 100.0                                  # dimensions of the cuboid 
width = 80.0          
height = 140.0
theta0 = 300.0                                  # initial temperature
initValues = [0.0, 0.0, 0.0, theta0]            # initial displacements and temperature
numberOfElements = 8
numberOfNodes = 27
dofPerNode = 4
nodesPerElement = 8
numberOfGaussPoints = 8
simulationTime = 500      # load steps
timeStep = 1              # step size
maxNewtonIterations = 50
#load = -1000000.0


### boundary conditions

# node : [list, of, dofs]
fixedDofs = {\
        0 : [0, 1, 2], \
        1 : [1, 2], \
        2 : [1, 2], \
        3 : [0, 2], \
        4 : [2], \
        5 : [2], \
        6 : [0, 2], \
        7 : [2], \
        8 : [2]}

# forces
# node: [list, of, forces per dof]
heatFlux = 0#-50000000
fz = 8000000
externalForces = {\
        18 : [0, 0, fz, heatFlux / 4 / 4], \
        19 : [0, 0, fz, heatFlux / 4 / 2], \
        20 : [0, 0, fz, heatFlux / 4 / 4], \
        21 : [0, 0, fz, heatFlux / 4 / 2], \
        22 : [0, 0, fz, heatFlux / 4 / 1], \
        23 : [0, 0, fz, heatFlux / 4 / 2], \
        24 : [0, 0, fz, heatFlux / 4 / 4], \
        25 : [0, 0, fz, heatFlux / 4 / 2], \
        26 : [0, 0, fz, heatFlux / 4 / 4]}  

externalGradient = [[0,10, 0.1, 0],\
                [10,20,-0.1, 1],\
]

endHeatFlux = 100



#############################################################never mind##############################################################
parameters = {"alphaNM" : alphaNM,
              "betaNM": betaNM,
              "residualPrecision": residualPrecision,
              "nu": nu,
              "emod": emod,
              "llambda": llambda,
              "lmu": lmu,
              "alphaS": alphaS,
              "kappa": kappa,
              "alphaT": alphaT,
              "cp": cp,
              "rho": rho,
              "rT": rT,
              "length": length,
              "width": width,
              "height": height,
              "theta0": theta0,
              "numberOfElements": numberOfElements,
              "numberOfNodes": numberOfNodes,
              "dofPerNode":dofPerNode,
              "nodesPerElement": nodesPerElement,
              "numberOfGaussPoints": numberOfGaussPoints,
              "simulationTime": simulationTime,
              "timeStep": timeStep,
              "maxNewtonIterations": maxNewtonIterations,
              "initValues":  initValues,
              "fixedDofs":  fixedDofs,
              "externalForces": externalForces,
              "externalGradient": externalGradient,
              "endHeatFlux": endHeatFlux
}


import sys, json 
with open(sys.argv[1], 'w') as f:
    json.dump(parameters, f)


