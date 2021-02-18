# element class 

from node import Node
from numpy import linspace, array, zeros, ones, matmul, insert, delete, round, count_nonzero, float64
from numpy.linalg import inv, det
from math import sqrt
from numba import njit


# this is calculated externally because numba cannot completely support nested custom classes currently
@njit
def computeStiffness(fv, fa, stiffness, numberOfGaussPoints, dofPerNode, numberOfNodes, bmatrix, mC, dVol, shapeFunctionValues, balanceOfEnergy, balanceOfEnergypTheta, balanceOfEnergypThetaDot, balanceOfEnergypEpsDot, sigmapTheta, qlinpGradTheta):
    for gp in range(0, numberOfGaussPoints):
        for node1 in range(0, numberOfNodes):
            #mechanical part K_uu
            for node2 in range(0, numberOfNodes):
                for i in range(0, 3): 
                    for j in range(0, 3):
                        for k in range(0, 2*3):
                            for l in range(0, 2*3):
                                stiffness[node1*dofPerNode + i][node2*dofPerNode + j] \
                                                  += bmatrix[gp][node1][k][i] * mC[k][l] \
                                                   * bmatrix[gp][node2][l][j] * dVol[gp]

                # thermal part K_tt
                for i in range(0, 3):
                    for j in range(0, 3):
                        stiffness[node1*dofPerNode + 3][node2*dofPerNode + 3] \
                                          += shapeFunctionValues[gp][node1][i] * qlinpGradTheta[i][j] \
                                           * shapeFunctionValues[gp][node2][j] * dVol[gp]

                stiffness[node1*dofPerNode + 3][node2*dofPerNode + 3] \
                                  += shapeFunctionValues[gp][node1][3] * balanceOfEnergypTheta[gp] \
                                   * shapeFunctionValues[gp][node2][3] * dVol[gp]
                                   
                stiffness[node1*dofPerNode + 3][node2*dofPerNode + 3] \
                                  += shapeFunctionValues[gp][node1][3] * balanceOfEnergypThetaDot \
                                   * shapeFunctionValues[gp][node2][3] * fv * dVol[gp]
  
                # coupling part K_ut
                for i in range(0, 3):
                    for j in range(0, 2*3):
                        stiffness[node1*dofPerNode + i][node2*dofPerNode + 3] \
                                          += bmatrix[gp][node1][j][i] * sigmapTheta[gp][j] \
                                           * shapeFunctionValues[gp][node2][3] * dVol[gp]

                # coupling part K_tu
                for i in range(0, 3):
                    for j in range(0, 2*3):
                        stiffness[node1*dofPerNode + 3][node2*dofPerNode + i] \
                                          += shapeFunctionValues[gp][node1][3] * balanceOfEnergypEpsDot[gp][j] \
                                           * bmatrix[gp][node2][j][i] * fv * dVol[gp]
                                           

@njit(parallel=True)
def computeForces(forces, numberOfGaussPoints, dofPerNode, numberOfNodes, bmatrix, sigma, dVol, shapeFunctionValues, qlin, balanceOfEnergy):
    for gp in range(0, numberOfGaussPoints):
        for node in range(0, numberOfNodes):
            #mechanical part f_u
            for i in range(0, 3):
                for j in range(0, 2*3):
                    forces[node][i] -= bmatrix[gp][node][j][i] * sigma[gp][j] * dVol[gp]
                    
            # thermal part f_tt
            for i in range(0, 3):
                forces[node][3] -= shapeFunctionValues[gp][node][i] * qlin[gp][i] * dVol[gp]
            forces[node][3] -= shapeFunctionValues[gp][node][3] * balanceOfEnergy[gp] * dVol[gp]
    
 
 

class Element:
    def __init__(self, nodesPerElement, dofPerNode, numberOfGaussPoints, mC, theta0, lmu, llambda, \
                 alphaS, kappa, alphaT, cp, rho, rT): 
        self.numberOfNodes = nodesPerElement
        self.dofPerNode = dofPerNode
        self.nodes = []
        self.gradU = zeros((numberOfGaussPoints, 3, 3), dtype=float64)
        self.gradUDot = zeros((numberOfGaussPoints, 3, 3), dtype=float64)
        self.theta0 = array([theta0] * numberOfGaussPoints, dtype=float64)
        self.theta = zeros(numberOfGaussPoints, dtype=float64)
        self.thetaDot = zeros(numberOfGaussPoints, dtype=float64)
        self.gradTheta = zeros((numberOfGaussPoints, 3), dtype=float64)
        self.vonMises = zeros(numberOfGaussPoints, dtype=float64) 
        self.stiffness = zeros((nodesPerElement * dofPerNode, nodesPerElement * dofPerNode), dtype=float64)
        self.forces = zeros((nodesPerElement, dofPerNode), dtype=float64)
        self.numberOfGaussPoints = numberOfGaussPoints
        self.weights = ones(nodesPerElement, dtype=float64) # can be customized
        self.dVol = zeros(nodesPerElement, dtype=float64)
        self.shapeFunctionValues = zeros((numberOfGaussPoints, nodesPerElement, dofPerNode), dtype=float64) 
        self.bmatrix = zeros((numberOfGaussPoints, nodesPerElement, 2*3, 3), dtype=float64) 

        # material parameters
        self.lmu = lmu
        self.llambda = llambda
        self.alphaS = alphaS
        self.kappa = kappa
        self.alphaT = alphaT
        self.cp = cp
        self.rho = rho
        self.rT = rT

        # material fields 
        self.mC = mC
        self.epsilon =     zeros((numberOfGaussPoints, 6), dtype=float64)
        self.epsilonDot =  zeros((numberOfGaussPoints, 6), dtype=float64)
        self.traceEps =    zeros((numberOfGaussPoints), dtype=float64)
        self.traceEpsDot = zeros((numberOfGaussPoints), dtype=float64)
        self.sigma =       zeros((numberOfGaussPoints, 6), dtype=float64)
        self.qlin  =       zeros((numberOfGaussPoints, 3), dtype=float64)
        self.qlinpGradTheta = zeros((3, 3), dtype=float64)
        self.sigmapTheta = zeros((numberOfGaussPoints, 6), dtype=float64)
        self.balanceOfEnergy = zeros(numberOfGaussPoints, dtype=float64)
        self.balanceOfEnergypTheta = zeros(numberOfGaussPoints, dtype=float64) 
        self.balanceOfEnergypThetaDot = 0.0
        self.balanceOfEnergypEpsDot = zeros((numberOfGaussPoints, 6), dtype=float64)
        
        self.nodePositions = array([[-1.0, -1.0, -1.0],
                            [ 1.0, -1.0, -1.0],
                            [ 1.0,  1.0, -1.0],
                            [-1.0,  1.0, -1.0],
                            [-1.0, -1.0,  1.0],
                            [ 1.0, -1.0,  1.0],
                            [ 1.0,  1.0,  1.0],
                            [-1.0,  1.0,  1.0]], dtype=float64)

    
        self.gaussPoints = self.nodePositions / sqrt(3)
        
        
    def addNode(self, node):
        if len(self.nodes) < self.numberOfNodes:
            self.nodes.append(node)
        else:
            raise ValueError("Trying to assign more than 8 nodes to element")


    # position are the isoparametric coordinates of the node, 
    # point are the isoparametric coordinates where you want to evaluate the shape function
    def shapeFunction(self, position, point):
        return 1.0/8.0 * (1.0 + position[0]*point[0]) * (1.0 + position[1]*point[1]) * (1.0 + position[2]*point[2])

    # derivatives of shape function:
    def shapeFunctionX(self, position, point):
        return 1.0/8.0 * position[0] * (1.0 + position[1]*point[1]) * (1.0 + position[2]*point[2])

    def shapeFunctionY(self, position, point):
        return 1.0/8.0 * position[1] * (1.0 + position[2]*point[2]) * (1.0 + position[0]*point[0])

    def shapeFunctionZ(self, position, point):
        return 1.0/8.0 * position[2] * (1.0 + position[0]*point[0]) * (1.0 + position[1]*point[1])
    
    
    def resetStresses(self):
        #self.mC = zeros((self.numberOfGaussPoints, 6, 6))  ### uncomment if mC isn't constant
        self.epsilon =     zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.epsilonDot =  zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.traceEps =    zeros((self.numberOfGaussPoints), dtype=float64)
        self.traceEpsDot = zeros((self.numberOfGaussPoints), dtype=float64)
        self.sigma =       zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.vonMises =    zeros(self.numberOfGaussPoints, dtype=float64) 
        self.qlin  =       zeros((self.numberOfGaussPoints, 3), dtype=float64)
        self.qlinpGradTheta = zeros((3, 3), dtype=float64)
        self.sigmapTheta = zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.balanceOfEnergy = zeros(self.numberOfGaussPoints, dtype=float64)
        self.balanceOfEnergypTheta = zeros(self.numberOfGaussPoints, dtype=float64) 
        self.balanceOfEnergypThetaDot = 0.0
        self.balanceOfEnergypEpsDot = zeros((self.numberOfGaussPoints, 6), dtype=float64)


    def resetStiffness(self):
        self.stiffness = zeros((self.numberOfNodes * self.dofPerNode, self.numberOfNodes * self.dofPerNode), dtype=float64)
        
        
    def resetForces(self):
        self.forces = zeros((self.numberOfNodes, self.dofPerNode), dtype=float64)

    #@jit
    def calculateShapeFunctions(self):
        # the following variables are only needed locally, so we don't have to save these in the element
        for gp in range(0, self.numberOfGaussPoints):
            isoDeformationGrad = zeros((3, 3), dtype=float64)
            inverseIsoDefoGrad = zeros((3, 3), dtype=float64)
            shapeFunctionDerivatives = zeros((self.numberOfNodes, 3), dtype=float64)

            for node in range(0, self.numberOfNodes):
                # we don't need to save these in the element for further calculations
                shapeFunctionDerivatives[node] = array([self.shapeFunctionX(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionY(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionZ(self.nodePositions[node], self.gaussPoints[gp])])

            for i in range(0,3):
                for j in range(0,3):
                    for k in range(0, self.numberOfNodes):
                        isoDeformationGrad[i][j] += self.nodes[k].position[i] * shapeFunctionDerivatives[k][j]   ###refPos?

            self.dVol[gp] = det(isoDeformationGrad) * self.weights[gp]

            # get inverse Jacobi matrix
            inverseIsoDefoGrad = inv(isoDeformationGrad)

            for k in range(0, self.numberOfNodes):
                # this array saves all nodal shape function values at each Gauss point
                self.shapeFunctionValues[gp][k] = array([inverseIsoDefoGrad[0][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[0][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[0][2] * shapeFunctionDerivatives[k][2], \
                                                         inverseIsoDefoGrad[1][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[1][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[1][2] * shapeFunctionDerivatives[k][2], \
                                                         inverseIsoDefoGrad[2][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[2][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[2][2] * shapeFunctionDerivatives[k][2], \
                                                         self.shapeFunction(self.nodePositions[k], self.gaussPoints[gp])]) 

                self.bmatrix[gp][k] = array([[self.shapeFunctionValues[gp][k][0], 0, 0],\
                                             [0, self.shapeFunctionValues[gp][k][1], 0],\
                                             [0, 0, self.shapeFunctionValues[gp][k][2]],\
                                             [self.shapeFunctionValues[gp][k][1], self.shapeFunctionValues[gp][k][0], 0],\
                                             [0, self.shapeFunctionValues[gp][k][2], self.shapeFunctionValues[gp][k][1]],\
                                             [self.shapeFunctionValues[gp][k][2], 0, self.shapeFunctionValues[gp][k][0]]])



    # the field variables for Newton-Raphson
    def computeFieldVars(self):
        self.gradU     = zeros((self.numberOfGaussPoints, 3, 3), dtype=float64)
        self.gradUDot  = zeros((self.numberOfGaussPoints, 3, 3), dtype=float64)
        self.theta    = zeros(self.numberOfGaussPoints, dtype=float64)
        self.thetaDot  = zeros(self.numberOfGaussPoints, dtype=float64)
        self.gradTheta = zeros((self.numberOfGaussPoints, 3), dtype=float64)         
        
        for gp in range(0, self.numberOfGaussPoints):
            for k in range(0, self.numberOfNodes):
                for i in range(0, 3):
                    for j in range(0, 3):
                        self.gradU[gp][i][j]    += self.shapeFunctionValues[gp][k][j] * self.nodes[k].displacement[i]
                        self.gradUDot[gp][i][j] += self.shapeFunctionValues[gp][k][j] * self.nodes[k].velocity[i]
                        
                    self.gradTheta[gp][i] += self.shapeFunctionValues[gp][k][i] * self.nodes[k].displacement[3]
                    
                self.theta[gp] += self.shapeFunctionValues[gp][k][3] * self.nodes[k].displacement[3]
                self.thetaDot[gp] += self.shapeFunctionValues[gp][k][3] * self.nodes[k].velocity[3]


    # "material routine"
    def computeStresses(self):
        for gp in range(0, self.numberOfGaussPoints):
            # linear strain vector
            self.epsilon[gp][0] = self.gradU[gp][0][0]
            self.epsilon[gp][1] = self.gradU[gp][1][1]
            self.epsilon[gp][2] = self.gradU[gp][2][2]
            self.epsilon[gp][3] = 1/2* (self.gradU[gp][0][1] + self.gradU[gp][1][0])
            self.epsilon[gp][4] = 1/2*(self.gradU[gp][1][2] + self.gradU[gp][2][1])
            self.epsilon[gp][5] = 1/2*(self.gradU[gp][2][0] + self.gradU[gp][0][2])

            self.epsilonDot[gp][0] = self.gradUDot[gp][0][0]
            self.epsilonDot[gp][1] = self.gradUDot[gp][1][1]
            self.epsilonDot[gp][2] = self.gradUDot[gp][2][2]
            self.epsilonDot[gp][3] = 1/2*(self.gradUDot[gp][0][1] + self.gradUDot[gp][1][0])
            self.epsilonDot[gp][4] = 1/2*(self.gradUDot[gp][1][2] + self.gradUDot[gp][2][1])
            self.epsilonDot[gp][5] = 1/2*(self.gradUDot[gp][2][0] + self.gradUDot[gp][0][2])

            for i in range(0,3):
                self.traceEps[gp] += self.epsilon[gp][i]
                self.traceEpsDot[gp] += self.epsilonDot[gp][i]

            self.sigma[gp][0] = 2.0 * self.lmu * self.epsilon[gp][0] + self.llambda * self.traceEps[gp] \
                              - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            self.sigma[gp][1] = 2.0 * self.lmu * self.epsilon[gp][1] + self.llambda * self.traceEps[gp] \
                              - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            self.sigma[gp][2] = 2.0 * self.lmu * self.epsilon[gp][2] + self.llambda * self.traceEps[gp] \
                              - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            self.sigma[gp][3] = 2.0 * self.lmu * self.epsilon[gp][3]
            self.sigma[gp][4] = 2.0 * self.lmu * self.epsilon[gp][4]
            self.sigma[gp][5] = 2.0 * self.lmu * self.epsilon[gp][5]
            
            self.qlin[gp][0] = -self.alphaT * self.gradTheta[gp][0]
            self.qlin[gp][1] = -self.alphaT * self.gradTheta[gp][1]
            self.qlin[gp][2] = -self.alphaT * self.gradTheta[gp][2]
            

            self.balanceOfEnergy[gp] = -self.rho * self.cp * self.thetaDot[gp] \
                                       - 3.0 * self.alphaS * self.theta[gp] * self.kappa * self.traceEpsDot[gp] \
                                       + self.rho * self.rT
                                       
  
            # elastic tangent modulus in Voigt notation                                  
            self.mC[0][0] = 2 * self.lmu + self.llambda
            self.mC[0][1] = self.llambda
            self.mC[0][2] = self.llambda
            self.mC[1][0] = self.llambda
            self.mC[1][1] = 2 * self.lmu + self.llambda
            self.mC[1][2] = self.llambda
            self.mC[2][0] = self.llambda
            self.mC[2][1] = self.llambda
            self.mC[2][2] = 2 * self.lmu + self.llambda
            self.mC[3][3] = 2 * self.lmu
            self.mC[4][4] = 2 * self.lmu
            self.mC[5][5] = 2 * self.lmu
                                       
            self.sigmapTheta[gp][0] = -3.0 * self.alphaS * self.kappa
            self.sigmapTheta[gp][1] = -3.0 * self.alphaS * self.kappa
            self.sigmapTheta[gp][2] = -3.0 * self.alphaS * self.kappa
            self.sigmapTheta[gp][3] = 0.0
            self.sigmapTheta[gp][4] = 0.0
            self.sigmapTheta[gp][5] = 0.0
            
            self.qlinpGradTheta[0][0] = -self.alphaT 
            self.qlinpGradTheta[1][1] = -self.alphaT
            self.qlinpGradTheta[2][2] = -self.alphaT

            self.balanceOfEnergypTheta[gp] = -3.0 * self.alphaS * self.kappa * self.traceEpsDot[gp]

            self.balanceOfEnergypThetaDot = -self.rho * self.cp

            self.balanceOfEnergypEpsDot[gp][0] = -3.0 * self.alphaS * self.kappa * self.theta[gp]
            self.balanceOfEnergypEpsDot[gp][1] = -3.0 * self.alphaS * self.kappa * self.theta[gp]
            self.balanceOfEnergypEpsDot[gp][2] = -3.0 * self.alphaS * self.kappa * self.theta[gp]
            self.balanceOfEnergypEpsDot[gp][3] = 0.0
            self.balanceOfEnergypEpsDot[gp][4] = 0.0
            self.balanceOfEnergypEpsDot[gp][5] = 0.0

            # von Mises stress
            self.vonMises[gp] += (self.sigma[gp][0] - self.sigma[gp][1])**2 + (self.sigma[gp][1] - self.sigma[gp][2])**2 \
                         + (self.sigma[gp][2] - self.sigma[gp][0])**2 \
                         + 6.0 * (self.sigma[gp][3]**2 + self.sigma[gp][4]**2 + self.sigma[gp][5]**2)
            self.vonMises[gp] = 1.0/2.0 * sqrt(self.vonMises[gp])

            # update nodal stresses and von Mises stresses
            for k in range(0, self.numberOfNodes):
                self.nodes[k].weightFactor += self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                self.nodes[k].vonMises += self.vonMises[gp] * self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                for i in range(0, 2*3):
                    self.nodes[k].sigma[i] += self.sigma[gp][i] * self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                    


    def computeStiffness(self):
        fv = self.nodes[0].betaNM/(self.nodes[0].alphaNM * self.nodes[0].timeStep)
        fa = 1.0 / (self.nodes[0].alphaNM * self.nodes[0].timeStep**2)
        
        computeStiffness(fv, fa, self.stiffness, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.mC, self.dVol, self.shapeFunctionValues, self.balanceOfEnergy, self.balanceOfEnergypTheta, self.balanceOfEnergypThetaDot, self.balanceOfEnergypEpsDot, self.sigmapTheta, self.qlinpGradTheta)

    def computeForces(self):
        computeForces(self.forces, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.sigma, self.dVol, self.shapeFunctionValues, self.qlin, self.balanceOfEnergy)
        # move forces to nodes
        for node in range(0, self.numberOfNodes):
            self.nodes[node].forces += self.forces[node]


    def printNodes(self):
        for node in self.nodes:
            node.printCoordinates()