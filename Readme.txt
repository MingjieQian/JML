JML

JML is a pure Java library for machine learning. There are already several popular java libraries for machine learning or data mining such as Weka or Mallet, but they're not easy to use. The goal of JML is to make machine learning methods easy to use and speed up the code translation from Matlab to Java.

Current version implements logistic regression, Maximum Entropy modeling (MaxEnt), LASSO, KMeans, spectral clustering, Nonnegative Matrix Factorization (NMF), sparse NMF, Latent Semantic Indexing (LSI), Latent Dirichlet Allocation (LDA) (by Gibbs sampling based on LdaGibbsSampler.java by Gregor Heinrich), joint l_{2,1}-norms minimization, Hidden Markov Model (HMM), and Conditional Random Field (CRF)  just for examples of implementing machine learning methods by using this general framework. The SVM package LIBLINEAR is also incorporated. I will try to add more important models such as Markov Random Field (MRF) to this package if I get the time:)

JML library's another advantage is its complete independence from feature engineering, thus any preprocessed data could be run. For example, in the area of natural language processing, feature engineering is a crucial part for MaxEnt, HMM, and CRF to work well and is often embedded in model training. However, we believe that it is better to separate feature engineering and parameter estimation. On one hand, modularization could be achieved so that people can simply focus on one module without need to consider other modules; on the other hand, implemented modules could be reused without incompatibility concerns.

JML also provides implementations of several efficient, scalable, and widely used general purpose optimization algorithms, which are very important for machine learning methods be applicable on large scaled data, though particular optimization strategy that considers the characteristics of a particular problem is more effective and efficient (e.g., dual coordinate descent for bound constrained quadratic programming in SVM). Currently supported optimization algorithms are limited-memory BFGS, projected limited-memory BFGS (non-negative constrained or bound constrained), nonlinear conjugate gradient, primal-dual interior-point method, accelerated proximal gradient, and accelerated gradient descent. I would always like to implement more practical efficient optimization algorithms.

I hope this library could help engineers and researchers speed up their productivity cycle.

Documentation:
For more details about JML API, please refer to the online documentation.

Examples:

# Multi-class SVM (linear kernel)

double[][] data = { {3.5, 4.4, 1.3},
                             {5.3, 2.2, 0.5},
                             {0.2, 0.3, 4.1},
                             {-1.2, 0.4, 3.2} };

double[][] labels = { {1, 0, 0},
                                {0, 1, 0},
                                {0, 0, 1} };

double C = 1;
double eps = 0.01;
Classifier multiClassSVM = new MultiClassSVM(C, eps);
multiClassSVM.feedData(data);
multiClassSVM.feedLabels(labels);
multiClassSVM.train();

RealMatrix Y_pred = multiClassSVM.predictLabelMatrix(data);
display(Y_pred);

# Predicted label matrix
        1        0        0 
        0        1        0 
        0        0        1

# Predicted label score matrix:
   0.7950  -0.2050  -0.5899 
  -0.3301   0.6635  -0.3333 
  -0.6782  -0.1602   0.8383 

# Projection matrix (with bias):
  -0.1723   0.2483  -0.0760 
   0.2337  -0.1901  -0.0436 
  -0.0152  -0.1310   0.1462 
  -0.1525   0.0431   0.1094 
  -0.0207   0.0114   0.0093 

# -------------------------------------------------------------------------- #

# Multi-class Logistic Regression

double[][] data = { {3.5, 4.4, 1.3},
                             {5.3, 2.2, 0.5},
                             {0.2, 0.3, 4.1},
                             {-1.2, 0.4, 3.2} };

double[][] labels = { {1, 0, 0},
                                {0, 1, 0},
                                {0, 0, 1} };

Options options = new Options();
options.epsilon = 1e-6;
// Multi-class logistic regression by using limited-memory BFGS method
Classifier logReg = new LogisticRegressionMCLBFGS(options);
logReg.feedData(data);
logReg.feedLabels(labels);
logReg.train();

RealMatrix Y_pred = logReg.predictLabelScoreMatrix(data);
display(Y_pred);

# Predicted probability matrix:
   1.0000   0.0000   0.0000 
   0.0000   1.0000   0.0000 
   0.0000   0.0000   1.0000

# Projection matrix:
  -2.0348   3.0363  -1.0015 
   2.9002  -2.1253  -0.7749 
  -0.4261  -1.5264   1.9524 
  -2.0445   0.4998   1.5447   

# -------------------------------------------------------------------------- #

# Spectral Clustering

double[][] data = { {3.5, 4.4, 1.3},
                             {5.3, 2.2, 0.5},
                             {0.2, 0.3, 4.1},
                             {-1.2, 0.4, 3.2} };

SpectralClusteringOptions options = new SpectralClusteringOptions();
options.nClus = 2;
options.verbose = false;
options.maxIter = 100;
options.graphType = "nn";
options.graphParam = 2;
options.graphDistanceFunction = "cosine";
options.graphWeightType = "heat";
options.graphWeightParam = 1;
       
Clustering spectralClustering = new SpectralClustering(options);
spectralClustering.feedData(data);
spectralClustering.clustering();
display(spectralClustering.getIndicatorMatrix());

# Output
Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 2.
KMeans complete.
Spectral clustering complete.
        1        0 
        1        0 
        0        1

# -------------------------------------------------------------------------- #

# Limited-memory BFGS in the train() method of the LogisticRegressionMCLBFGS class

W = repmat(zeros(nFea, 1), new int[]{1, K});
A = X.transpose().multiply(W);
V = sigmoid(A);
G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nSample);
fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample;

boolean flags[] = null;
while (true) {
    flags = LBFGS.run(G, fval, epsilon, W);
    if (flags[0])
        break;
    A = X.transpose().multiply(W);
    V = sigmoid(A);
    fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample;
    if (flags[1])
        G = rdivide(X.multiply(V.subtract(Y)), nSample);
}

# -------------------------------------------------------------------------- #

# LASSO

double[][] data = {{1, 2, 3, 2},
                            {4, 2, 3, 6},
                            {5, 1, 2, 1}};

double[][] depVars = {{3, 2},
                                  {2, 3},
                                  {1, 4}};

Options options = new Options();
options.maxIter = 600;
options.lambda = 0.05;
options.verbose = !true;
options.epsilon = 1e-5;

Regression LASSO = new LASSO(options);
LASSO.feedData(data);
LASSO.feedDependentVariables(depVars);
LASSO.train();

fprintf("Projection matrix:\n");
display(LASSO.W);
       
RealMatrix Yt = LASSO.predict(data);
fprintf("Predicted dependent variables:\n");
display(Yt);

# Output
Projection matrix:
-0.2295    0.5994  
          0             0  
 1.1058    0.5858  
-0.0631   -0.1893  

Predicted dependent variables:
 2.9618    1.9782  
 2.0209    3.0191  
 1.0009    3.9791

# -------------------------------------------------------------------------- #

# LDA

int[][] documents = { {1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6},
                                 {2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2},
                                 {1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0},
                                 {5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0},
                                 {2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0},
                                 {5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2}};
       
LDAOptions LDAOptions = new LDAOptions();
LDAOptions.nTopic = 2;
LDAOptions.iterations = 5000;
LDAOptions.burnIn = 1500;
LDAOptions.thinInterval = 200;
LDAOptions.sampleLag = 10;
LDAOptions.alpha = 2;
LDAOptions.beta = 0.5;

LDA LDA = new LDA(LDAOptions);
LDA.readCorpus(documents);
LDA.train();

fprintf("Topic--term associations: \n");
display(LDA.topicMatrix);

fprintf("Document--topic associations: \n");
display(LDA.indicatorMatrix);

# Output

Topic--term associations:
   0.1258   0.0176 
   0.1531   0.0846 
   0.0327   0.3830 
   0.0418   0.1835 
   0.0360   0.2514 
   0.2713   0.0505 
   0.3393   0.0294 

Document--topic associations:
   0.2559   0.7441 
   0.1427   0.8573 
   0.8573   0.1427 
   0.6804   0.3196 
   0.5491   0.4509 
   0.4420   0.5580

# -------------------------------------------------------------------------- #

# Joint l_{2,1}-norms minimization: Supervised Feature Selection

double[][] data = { {3.5, 4.4, 1.3},
                             {5.3, 2.2, 0.5},
                             {0.2, 0.3, 4.1},
                             {-1.2, 0.4, 3.2} };

double[][] labels = { {1, 0, 0},
                               {0, 1, 0},
                               {0, 0, 1} };

SupervisedFeatureSelection robustFS = new JointL21NormsMinimization(2.0);
robustFS.feedData(data);
robustFS.feedLabels(labels);
robustFS.run();

System.out.println("Projection matrix:");
display(robustFS.getW());

# Output
Projection matrix:
  -0.0144   0.1194   0.0066 
   0.1988  -0.0777  -0.0133 
  -0.0193  -0.0287   0.2427 
  -0.0005   0.0004   0.0009

# -------------------------------------------------------------------------- #

# Hidden Markov Models

int numStates = 3;
int numObservations = 2;
double epsilon = 1e-6;
int maxIter = 1000;

double[] pi = new double[] {0.33, 0.33, 0.34};

double[][] A = new double[][] {
        {0.5, 0.3, 0.2},
        {0.3, 0.5, 0.2},
        {0.2, 0.4, 0.4}
};

double[][] B = new double[][] {
        {0.7, 0.3},
        {0.5, 0.5},
        {0.4, 0.6}
};

// Generate the data sequences for training
int D = 10000;
int T_min = 5;
int T_max = 10;
int[][][] data = HMM.generateDataSequences(D, T_min, T_max, pi, A, B);
int[][] Os = data[0];
int[][] Qs = data[1];

// Train HMM
HMM HMM = new HMM(numStates, numObservations, epsilon, maxIter);
HMM.feedData(Os);
HMM.feedLabels(Qs);
HMM.train();
HMM.saveModel("HMMModel.dat");

// Predict the single best state path
int ID = new Random().nextInt(D);
int[] O = Os[ID];
       
HMM HMMt = new HMM();
HMMt.loadModel("HMMModel.dat");
int[] Q = HMMt.predict(O);

fprintf("Observation sequence: \n");
HMMt.showObservationSequence(O);
fprintf("True state sequence: \n");
HMMt.showStateSequence(Qs[ID]);
fprintf("Predicted state sequence: \n");
HMMt.showStateSequence(Q);
double p = HMMt.evaluate(O);
System.out.format("P(O|Theta) = %f\n", p);

# Output

True Model Parameters:
Initial State Distribution:
   0.3300 
   0.3300 
   0.3400 

State Transition Probability Matrix:
   0.5000   0.3000   0.2000 
   0.3000   0.5000   0.2000 
   0.2000   0.4000   0.4000 

Observation Probability Matrix:
   0.7000   0.3000 
   0.5000   0.5000 
   0.4000   0.6000 

Trained Model Parameters:
Initial State Distribution:
   0.3556 
   0.3511 
   0.2934 

State Transition Probability Matrix:
   0.5173   0.3030   0.1797 
   0.2788   0.4936   0.2276 
   0.2307   0.3742   0.3951 

Observation Probability Matrix:
   0.6906   0.3094 
   0.5324   0.4676 
   0.3524   0.6476 

Observation sequence:
0 0 0 0 1 1 1 1 1
True state sequence:
0 2 0 0 2 0 2 2 2
Predicted state sequence:
0 0 0 0 2 2 2 2 2
P(O|Theta) = 0.001928

# -------------------------------------------------------------------------- #

# Maximum Entropy Modeling Using Limited-memory BFGS

double[][][] data = new double[][][] {
        {{1, 0, 0}, {2, 1, -1}, {0, 1, 2}, {-1, 2, 1}},
        {{0, 2, 0}, {1, 0, -1}, {0, 1, 1}, {-1, 3, 0.5}},
        {{0, 0, 0.8}, {2, 1, -1}, {1, 3, 0}, {-0.5, -1, 2}},
        {{0.5, 0, 0}, {1, 1, -1}, {0, 0.5, 1.5}, {-2, 1.5, 1}},
};

/*double [][] labels = new double[][] {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0}
};*/
int[] labels = new int[] {1, 2, 3, 1};

MaxEnt maxEnt = new MaxEnt();
maxEnt.feedData(data);
maxEnt.feedLabels(labels);
maxEnt.train();

fprintf("MaxEnt parameters:\n");
display(maxEnt.W);
String modelFilePath = "MaxEnt-Model.dat";
maxEnt.saveModel(modelFilePath);

maxEnt = new MaxEnt();
maxEnt.loadModel(modelFilePath);
fprintf("Predicted probability matrix:\n");
display(maxEnt.predictLabelScoreMatrix(data));
fprintf("Predicted label matrix:\n");
display(full(maxEnt.predictLabelMatrix(data)));
fprintf("Predicted labels:\n");
display(maxEnt.predict(data));

# Output

Training time: 0.087 seconds

MaxEnt parameters:
  12.1659 
  -1.8211 
  -4.4031 
  -1.8199 

Model saved.
Loading model...
Model loaded.
Predicted probability matrix:
   1.0000   0.0000   0.0000 
   0.0000   1.0000   0.0000 
   0.0000   0.0000   1.0000 
   1.0000   0.0000   0.0000 

Predicted label matrix:
        1        0        0 
        0        1        0 
        0        0        1 
        1        0        0 

Predicted labels:
        1 
        2 
        3 
        1

# -------------------------------------------------------------------------- #

# Conditional Random Field Using L-BFGS
// Number of data sequences
int D = 10;
// Minimal length for the randomly generated data sequences
int n_min = 5;
// Maximal length for the randomly generated data sequences
int n_max = 10;
// Number of feature functions
int d = 5;
// Number of states
int N = 3;
// Sparseness for the feature matrices
double sparseness = 0.8;

// Randomly generate labeled sequential data for CRF
Object[] dataSequences = CRF.generateDataSequences(D, n_min, n_max, d, N, sparseness);
RealMatrix[][][] Fs = (RealMatrix[][][]) dataSequences[0];
int[][] Ys = (int[][]) dataSequences[1];

// Train a CRF model for the randomly generated sequential data with labels
double epsilon = 1e-4;
CRF CRF = new CRF(epsilon);
CRF.feedData(Fs);
CRF.feedLabels(Ys);
CRF.train();

// Save the CRF model
String modelFilePath = "CRF-Model.dat";
CRF.saveModel(modelFilePath);
fprintf("CRF Parameters:\n");
display(CRF.W);

// Prediction
CRF = new CRF();
CRF.loadModel(modelFilePath);
int ID = new Random().nextInt(D);
int[] Yt = Ys[ID];
RealMatrix[][] Fst = Fs[ID];

fprintf("True label sequence:\n");
display(Yt);
fprintf("Predicted label sequence:\n");
display(CRF.predict(Fst));

# Output

Initial ofv: 117.452
Iter 1, ofv: 84.5760, norm(Grad): 4.71289
Iter 2, ofv: 11.9435, norm(Grad): 3.75200
Iter 3, ofv: 11.7764, norm(Grad): 0.619218
Objective function value doesn't decrease, iteration stopped!
Iter 4, ofv: 11.7764, norm(Grad): 0.431728
Model saved.
CRF Parameters:
1.1178 
3.1087 
-2.3664 
-0.3754 
-0.8732 

Loading model...
Model loaded.
True label sequence:
2          1          1          0          1 
Predicted label sequence:
P*(YPred|x) = 0.382742
2          1          1          0          1

# -------------------------------------------------------------------------- #

# General Quadratic Programming by Primal-dual Interior-point Methods

/*
 * Number of unknown variables
 */
int n = 5;

/*
 * Number of inequality constraints
 */
int m = 6;

/*
 * Number of equality constraints
 */
int p = 3;

RealMatrix x = rand(n, n);
RealMatrix Q = x.multiply(x.transpose()).add(times(rand(1), eye(n)));
RealMatrix c = rand(n, 1);

double HasEquality = 1;
RealMatrix A = times(HasEquality, rand(p, n));
x = rand(n, 1);
RealMatrix b = A.multiply(x);
RealMatrix B = rand(m, n);
double rou = -2;
RealMatrix d = plus(B.multiply(x), times(rou, ones(m, 1)));

/*
 * General quadratic programming:
 *
 *      min 2 \ x' * Q * x + c' * x
 * s.t. A * x = b
 *      B * x <= d
 */
GeneralQP.solve(Q, c, A, b, B, d);

# Output

Phase I:

Terminate successfully.

x_opt:
  4280.0981  2366.3295  -5241.6878  -3431.7929  -1243.1621 

s_opt:
   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 

lambda for the inequalities s_i >= 0:
   1.0000   1.0000   1.0000   1.0000   1.0000   1.0000 

B * x - d:
  -1622.6174  -3726.7628  -326.3261  -2610.0973  -3852.1984  -3804.4964 

lambda for the inequalities fi(x) <= s_i:
   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 

nu for the equalities A * x = b:
  -0.0000  -0.0000  -0.0000 

residual: 1.15753e-11

A * x - b:
  -0.0000  -0.0000   0.0000 

norm(A * x - b, "fro"): 0.000000

fval_opt: 4.78833e-11

The problem is feasible.

Computation time: 0.083000 seconds

halt execution temporarily in 1 seconds...

Phase II:

Terminate successfully.

residual: 4.97179e-12

Optimal objective function value: 201.252

Optimizer:
   7.0395  21.8530  -12.9418  -5.8026  -14.0337 

B * x - d:
  -0.0000  -9.1728  -0.0000  -2.9475  -0.5927  -7.7247 

lambda:
  17.6340   0.0000  186.0197   0.0000   0.0000   0.0000 

nu:
   7.4883  -55.7357  -57.6134 

norm(A * x - b, "fro"): 0.000000

Computation time: 0.048000 seconds

# -------------------------------------------------------------------------- #

Features:
    A general framework is provided for users to implement machine learning tools from MATLAB code.
    jml.clustering package implements clustering related models.
    jml.classification package implements classification related methods.
    jml.regression package implements regression models.
    jml.topics includes topic modeling and topic mining methods.
    jml.data provides reading and writing functions for dense or sparse matrix which can be easily read by MATLAB.
    tmj.kernel compute kernel matrix between two matrices, currently supported kernel types are linear, poly, rbf, and cosine.
    jml.manifold implements manifold learning related functions, i.e., computation of adjacency matrix and Laplacian matrix. This package is very useful for semi-supervised learning.
    jml.matlab implements some frequently used Matlab matrix functions with the same function input signature such as sort, sum, max, min, kron, vec, repmat, reshape, and colon. Thus Matlab code could be more easily converted to Java code.
    jml.optimization provides implementations for several most important general purpose optimization algorithm, currently supported optimization tools are limited-memory BFGS, projected limited-memory BFGS (non-negative constrained or bound constrained), nonlinear conjugate gradient, and primal-dual interior-point method.
    jml.feature.selection provides feature selection algorithms (supervised, unsupervised, or semi-supervised).
    jml.sequence implements sequential learning algorithms (e.g., HMM or CRF).
    jml.random implements random distributions. Currently supported is multivariate Gaussian distribution.
    Feature engineering and model training are separated completely, which increases the applicability and flexibility of the included learning models and methods in the library. For feature generation, we suggest using TextProcessor package.
    Well documented source code.

Note that the advantage of this library is that it is very convenient for users to translate a Matlab implementation into a Java implementation by using jml.matlab functions.

Dependencies:
JML depends on Apache Commons-Math library (commons-math-2.2 or later) and LIBLINEAR.

Note:
I choose Commons-Math because it supports both dense and sparse matrix and it has a good numerical computation performance by using BlockRealMatrix class in term of both speed and memory. For moderate scaled data, JMLBLAS is faster than JML because it exploits jblas library for basic matrix operations.

-----------------------------------
Author: Mingjie Qian
Version: 2.4
Date: Mar. 12th, 2013
 