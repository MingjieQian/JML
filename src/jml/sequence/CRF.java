package jml.sequence;

import static jml.matlab.Matlab.display;
import static jml.matlab.Matlab.exp;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.isnan;
import static jml.matlab.Matlab.log;
import static jml.matlab.Matlab.max;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.repmat;
import static jml.matlab.Matlab.setMatrix;
import static jml.matlab.Matlab.sum;
import static jml.matlab.Matlab.sumAll;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.uminus;
import static jml.operation.ArrayOperation.allocateIntegerVector;
import static jml.operation.ArrayOperation.allocateVector;
import static jml.operation.ArrayOperation.argmax;
import static jml.operation.ArrayOperation.sum;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;
import java.util.TreeMap;

import jml.optimization.LBFGS;

import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;


/***
 * A Java implementation for the basic Conditional Random Field (CRF).
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 22ed, 2013
 */
public class CRF {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// Number of data sequences
		int D = 50;
		// Minimal length for the randomly generated data sequences
		int n_min = 5;
		// Maximal length for the randomly generated data sequences
		int n_max = 10;
		// Number of feature functions
		int d = 5;
		// Number of states
		int N = 3;
		// Sparseness for the feature matrices
		double sparseness = 0.2;
		
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
		/*fprintf("Predicted label sequence without normalization:\n");
		display(CRF.predict2(Fst));*/
		
	}
	
	/**
	 * A 3D {@code RealMatrix} array, where F[k][i][j] is the sparse
	 * feature matrix for the j-th feature of the k-th observation sequence
	 * at position i, i.e., f_{j}^{{\bf x}_k, i}.
	 */
	RealMatrix[][][] Fs;
	
	/**
	 * Number of data sequences for training.
	 */
	int D;
	
	/**
	 * Number of parameters or number of features
	 */
	int d;
	
	/**
	 * Number of states in the state space.
	 */
	int numStates;
	
	/**
	 * Index for the start state in the state space.
	 */
	int startIdx;
	
	/**
	 * A 2D integer array, where Ys[k][i] is the label index for the label 
	 * of the k-th training data sequence at position i in the label space.
	 */
	int[][] Ys;
	
	/**
	 * d x 1 parameter vector.
	 */
	RealMatrix W;
	
	/**
	 * Convergence precision.
	 */
	double epsilon;
	
	/**
	 * Regularization parameter
	 */
	double sigma = 1;
	
	/**
	 * Maximal number of iterations.
	 */
	int maxIter = 50;
	
	/**
	 * Constructor for a CRF instance.
	 */
	public CRF() {
		this.epsilon = 1e-4;
		startIdx = 0;
	}
	
	/**
	 * Constructor for a CRF instance.
	 * 
	 * @param epsilon convergence precision
	 * 
	 */
	public CRF(double epsilon) {
		this.epsilon = epsilon;
		startIdx = 0;
	}
	
	/**
	 * Constructor for a CRF instance.
	 * 
	 * @param d number of feature functions
	 * 
	 */
	public CRF(int d) {
		this.d = d;
		this.epsilon = 1e-4;
		startIdx = 0;
	}
	
	/**
	 * Constructor for a CRF instance.
	 * 
	 * @param d number of feature functions
	 * 
	 * @param epsilon convergence precision
	 * 
	 */
	public CRF(int d, double epsilon) {
		this.d = d;
		this.epsilon = epsilon;
		startIdx = 0;
	}
	
	/**
	 * Feed data sequences for training.
	 * 
	 * @param Fs a 3D {@code RealMatrix} array, where F[k][i][j] is the sparse
	 * 			 feature matrix for the j-th feature of the k-th observation sequence
	 * 		     at position i, i.e., f_{j}^{{\bf x}_k, i}
	 * 
	 */
	public void feedData(RealMatrix[][][] Fs) {
		this.Fs = Fs;
		D = Fs.length;
		d = Fs[0][0].length;
		numStates = Fs[0][0][0].getRowDimension();
	}
	
	/**
	 * Feed labels for training data sequences.
	 * 
	 * @param Ys a 2D integer array, where Ys[k][i] is the label index for the label 
	 *           of the k-th training data sequence at position i in the label space
	 * 
	 */
	public void feedLabels(int[][] Ys) {
		this.Ys = Ys;
	}
	
	/**
	 * Generate random data sequences to train a CRF model.
	 *  
	 * @param D number of data sequences to be generated
	 * 
	 * @param n_min minimal length for the randomly generated data sequences
	 * 
	 * @param n_max maximal length for the randomly generated data sequences
	 * 
	 * @param d number of feature functions
	 * 
	 * @param N number of states
	 * 
	 * @param sparseness sparseness for the feature matrices
	 * 
	 * @return a data label pair res[2], res[0] is the data sequences and res[1]
	 *         is the corresponding label sequences
	 * 
	 */
	public static Object[] generateDataSequences(int D, int n_min, int n_max, int d, int N, double sparseness) {
		
		Object[] res = new Object[2];
		RealMatrix[][][] Fs = new RealMatrix[D][][];
		int[][] Ys = new int[D][];
		
		Random generator = new Random();
		double prob = 0;
		double feaVal = 0;
		for (int k = 0; k < D; k++) {
			int n_x = generator.nextInt(n_max - n_min + 1) + n_min;
			Fs[k] = new RealMatrix[n_x][];
			Ys[k] = new int[n_x];
			for (int i = 0; i < n_x; i++) {
				Ys[k][i] = generator.nextInt(N);
				Fs[k][i] = new RealMatrix[d];
				for (int j = 0; j < d; j++) {
					Fs[k][i][j] = new OpenMapRealMatrix(N, N);
					for (int previous = 0; previous < N; previous++) {
						for (int current = 0; current < N; current++) {
							prob = generator.nextDouble();
							if (prob < sparseness) {
								feaVal = 1.0;
								Fs[k][i][j].setEntry(previous, current, feaVal);
							}
						}
					}
				}
			}
		}
		
		res[0] = Fs;
		res[1] = Ys;
		
		return res;
		
	}
	
	/**
	 * Estimate parameters for the basic CRF by a maximum conditional 
	 * log-likelihood estimation principle.
	 */
	public void train() {
	
		double fval = 0;
		
		/*
		 * Initialize the parameters 
		 */
		W = times(10, ones(d, 1));
		
		RealMatrix Grad = new BlockRealMatrix(d, 1);
		
		fval = computeObjectiveFunctionValue(true, Grad, W);
		
		boolean flags[] = null;
		// double fval_pre = 0;
		int k = 0;
		while (true) {
			flags = LBFGS.run(Grad, fval, epsilon, W);
			if (flags[0])
				break;
			// display(W);
			if (sumAll(isnan(W)) > 0) {
				int a = 1;
				a = a + 1;
			}
			
			/*
			 *  Compute the objective function value, if flags[1] is true
			 *  gradient will also be computed.
			 */
			fval = computeObjectiveFunctionValue(flags[1], Grad, W);
			if (flags[1]) {
				k = k + 1;
				// Compute the gradient
				if (k > maxIter)
					break;
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
			}
			
		}
	}
	
	/**
	 * Compute the objective function value (the mean log-likelihood on training
	 * data for CRFs). Gradient is also calculated if required.
	 *  
	 * @param calcGrad if gradient required
	 * 
	 * @param Grad gradient to be assigned in place if required
	 *  
	 * @param W model parameters
	 * 
	 * @return objective function value
	 * 
	 */
	public double computeObjectiveFunctionValue(boolean calcGrad, RealMatrix Grad, RealMatrix W) {
		
		double fval = 0;
		
		int[] Y = null;
		
		/*
		 * d x 1 feature vector for D training data sequences
		 */
		RealMatrix F = new BlockRealMatrix(d, 1);
		
		/*
		 * d x 1 array conditional expectation of feature functions 
		 * for D training data sequences, i.e., 
		 * EF = sum_k E_{P_{\bf \lambda}(Y|x_k)}[F(Y, x_k)]
		 */
		RealMatrix EF = new BlockRealMatrix(d, 1);
		double[] EFArr = allocateVector(d);
		
		/*
		 * Compute global feature vector for all training data sequences, 
		 * i.e., sum_k F(y_k, x_k)
		 */
		int n_x = 0;
		double f = 0;	
		for (int j = 0; j < d; j++) {
			f = 0;
			for (int k = 0; k < D; k++) {
				Y = Ys[k];
				n_x = Fs[k].length;
				for (int i = 0; i < n_x; i++) {
					if (i == 0)
						f += Fs[k][i][j].getEntry(0, Y[i]);
					else
						f += Fs[k][i][j].getEntry(Y[i - 1], Y[i]);
				}
			}
			F.setEntry(j, 0, f);
		}
		
		RealMatrix[] Ms = null;
		RealMatrix f_j_x_i = null;
		
		for (int k = 0; k < D; k++) {
			
			n_x = Fs[k].length;
			Ms = new RealMatrix[n_x];
			
			/*
			 * Compute transition matrix set
			 */
			for (int i = 0; i < n_x; i++) {
				for (int j = 0; j < d; j++) {
					f_j_x_i = Fs[k][i][j];
					if (j == 0)
						Ms[i] = times(W.getEntry(j, 0), f_j_x_i);
					else
						Ms[i] = plus(Ms[i], times(W.getEntry(j, 0), f_j_x_i));
				}
				Ms[i] = exp(Ms[i]);
				for (int s = 0; s < numStates; s++) {
					if (sum(Ms[i].getRow(s)) == 0) {
						Ms[i].setRow(s, allocateVector(numStates, 1e-10));
					}
				}
			}
			
			/*
			 * Forward recursion with scaling
			 */
			RealMatrix Alpha_hat = new BlockRealMatrix(numStates, n_x);
			RealMatrix Alpha_hat_0 = new BlockRealMatrix(numStates, 1);
			RealVector e_start = new ArrayRealVector(numStates);
			e_start.setEntry(startIdx, 1);
			double[] c = allocateVector(n_x);
			
			Alpha_hat_0.setColumnVector(0, e_start);
			for (int i = 0; i < n_x; i++) {
				if (i == 0) {
					Alpha_hat.setColumnMatrix(i, Ms[i].transpose().multiply(Alpha_hat_0));
				} else {
					Alpha_hat.setColumnMatrix(i, Ms[i].transpose().multiply(Alpha_hat.getColumnMatrix(i - 1)));
				}
				c[i] = 1.0 / sum(Alpha_hat.getColumnVector(i));
				if (Double.isInfinite(c[i])) {
					int a = 1;
					a = a + 1;
				}
				Alpha_hat.setColumnMatrix(i, times(c[i], Alpha_hat.getColumnMatrix(i)));
			}
			
			/*
			 * Backward recursion with scaling
			 */
			RealMatrix Beta_hat = new BlockRealMatrix(numStates, n_x);
			for (int i = n_x - 1; i >= 0; i--) {
				if ( i == n_x - 1) {
					Beta_hat.setColumnMatrix(i, ones(numStates, 1));
				} else {
					Beta_hat.setColumnMatrix(i, mtimes(Ms[i + 1], Beta_hat.getColumnMatrix(i + 1)));
				}
				Beta_hat.setColumnMatrix(i, times(c[i], Beta_hat.getColumnMatrix(i)));
			}
			
			/*
			 * Accumulate the negative conditional log-likelihood on the
			 * D training data sequences
			 */
			for (int i = 0; i < n_x; i++) {
				fval -= Math.log(c[i]);
			}
			
			/*if (Double.isNaN(fval)) {
				int a = 1;
				a = a + 1;
			}*/
			
			if (!calcGrad)
				continue;
			/*
			 * Compute E_{P_{\bf \lambda}(Y|x_k)}[F(Y, x_k)]
			 */
			for (int j = 0; j < d; j++) {
				/*
				 * Compute E_{P_{\bf \lambda}(Y|x_k)}[F_{j}(Y, x_k)]
				 */
				for (int i = 0; i < n_x; i++) {
					if (i == 0) {
						EFArr[j] += Alpha_hat_0.transpose().multiply(times(Ms[i], f_j_x_i)).multiply(Beta_hat.getColumnMatrix(i)).getEntry(0, 0);
					} else {
						EFArr[j] += Alpha_hat.getColumnMatrix(i - 1).transpose().multiply(times(Ms[i], f_j_x_i)).multiply(Beta_hat.getColumnMatrix(i)).getEntry(0, 0);
					}
				}
			}
			
		}
		
		
		/*
		 * Calculate the eventual negative conditional log-likelihood
		 */
		fval -= innerProduct(W, F);
		fval += sigma * innerProduct(W, W);
		fval /= D;
		
		
		if (!calcGrad) {
			return fval;
		}
		
		/*
		 * Calculate the gradient of negative conditional log-likelihood
		 * w.r.t. W. on the D training data sequences
		 */
		EF.setColumn(0, EFArr);
		setMatrix(Grad, (rdivide(plus(minus(EF, F), times(2 * sigma, W)), D)));
		
		return fval;
		
	}
	
	/**
	 * Predict the single best label sequence given the features for an 
	 * observation sequence by Viterbi algorithm without normalization.
	 * 
	 * @param Fs a 2D {@code RealMatrix} array, where F[i][j] is the sparse
	 * 			 feature matrix for the j-th feature of the observation sequence
	 *	 	 	 at position i, i.e., f_{j}^{{\bf x}, i}
	 *
	 * @return the single best label sequence for an observation sequence
	 * 
	 */
	public int[] predict2(RealMatrix[][] Fs) {
		
		int n_x = Fs.length;
		
		/*
		 * sum_j lambda_j * f_j(y_{i - 1}, y_i, x, i) 
		 */
		RealMatrix[] positionScoreMatrices = new RealMatrix[n_x];
		
		for (int i = 0; i < n_x; i++) {
			positionScoreMatrices[i] = new OpenMapRealMatrix(numStates, numStates);
			for (int j = 0; j < d; j++) {
				positionScoreMatrices[i] = plus(positionScoreMatrices[i], times(W.getEntry(j, 0), Fs[i][j]));
			}
		}
		
		/*
		 * Viterbi algorithm
		 */
		RealMatrix VarPhi = new BlockRealMatrix(n_x, numStates);
		RealMatrix Psi = new BlockRealMatrix(n_x, numStates);
		RealMatrix VarPhi_i_minus_1 = null;
		TreeMap<String, RealMatrix> maxResults = null;
		
		for (int i = 0; i < n_x; i++) {
			if (i == 0) { // Initialization
				VarPhi.setRowMatrix(i, positionScoreMatrices[i].getRowMatrix(startIdx));
				Psi.setRowMatrix(i, uminus(ones(1, numStates)));
			} else {
				VarPhi_i_minus_1 = VarPhi.getRowMatrix(i - 1).transpose();
				maxResults = max(plus(repmat(VarPhi_i_minus_1, 1, numStates), positionScoreMatrices[i]), 1);
				VarPhi.setRowMatrix(i, maxResults.get("val"));
				Psi.setRowMatrix(i, maxResults.get("idx"));
			}
		}
		
		/*
		 *  Predict the single best label sequence.
		 */
		double[] phi_n_x = VarPhi.getRow(n_x - 1);
		int[] YPred = allocateIntegerVector(n_x);
		for (int i = n_x - 1; i >= 0; i--) {
			if (i == n_x - 1) {
				YPred[i] = argmax(phi_n_x);
			} else {
				YPred[i] = (int)Psi.getEntry(i + 1, YPred[i + 1]);
			}
		}
		
		/*display(Phi);
		display(Psi);*/
		
		/*
		 *  Predict the optimal conditional probability: P*(y|x)
		 */
		double p = Math.exp(phi_n_x[YPred[n_x - 1]]);
		fprintf("P*(YPred|x) = %g\n", p);
		
		return YPred;
		
	}
	/**
	 * Predict the single best label sequence given the features for an 
	 * observation sequence by Viterbi algorithm.
	 * 
	 * @param Fs a 2D {@code RealMatrix} array, where F[i][j] is the sparse
	 * 			 feature matrix for the j-th feature of the observation sequence
	 *	 	 	 at position i, i.e., f_{j}^{{\bf x}, i}
	 *
	 * @return the single best label sequence for an observation sequence
	 *
	 */
	public int[] predict(RealMatrix[][] Fs) {
		
		RealMatrix[] Ms = computeTransitionMatrix(Fs);
		
		/*
		 * Alternative backward recursion with scaling for the Viterbi
		 * algorithm
		 */
		int n_x = Fs.length;
		double[] b = allocateVector(n_x);
		RealMatrix Beta_tilta = new BlockRealMatrix(numStates, n_x);
		for (int i = n_x - 1; i >= 0; i--) {
			if ( i == n_x - 1) {
				Beta_tilta.setColumnMatrix(i, ones(numStates, 1));
			} else {
				Beta_tilta.setColumnMatrix(i, mtimes(Ms[i + 1], Beta_tilta.getColumnMatrix(i + 1)));
			}
			b[i] = 1.0 / sum(Beta_tilta.getColumnVector(i));
			Beta_tilta.setColumnMatrix(i, times(b[i], Beta_tilta.getColumnMatrix(i)));
		}
		
		/*fprintf("Beta:\n");
		display(Beta_tilta);*/
		
		/*
		 * Gammas[i](y_{i-1}, y_[i]) is P(y_i|y_{i-1}, Lambda), thus each row of
		 * Gammas[i] should be sum to one.
		 */
		RealMatrix[] Gammas = new RealMatrix[n_x];
		// fprintf("Gammas:\n");
		for (int i = 0; i < n_x; i++) {
			Gammas[i] = times(Ms[i], repmat(Beta_tilta.getColumnMatrix(i).transpose(), numStates, 1));
			/*fprintf("Original Gammas[%d]:\n", i);
			display(Gammas[i]);*/
			/*
			 * Sum to 1 for every row of Gammas[i]
			 */
			/*display(sum(Gammas[i], 2));
			display(repmat(sum(Gammas[i], 2), 1, numStates));*/
			Gammas[i] = rdivide(Gammas[i], repmat(sum(Gammas[i], 2), 1, numStates));
			/*fprintf("Gammas[%d]:\n", i);
			display(Gammas[i]);
			fprintf("Ms[%d]:\n", i);
			display(full(Ms[i]));*/
		}
		
		/*
		 * Viterbi algorithm
		 */
		RealMatrix Phi = new BlockRealMatrix(n_x, numStates);
		RealMatrix Psi = new BlockRealMatrix(n_x, numStates);
		RealMatrix Phi_i_minus_1 = null;
		TreeMap<String, RealMatrix> maxResults = null;
		
		for (int i = 0; i < n_x; i++) {
			if (i == 0) { // Initialization
				Phi.setRowMatrix(i, log(Gammas[i].getRowMatrix(startIdx)));
				Psi.setRowMatrix(i, uminus(ones(1, numStates)));
			} else {
				Phi_i_minus_1 = Phi.getRowMatrix(i - 1).transpose();
				maxResults = max(plus(repmat(Phi_i_minus_1, 1, numStates), log(Gammas[i])), 1);
				Phi.setRowMatrix(i, maxResults.get("val"));
				Psi.setRowMatrix(i, maxResults.get("idx"));
			}
		}
		
		/*
		 *  Predict the single best label sequence.
		 */
		double[] phi_n_x = Phi.getRow(n_x - 1);
		int[] YPred = allocateIntegerVector(n_x);
		for (int i = n_x - 1; i >= 0; i--) {
			if (i == n_x - 1) {
				YPred[i] = argmax(phi_n_x);
			} else {
				YPred[i] = (int)Psi.getEntry(i + 1, YPred[i + 1]);
			}
		}
		
		/*display(Phi);
		display(Psi);*/
		
		/*
		 *  Predict the optimal conditional probability: P*(y|x)
		 */
		double p = Math.exp(phi_n_x[YPred[n_x - 1]]);
		fprintf("P*(YPred|x) = %g\n", p);
		
		return YPred;
		
	}
	
	/**
	 * Compute transition matrix set, i.e., {M^{\bf x}_i}, i = 1, 2, ..., n_x
	 * 
	 * @param Fs A 2D {@code RealMatrix} array, where F[i][j] is the sparse
	 * 			 feature matrix for the j-th feature of the observation sequence
	 *	 	 	 at position i, i.e., f_{j}^{{\bf x}, i}
	 *
	 * @return a transition matrix sequences of length n_x
	 * 
	 */
	public RealMatrix[] computeTransitionMatrix(RealMatrix[][] Fs) {
		
		int n_x = Fs.length;
		RealMatrix f_j_x_i = null;
		RealMatrix[] Ms = new RealMatrix[n_x];
		for (int i = 0; i < n_x; i++) {
			for (int j = 0; j < d; j++) {
				f_j_x_i = Fs[i][j];
				if (j == 0)
					Ms[i] = times(W.getEntry(j, 0), f_j_x_i);
				else
					Ms[i] = plus(Ms[i], times(W.getEntry(j, 0), f_j_x_i));
			}
			Ms[i] = exp(Ms[i]);
			for (int s = 0; s < numStates; s++) {
				if (sum(Ms[i].getRow(s)) == 0) {
					Ms[i].setRow(s, allocateVector(numStates, 1e-10));
				}
			}
		}
		return Ms;
		
	}
	
	/**
	 * Compute global feature vector for a data sequences Fs with label Ys.
	 * 
	 * @param Fs a 2D {@code RealMatrix} array, where F[i][j] is the sparse
	 * 			 feature matrix for the j-th feature of the observation sequence
	 *	 	 	 at position i, i.e., f_{j}^{{\bf x}, i}
	 *
	 * @param Ys a 1D integer array, where Ys[i] is the label index in the label 
	 *           space for the label at position i for the data sequence 
	 * 
	 * @return global feature vector
	 * 
	 */
	public RealMatrix computeFeatureVector(RealMatrix[][] Fs, int[] Ys) {
		
		/*
		 * d x 1 feature vector for D training data sequences
		 */
		RealMatrix F = new BlockRealMatrix(d, 1);
		
		int[] Y = null;
		
		/*
		 * Compute feature vector, i.e., F(y, x)
		 */
		int n_x = 0;
		double f = 0;	
		for (int j = 0; j < d; j++) {
			f = 0;
			Y = Ys;
			n_x = Fs.length;
			for (int i = 0; i < n_x; i++) {
				if (i == 0)
					f += Fs[i][j].getEntry(0, Y[i]);
				else
					f += Fs[i][j].getEntry(Y[i - 1], Y[i]);
			}
			F.setEntry(j, 0, f);
		}
		
		return F;
		
	}
	
	/**
	 * Alternative backward recursion with scaling for the Viterbi algorithm.
	 * 
	 * @param Ms a set of transition matrices, i.e., 
	 *           {M^{\bf x}_i}, i = 1, 2, ..., n_x
	 *           
	 * @return scaled backward matrix variable {\bf \tilde \beta}(y_i, i)
	 * 
	 */
	public RealMatrix backwardRecursion4Viterbi(RealMatrix[] Ms) {
		
		/*
		 * Alternative backward recursion with scaling for the Viterbi
		 * algorithm
		 */
		int n_x = Fs.length;
		double[] b = allocateVector(n_x);
		RealMatrix Beta_tilta = new BlockRealMatrix(numStates, n_x);
		for (int i = n_x - 1; i >= 0; i--) {
			if ( i == n_x - 1) {
				Beta_tilta.setColumnMatrix(i, ones(numStates, 1));
			} else {
				Beta_tilta.setColumnMatrix(i, mtimes(Ms[i + 1], Beta_tilta.getColumnMatrix(i + 1)));
			}
			b[i] = 1.0 / sum(Beta_tilta.getColumnVector(i));
			Beta_tilta.setColumnMatrix(i, times(b[i], Beta_tilta.getColumnMatrix(i)));
		}
		
		return Beta_tilta;
	}
	
	/**
	 * Save the model to a file.
	 * 
	 * @param filePath file path to save the model
	 * 
	 */
	public void saveModel(String filePath) {

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(new CRFModel(numStates, startIdx, W));
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Load the model from a file.
	 * 
	 * @param filePath file Path to load the model from
	 * 
	 */
	public void loadModel(String filePath) {

		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			CRFModel CRFModel = (CRFModel)ois.readObject();
			W = CRFModel.W;
			d = CRFModel.d;
			startIdx = CRFModel.startIdx;
			numStates = CRFModel.numStates;
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

	}

}

/***
 * HMM model parameters.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 15th, 2013
 */
class CRFModel implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -2734854735411482584L;

	// **************** Model Parameters: **************** //

	/**
	 * Number of states in the state space.
	 */
	int numStates;
	
	/**
	 * Index for the start state in the state space.
	 */
	int startIdx;
	
	/**
	 * Number of parameters or number of features
	 */
	int d;
	
	/**
	 * d x 1 parameter vector.
	 */
	RealMatrix W;

	// *************************************************** //

	/**
	 * Constructor for a CRF model.
	 * 
	 * @param numStates number of states in the state space
	 * 
	 * @param startIdx index for the start state in the state space
	 * 
	 * @param W parameter vector
	 * 
	 */
	public CRFModel(int numStates, int startIdx, RealMatrix W) {
		this.numStates = numStates;
		this.W = W;
		this.startIdx = startIdx;
		this.d = W.getRowDimension();
	}

}
