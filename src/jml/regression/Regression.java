package jml.regression;

import jml.options.Options;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * Abstract super class for all regression methods.
 * 
 * @version 1.0 Jan. 14th, 2013
 * 
 * @author Mingjie Qian
 */
public abstract class Regression {

	/**
	 * Number of dependent variables.
	 */
	public int ny;
	
	/**
	 * Number of independent variables.
	 */
	public int p;
	
	/**
	 * Number of samples.
	 */
	public int n;
	
	/**
	 * Training data matrix (n x p) with each row being
	 * a data example.
	 */
	public RealMatrix X;
	
	/**
	 * Dependent variable matrix for training (n x ny).
	 */
	public RealMatrix Y;
	
	/**
	 * Unknown parameters represented as a matrix (p x ny).
	 */
	public RealMatrix W;
	
	/**
	 * Convergence tolerance.
	 */
	public double epsilon;
	
	/**
	 * Maximal number of iterations.
	 */
	public int maxIter;
	
	public Regression() {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		epsilon = 1e-6;
		maxIter = 600;
	}
	
	public Regression(double epsilon) {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		this.epsilon = epsilon;
		maxIter = 600;
	}
	
	public Regression(int maxIter, double epsilon) {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		this.epsilon = epsilon;
		this.maxIter = maxIter;
	}
	
	public Regression(Options options) {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		this.epsilon = options.epsilon;
		this.maxIter = options.maxIter;
	}
	
	/**
	 * Feed training data for the regression model.
	 * 
	 * @param X data matrix with each row being a data example
	 */
	public void feedData(RealMatrix X) {
		this.X = X;
		p = X.getColumnDimension();
		n = X.getRowDimension();
		if (Y != null && X.getRowDimension() != Y.getRowDimension()) {
			System.err.println("The number of dependent variable vectors and " +
					"the number of data samples do not match!");
			System.exit(1);
		}
	}
	
	/**
	 * Feed training data for this regression model.
	 * 
	 * @param data an n x d 2D {@code double} array with each
	 *             row being a data example
	 */
	public void feedData(double[][] data) {
		feedData(new BlockRealMatrix(data));
	}
	
	/**
	 * Feed training dependent variables for this regression model.
	 * 
	 * @param Y dependent variable matrix for training with each row being
	 *          the dependent variable vector for each data training data
	 *          example
	 */
	public void feedDependentVariables(RealMatrix Y) {
		this.Y = Y;
		ny = Y.getColumnDimension();
		if (X != null && Y.getRowDimension() != n) {
			System.err.println("The number of dependent variable vectors and " +
					"the number of data samples do not match!");
			System.exit(1);
		}
	}
	
	/**
	 * Feed training dependent variables for this regression model.
	 * 
	 * @param depVars an n x c 2D {@code double} array
	 * 
	 */
	public void feedDependentVariables(double[][] depVars) {
		feedDependentVariables(new BlockRealMatrix(depVars));
	}
	
	/**
	 * Train the regression model.
	 */
	public abstract void train();
	
	public abstract RealMatrix train(RealMatrix X, RealMatrix Y);
	
	/**
	 * Predict the dependent variables for test data Xt.
	 * 
	 * @param Xt test data matrix with each row being a
	 *           data example.
	 *        
	 * @return dependent variables for Xt
	 * 
	 */
	public RealMatrix predict(RealMatrix Xt) {
		if (Xt.getColumnDimension() != p) {
			System.err.println("Dimensionality of the test data " +
					"doesn't match with the training data!");
			System.exit(1);
		}
		return Xt.multiply(W);
	}
	
	/**
	 * Predict the dependent variables for test data Xt.
	 * 
	 * @param Xt an n x d 2D {@code double} array with each
	 *           row being a data example
	 *           
	 * @return dependent variables for Xt
	 * 
	 */
	public RealMatrix predict(double[][] Xt) {
		return predict(new BlockRealMatrix(Xt));
	}
	
	public abstract void loadModel(String filePath);
	
	public abstract void saveModel(String filePath);
	
}
