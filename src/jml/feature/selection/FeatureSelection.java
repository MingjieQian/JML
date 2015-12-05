package jml.feature.selection;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * Abstract class for feature selection algorithms.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 4th, 2012
 */
public abstract class FeatureSelection {
	
	/**
	 * Data matrix with each column being a data sample.
	 */
	protected RealMatrix X;
	
	/**
	 * A d x c projection matrix.
	 */
	protected RealMatrix W;
	
	/**
	 * Feed data for this feature selection algorithm.
	 * 
	 * @param X a d x n data matrix with each column being
	 *          a data sample
	 * 
	 */
	public void feedData(RealMatrix X) {
		this.X = X.copy();
	}
	
	/**
	 * Feed data for this feature selection algorithm.
	 * 
	 * @param data a d x n 2D {@code double} array with each
	 *             column being a data sample
	 * 
	 */
	public void feedData(double[][] data) {
		this.X = new BlockRealMatrix(data);
	}
	
	/**
	 * Do feature selection.
	 */
	public abstract void run();
	
	/**
	 * Get the data matrix.
	 * 
	 * @return a d x n data matrix
	 * 
	 */
	public RealMatrix getX() {
		return X;
	}
	
	/**
	 * Get the projection matrix.
	 * 
	 * @return a d x c projection matrix
	 * 
	 */
	public RealMatrix getW() {
		return W;
	}

}
