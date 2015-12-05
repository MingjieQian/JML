package jml.subspace;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

public abstract class DimensionalityReduction {
	
	/**
	 * d x n data matrix.
	 */
	protected RealMatrix X;
	
	/**
	 * Reduced r x n data matrix.
	 */
	protected RealMatrix R;
	
	/**
	 * Reduced dimensionality.
	 */
	protected int r;
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public DimensionalityReduction(int r) {
		this.r = r;
	}
	
	/**
	 * Do dimensionality reduction.
	 */
	public abstract void run();
	
	public void feedData(RealMatrix X) {
		this.X = X;
	}
	
	public void feedData(double[][] data) {
		this.X = new BlockRealMatrix(data);
	}
	
	public RealMatrix getDataMatrix() {
		return X;
	}
	
	public RealMatrix getReducedDataMatrix() {
		return R;
	}
	
	public void setReducedDimensionality(int r) {
		this.r = r;
	}
	
	public int getReducedDimensionality() {
		return r;
	}

}
