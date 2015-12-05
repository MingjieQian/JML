package jml.graph.cut;

import org.apache.commons.math.linear.RealMatrix;

public abstract class GraphCut {
	
	/**
	 * Number of clusters on the graph.
	 */
	private int K;
	
	/**
	 * Weight matrix on the graph.
	 */
	private RealMatrix W;
	
	/**
	 * Cluster indicator matrix.
	 */
	private RealMatrix Q;
	
	/**
	 * Constructor.
	 * 
	 * @param K number of clusters on the graph.
	 * 
	 */
	public GraphCut(int K) {
		this.setK(K);
	}
	
	/**
	 * Initialization.
	 */
	public abstract void initialize();
	
	/**
	 * Do graph cut/clustering.
	 */
	public abstract void cut();
	
	/**
	 * Get the weight matrix on the graph.
	 * 
	 * @return an N x N weight matrix
	 * 
	 */
	public RealMatrix getWeightMatrix() {
		return W;
	}
	
	/**
	 * Get the N x K cluster indicator matrix.
	 * 
	 * @return an N x K cluster indicator matrix
	 * 
	 */
	public RealMatrix getIndicatorMatrix() {
		return Q;
	}

	/**
	 * Get the number of clusters.
	 * 
	 * @return number of clusters on the graph
	 * 
	 */
	public int getK() {
		return K;
	}

	/**
	 * Set the number of clusters.
	 * 
	 * @param K number of clusters to be set on the graph
	 * 
	 */
	public void setK(int K) {
		this.K = K;
	}

}
