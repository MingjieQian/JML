package jml.clustering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import jml.options.ClusteringOptions;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/**
 * Abstract class for clustering algorithms.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 3rd, 2013
 */
public abstract class Clustering {

	/**
	 * Number of clusters.
	 */
	public int nClus;
	
	/**
	 * Number of features.
	 */
	public int nFeature;
	
	/**
	 * Number of samples.
	 */
	public int nSample;
	
	/**
	 * Data matrix (nFeature x nSample), each column is a feature vector
	 */
	protected RealMatrix dataMatrix;
	
	/**
	 * Cluster indicator matrix (nSample x nClus).
	 */
	protected RealMatrix indicatorMatrix;
	
	/**
	 * Cluster matrix (nFeature x nClus), column i is the projector for class i.
	 */
	protected RealMatrix centers;
	
	/**
	 * Default constructor for this clustering algorithm.
	 */
	public Clustering() {
		this.nClus = 0;
	}
	
	/**
	 * Constructor for this clustering algorithm initialized with options
	 * wrapped in a {@code ClusteringOptions} object.
	 * 
	 * @param clusteringOptions clustering options
	 * 
	 */
	public Clustering(ClusteringOptions clusteringOptions) {
		this.nClus = clusteringOptions.nClus;
	}
	
	/**
	 * Constructor for this clustering algorithm given number of
	 * clusters to be set.
	 * 
	 * @param nClus number of clusters
	 * 
	 */
	public Clustering(int nClus) {
		if (nClus < 1) {
			System.err.println("Number of clusters less than one!");
			System.exit(1);
		}
		this.nClus = nClus;
	}
	
	/**
	 * Feed training data for this clustering algorithm.
	 * 
	 * @param dataMatrix a d x n data matrix with each column being
	 *                   a data example
	 * 
	 */
	public void feedData(RealMatrix dataMatrix) {
		this.dataMatrix = dataMatrix;
		nFeature = dataMatrix.getRowDimension();
		nSample = dataMatrix.getColumnDimension();
	}
	
	/**
	 * Feed training data for this feature selection algorithm.
	 * 
	 * @param data a d x n 2D {@code double} array with each
	 *             column being a data example
	 * 
	 */
	public void feedData(double[][] data) {
		this.feedData(new BlockRealMatrix(data));
	}
	
	/**
	 * Initialize the indicator matrix.
	 * 
	 * @param G0 initial indicator matrix
	 * 
	 */
	public void initialize(RealMatrix G0) {
		
		if (G0 != null) {
			this.indicatorMatrix = G0;
			return;
		}
		List<Integer> indList = new ArrayList<Integer>();
		for (int i = 0; i < nSample; i++) {
			indList.add(i);
		}
		
		Random rdn = new Random(System.currentTimeMillis());
		Collections.shuffle(indList, rdn);
		
		indicatorMatrix = new OpenMapRealMatrix(nSample, nClus);
		
		for (int i = 0; i < nClus; i++) {
			indicatorMatrix.setEntry(indList.get(i), i, 1);
		}
		
	}
	
	/**
	 * Initialize the indicator matrix.
	 * 
	 * @param indicators 1D {@code int} array
	 *//*
	public void initialize(int[] indicators) {
		
		if (indicators != null) {
			indicatorMatrix = new OpenMapRealMatrix(nSample, nClus);
			for (int i = 0; i < indicators.length; i++) {
				indicatorMatrix.setEntry(i, indicators[i], 1);
			}
			return;
		}
		List<Integer> indList = new ArrayList<Integer>();
		for (int i = 0; i < nSample; i++) {
			indList.add(i);
		}
		
		Random rdn = new Random(System.currentTimeMillis());
		Collections.shuffle(indList, rdn);
		
		indicatorMatrix = new OpenMapRealMatrix(nSample, nClus);
		
		for (int i = 0; i < nClus; i++) {
			indicatorMatrix.setEntry(indList.get(i), i, 1);
		}
		
	}*/
	
	/**
	 * Do clustering. Please call initialize() before
	 * using this method. 
	 */
	public abstract void clustering();
	
	/**
	 * Do clustering with a specified initializer. Please use null if
	 * you want to use random initialization.
	 * 
	 * @param G0 initial indicator matrix, if null random initialization
	 *           will be used
	 */
	public void clustering(RealMatrix G0) {
		initialize(G0);
		clustering();
	}
	
	/**
	 * Fetch data matrix.
	 * 
	 * @return a d x n data matrix
	 * 
	 */
	public RealMatrix getData() {
		return dataMatrix;
	}
	
	/**
	 * Get cluster centers.
	 * 
	 * @return a d x K basis matrix
	 * 
	 */
	public RealMatrix getCenters() {
		return centers;
	}
	
	/**
	 * Get cluster indicator matrix.
	 * 
	 * @return an n x K cluster indicator matrix
	 * 
	 */
	public RealMatrix getIndicatorMatrix() {
		return indicatorMatrix;
	}
	
	/**
	 * Evaluating the clustering performance of this clustering algorithm
	 * by using the ground truth.
	 * 
	 * @param G predicted cluster indicator matrix
	 * 
	 * @param groundTruth true cluster assignments
	 * 
	 * @return evaluation metrics
	 * 
	 */
	public static double getAccuracy(RealMatrix G, RealMatrix groundTruth) {
		// To do
		System.out.println("Sorry, this function has not been implemented yet...");
		return 0;
	}
	
}
