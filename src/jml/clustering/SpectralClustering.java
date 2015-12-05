package jml.clustering;

import static jml.manifold.Manifold.adjacencyDirected;
import static jml.matlab.Matlab.diag;
import static jml.matlab.Matlab.display;
import static jml.matlab.Matlab.dotDivide;
import static jml.matlab.Matlab.eigs;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.find;
import static jml.matlab.Matlab.full;
import static jml.matlab.Matlab.max;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.sqrt;
import static jml.matlab.Matlab.sum;
import jml.matlab.utils.FindResult;
import jml.options.ClusteringOptions;
import jml.options.KMeansOptions;
import jml.options.SpectralClusteringOptions;

import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation for spectral clustering.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 4th, 2013
 */
public class SpectralClustering extends Clustering {

	public SpectralClusteringOptions options;
	
	public SpectralClustering() {
		super();
		options = new SpectralClusteringOptions();
	}

	public SpectralClustering(int nClus) {
		super(nClus);
		options = new SpectralClusteringOptions(nClus);
	}

	public SpectralClustering(ClusteringOptions options) {
		super(options);
		this.options = new SpectralClusteringOptions(options);
	}
	
	public SpectralClustering(SpectralClusteringOptions options) {
		this.options = options;
	}
	
	
	/**
	 * For spectral clustering, we don't need initialization in the
	 * current implementation.
	 */
	@Override
	public void initialize(RealMatrix G0) {
	}

	@Override
	public void clustering() {
		
		RealMatrix X = dataMatrix;
		String TYPE = options.graphType;
		double PARAM = options.graphParam;
		PARAM = Math.ceil(Math.log(size(X, 2)) + 1);
		if (PARAM == size(X, 2))
			PARAM--;
		String DISTANCEFUNCTION = options.graphDistanceFunction;
		RealMatrix A = adjacencyDirected(X, TYPE, PARAM, DISTANCEFUNCTION);
	
		RealMatrix Z = max(A, 2).get("val");
		double WEIGHTPARAM = options.graphWeightParam;
		WEIGHTPARAM = sum(Z).getEntry(0, 0) / Z.getRowDimension();
	    
		A = max(A, A.transpose());
		
		// W could be viewed as a similarity matrix
		RealMatrix W = A.copy();

		// Disassemble the sparse matrix
		FindResult findResult = find(A);
		int[] A_i = findResult.rows;
		int[] A_j = findResult.cols;
		double[] A_v = findResult.vals;
		
		String WEIGHTTYPE = options.graphWeightType;
		
		if (WEIGHTTYPE.equals("distance")) {
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], A_v[i]);
			}
		} else if (WEIGHTTYPE.equals("inner")) {
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], 1 - A_v[i] / 2);
			}
		} else if (WEIGHTTYPE.equals("binary")) {
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], 1);
			}
		} else if (WEIGHTTYPE.equals("heat")) {
			double t = -2 * WEIGHTPARAM * WEIGHTPARAM;
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], 
			               Math.exp(A_v[i] * A_v[i] / t));
			}
		} else {
			System.err.println("Unknown Weight Type.");
			System.exit(1);
		}
		
		// Construct L_sym
		RealMatrix D = sum(W, 2);
		RealMatrix Dsqrt = diag(dotDivide(1, sqrt(D)));
		RealMatrix L_sym = eye(size(W, 1)).subtract(Dsqrt.multiply(W).multiply(Dsqrt));
		// System.out.println(L_sym.getEntry(34, 110));
		RealMatrix eigRes[] = eigs(L_sym, this.options.nClus, "sm");
		RealMatrix V = eigRes[0];// display(V);
		RealMatrix U = Dsqrt.multiply(V);// display(U);
		
		KMeansOptions kMeansOptions = new KMeansOptions();
		kMeansOptions.nClus = this.options.nClus;
		kMeansOptions.maxIter = this.options.maxIter;
		kMeansOptions.verbose = this.options.verbose;
		
		KMeans KMeans = new KMeans(kMeansOptions);
		KMeans.feedData(U.transpose());
		KMeans.initialize(null);
		KMeans.clustering();
		this.indicatorMatrix = KMeans.indicatorMatrix;
		
		System.out.println("Spectral clustering complete.");
	
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		long start = System.currentTimeMillis();
		
		int nClus = 2;
		boolean verbose = false;
		int maxIter = 100;
		String graphType = "nn";
		double graphParam = 6;
		String graphDistanceFunction = "euclidean";
		String graphWeightType = "heat";
		double graphWeightParam = 1;
		ClusteringOptions options = new SpectralClusteringOptions(
				nClus,
				verbose,
				maxIter,
				graphType,
				graphParam,
				graphDistanceFunction,
				graphWeightType,
				graphWeightParam);
		
		Clustering spectralClustering = new SpectralClustering(options);
		
		/*String dataMatrixFilePath = "CNN - DocTermCount.txt";
		RealMatrix X = Data.loadMatrix(dataMatrixFilePath);*/
		
		double[][] data = { {3.5, 4.4, 1.3},
			    		    {5.3, 2.2, 0.5},
			                {0.2, 0.3, 4.1},
			                {-1.2, 0.4, 3.2} };
		/*RealMatrix X = new BlockRealMatrix(data);
		X = X.transpose();*/
		
		spectralClustering.feedData(data);
		spectralClustering.clustering(null);
		display(full(spectralClustering.getIndicatorMatrix()));
		
		/*String labelFilePath = "GroundTruth.txt";
		RealMatrix groundTruth = Data.loadMatrix(labelFilePath);
		getAccuracy(spectralClustering.indicatorMatrix, groundTruth);*/
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);

	}

}
