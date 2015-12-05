package jml.manifold;

import jml.matlab.utils.FindResult;
import jml.matlab.utils.SortResult;
import jml.options.GraphOptions;

import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import static jml.data.Data.*;
import static jml.kernel.Kernel.*;
import static jml.matlab.Matlab.*;
import static jml.utils.Time.*;

public class Manifold {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		String filePath = "CNN - DocTermCount.txt";
		RealMatrix X = loadMatrixFromDocTermCountFile(filePath);
		int NSample = Math.min(20, X.getColumnDimension());
		X = X.getSubMatrix(0, X.getRowDimension() - 1, 0, NSample - 1);
		System.out.println(String.format("%d samples loaded", X.getColumnDimension()));
		GraphOptions options = new GraphOptions();
		options.graphType = "nn";
		String type = options.graphType;
		double NN = options.graphParam;
		System.out.println(String.format("Graph type: %s with NN: %d", type, (int)NN));
		
		// Parameter setting for text data
		options.kernelType = "cosine";
		options.graphDistanceFunction = "cosine";
		
		// Parameter setting for image data
		/*options.kernelType = "rbf";
		options.graphDistanceFunction = "euclidean";*/
		
		options.graphNormalize = true;
		options.graphWeightType = "heat";
		
		boolean show = true && !false;
		
		// Test adjacency function - pass
		tic();
		String DISTANCEFUNCTION = options.graphDistanceFunction;
		RealMatrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		String adjacencyFilePath = "adjacency.txt";
		saveMatrix(adjacencyFilePath, A);
		if (show)
			display(A.getSubMatrix(0, 9, 0, 9));
		
		// Test laplacian function - pass
		tic();
		RealMatrix L = laplacian(X, type, options);
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		String LaplacianFilePath = "Laplacian.txt";
		saveMatrix(LaplacianFilePath, L);
		if (show)
			display(L.getSubMatrix(0, 9, 0, 9));
		
		// Test local learning regularization - pass
		NN = options.graphParam;
		String DISTFUNC = options.graphDistanceFunction;
		String KernelType = options.kernelType;
		double KernelParam = options.kernelParam;
		double lambda = 0.001;
		tic();
		RealMatrix LLR_text = calcLLR(X, NN, DISTFUNC, KernelType, KernelParam, lambda);
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		String LLRFilePath = "localLearningRegularization.txt";
		saveMatrix(LLRFilePath, LLR_text);
		if (show)
			display(LLR_text.getSubMatrix(0, 9, 0, 9));
		
	}
	
	/**
	 * Calculate the graph Laplacian of the adjacency graph of a data set
	 * represented as columns of a matrix X.
	 * 
	 * @param X data matrix with each column being a sample
	 * 
	 * @param type graph type, either "nn" or "epsballs"
	 * 
	 * @param options 
	 *        data structure containing the following fields
	 *        NN - integer if type is "nn" (number of nearest neighbors),
	 *             or size of "epsballs"
	 *        DISTANCEFUNCTION - distance function used to make the graph
	 *        WEIGHTTYPPE = "binary" | "distance" | "heat" | "inner"
	 * 	      WEIGHTPARAM = width for heat kernel
	 * 	      NORMALIZE = 0 | 1 whether to return normalized graph Laplacian or not
	 * 
	 * @return a sparse symmetric N x N matrix
	 * 
	 */
	public static RealMatrix laplacian(RealMatrix X, String type, GraphOptions options) {
		
		System.out.println("Computing Graph Laplacian...");
		
		double NN = options.graphParam;
		String DISTANCEFUNCTION = options.graphDistanceFunction;
		String WEIGHTTYPE = options.graphWeightType;
		double WEIGHTPARAM = options.graphWeightParam;
		boolean NORMALIZE = options.graphNormalize;
		
		if (WEIGHTTYPE.equals("inner") && !DISTANCEFUNCTION.equals("cosine"))
		    System.err.println("WEIGHTTYPE and DISTANCEFUNCTION mismatch.");
		
		// Calculate the adjacency matrix for DATA
		RealMatrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
		
		// W could be viewed as a similarity matrix
		RealMatrix W = A.copy();
		
		// Disassemble the sparse matrix
		FindResult findResult = find(A);
		int[] A_i = findResult.rows;
		int[] A_j = findResult.cols;
		double[] A_v = findResult.vals;
		
		/*HashMap<String, RealMatrix> map = find2(A);
		int[] A_i = doubleArray2IntArray(map.get("row").getColumn(0));
		int[] A_j = doubleArray2IntArray(map.get("col").getColumn(0));
		double[] A_v = map.get("val").getColumn(0);*/
		
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
		}
		
		RealMatrix D = sum(W, 2);
		RealMatrix L = null;
		if (!NORMALIZE)
			L = diag(D).subtract(W);
		else {
			// Normalized Laplacian
			D = diag(dotDivide(1, sqrt(D)));
			L = eye(size(W, 1)).subtract(D.multiply(W).multiply(D));
		}
		
		return L;
		
	}
	
	/**
	 * Compute the symmetric adjacency matrix of the data set represented as
	 * a real data matrix X. The diagonal elements of the sparse symmetric
	 * adjacency matrix are all zero indicating that a sample should not be
	 * a neighbor of itself. Note that in some cases, neighbors of a sample
	 * may coincide with the sample itself, we set eps for those entries in
	 * the sparse symmetric adjacency matrix.
	 * 
	 * @param X data matrix with each column being a feature vector
	 * 
	 * @param type graph type, either "nn" or "epsballs" ("eps")
	 * 
	 * @param param integer if type is "nn", real number if type is "epsballs" ("eps")
	 * 
	 * @param distFunc function mapping a (D x M) and a (D x N) matrix
     *                 to an M x N distance matrix (D: dimensionality)
     *                 either "euclidean" or "cosine"
     *                 
	 * @return a sparse symmetric N x N  matrix of distances between the
     *         adjacent points
     *         
	 */
	public static RealMatrix adjacency(RealMatrix X, String type, double param, String distFunc) {
		RealMatrix A = adjacencyDirected(X, type, param, distFunc);
		return max(A, A.transpose());
	}
	
	/**
	 * Compute the directed adjacency matrix of the data set represented as
	 * a real data matrix X. The diagonal elements of the sparse directed 
	 * adjacency matrix are all zero indicating that a sample should not be
	 * a neighbor of itself. Note that in some cases, neighbors of a sample
	 * may coincide with the sample itself, we set eps for those entries in
	 * the sparse directed adjacency matrix.
	 * 
	 * @param X data matrix with each column being a feature vector
	 * 
	 * @param type graph type, either "nn" or "epsballs" ("eps")
	 * 
	 * @param param integer if type is "nn", real number if type is "epsballs" ("eps")
	 * 
	 * @param distFunc function mapping a (D x M) and a (D x N) matrix
     *                 to an M x N distance matrix (D: dimensionality)
     *                 either "euclidean" or "cosine"
     *                 
	 * @return a sparse N x N matrix of distances between the
     *         adjacent points, not necessarily symmetric
     *         
	 */
	public static RealMatrix adjacencyDirected(RealMatrix X, String type, double param, String distFunc) {
		
		System.out.println("Computing directed adjacency graph...");
		
		int n = size(X, 2);
		
		if (type.equals("nn")) {
			System.out.println(String.format("Creating the adjacency matrix. Nearest neighbors, N = %d.", (int)param));
		} else if (type.equals("epsballs") || type.equals("eps")) {
			System.out.println(String.format("Creating the adjacency matrix. Epsilon balls, eps = %f.", param));
		} else {
			System.err.println("type should be either \"nn\" or \"epsballs\" (\"eps\")");
			System.exit(1);
		}
		
		RealMatrix A = new OpenMapRealMatrix(n, n);
		
		RealMatrix dt = null;
		for (int i = 0; i < n; i++) {
			
			if (i == 9) {
				int a = 0;
				a = a + 1;
			}
			
			if (distFunc.equals("euclidean")) {
				dt = euclidean(X.getColumnMatrix(i), X);
			} else if (distFunc.equals("cosine")) {
				dt = cosine(X.getColumnMatrix(i), X);
			}
			
			SortResult sortResult = sort(dt, 2);
			RealMatrix Z = sortResult.B;
			int[][] IX = sortResult.IX;
			
			if (type.equals("nn")) {
				for (int j = 0; j <= param; j++ ) {
					if (IX[0][j] != i)
						A.setEntry(i, IX[0][j], Z.getEntry(0, j) + eps);
				}
			} else if (type.equals("epsballs") || type.equals("eps")) {
				int j = 0;
				while (Z.getEntry(0, j) <= param) {
					if (IX[0][j] != i)
						A.setEntry(i, IX[0][j], Z.getEntry(0, j) + eps);
					j++;
				}
			}
			
			/*RealMatrix Z = dt;
			RealMatrix I = sort2(Z, 2);
			
			if (type.equals("nn")) {
				for (int j = 2; j <= param + 1; j++ ) {
					A.setEntry(i, (int)I.getEntry(0, j - 1), Z.getEntry(0, j - 1) + eps);
				}
			} else if (type.equals("epsballs") || type.equals("eps")) {
				int j = 2;
				while (Z.getEntry(0, j - 1) <= param) {
					A.setEntry(i, (int)I.getEntry(0, j - 1), Z.getEntry(0, j - 1) + eps);
					j++;
				}
			}*/
			
		}
		
		return A;
		
	}
	
	/**
	 * Compute the cosine distance matrix between column vectors in matrix A
	 * and column vectors in matrix B.
	 * 
	 * @param A data matrix with each column being a feature vector
	 * 
	 * @param B data matrix with each column being a feature vector
	 * 
	 * @return an n_A X n_B matrix with its (i, j) entry being the cosine
	 * distance between i-th feature vector in A and j-th feature
	 * vector in B, i.e.,
	 * ||A(:, i) - B(:, j)|| = 1 - A(:, i)' * B(:, j) / || A(:, i) || * || B(:, j)||
	 */
	public static RealMatrix cosine(RealMatrix A, RealMatrix B) {
		
		RealMatrix AA = sum(times(A, A));
		RealMatrix BB = sum(times(B, B));
		RealMatrix AB = A.transpose().multiply(B);
		RealMatrix C = times(scalarDivide(1, sqrt(kron(AA.transpose(), BB))), AB);
		return C.scalarMultiply(-1.0).scalarAdd(1.0);
		
	}
	
	/**
	 * Compute the Euclidean distance matrix between column vectors in matrix A
	 * and column vectors in matrix B.
	 * 
	 * @param A data matrix with each column being a feature vector
	 * 
	 * @param B data matrix with each column being a feature vector
	 * 
	 * @return an n_A X n_B matrix with its (i, j) entry being Euclidean
	 * distance between i-th feature vector in A and j-th feature
	 * vector in B, i.e., || X(:, i) - Y(:, j) ||_2
	 * 
	 */
	public static RealMatrix euclidean(RealMatrix A, RealMatrix B) {
		return l2Distance(A, B);
	}
	
	/**
	 * Compute local learning regularization matrix. Local learning
	 * regularization only depends on kernel selection, distance
	 * function, and neighborhood size.
	 * 
	 * @param X data matrix with each column being a feature vector
	 * 
	 * @param NN number of nearest neighbor
	 * 
	 * @param distFunc function mapping a (D x M) and a (D x N) matrix
     *        to an M x N distance matrix (D: dimensionality)
     *        either "euclidean" or "cosine"
     *        
	 * @param kernelType  'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam    --    | degree | sigma |    --
	 * 
	 * @param lambda graph regularization parameter
	 * 
	 * @return local learning regularization matrix
	 * 
	 */
	public static RealMatrix calcLLR(RealMatrix X,
			double NN, String distFunc, String kernelType,
			double kernelParam, double lambda) {
		
		String type = "nn";
		double param = NN;
		RealMatrix A = adjacencyDirected(X, type, param, distFunc);
		RealMatrix K = calcKernel(kernelType, kernelParam, X);
		
		int NSample = size(X, 2);
		int NFeature = size(X, 1);
		int n_i = (int)param;
		RealMatrix I_i = eye(n_i);
		RealMatrix I = eye(NSample);
		
		RealMatrix G = A.copy();
		
		int[] neighborIndices_i = null;
		RealMatrix neighborhood_X_i = null;
		RealMatrix K_i = null;
		RealMatrix k_i = null;
		RealMatrix x_i = null;
		RealMatrix alpha_i = null;
		// int[] IDs = double1DArray2Int1DArray(colon(0, NFeature - 1).getRow(0));
		int[] IDs = colon(0, NFeature - 1);
		for (int i = 0; i < NSample; i++) {
			neighborIndices_i = find(A.getRowVector(i));
			neighborhood_X_i = X.getSubMatrix(IDs, neighborIndices_i);
			K_i = K.getSubMatrix(neighborIndices_i, neighborIndices_i);
			x_i = X.getColumnMatrix(i);
			k_i = calcKernel(kernelType, kernelParam, neighborhood_X_i, x_i);
			alpha_i = mldivide(I_i.scalarMultiply(n_i * lambda).add(K_i), k_i);
			setSubMatrix(G, new int[]{i}, neighborIndices_i, alpha_i.transpose());
		}
		
		RealMatrix T = G.subtract(I);
		RealMatrix L = T.transpose().multiply(T);
		
		return L;
		
	}

}
