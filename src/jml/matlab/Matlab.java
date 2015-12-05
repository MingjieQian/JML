package jml.matlab;

import static java.lang.System.out;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;

import jml.matlab.utils.FindResult;
import jml.matlab.utils.Pair;
import jml.matlab.utils.Power;
import jml.matlab.utils.SingularValueDecompositionImpl;
import jml.matlab.utils.SortResult;
import jml.operation.ArrayOperation;
import jml.random.MultivariateGaussianDistribution;

import org.apache.commons.math.FunctionEvaluationException;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.InvalidMatrixException;
import org.apache.commons.math.linear.LUDecompositionImpl;
import org.apache.commons.math.linear.MatrixIndexException;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.QRDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;

/**
 * Matlab provides some frequently used Matlab matrix functions 
 * such as sort, sum, max, min, kron, vec, repmat, reshape, and 
 * colon.
 * 
 * @version 1.0 Mar. 9th, 2012
 * @version 2.0 Dec. 29th, 2012
 * @version 3.0 Jan. 1st, 2013
 * 
 * @author Mingjie Qian
 */
public class Matlab {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data0 = {{0, 2, 3, 4}, {2, 0, 4, 5}, {3, 4.1, 5, 6}, {2, 7, 1, 6}};
		RealMatrix X = new BlockRealMatrix(data0);
		display(X);
		display(linearIndexing(X, new int[] {0, 3, 5, 8}));
		linearIndexingAssignment(X, new int[] {0, 3, 5, 8}, new BlockRealMatrix(new double[][]{{2.9}, {3.9}, {4.9}, {5.9}}));
		display(X);
		linearIndexingAssignment(X, new int[] {0, 3, 5, 8}, 99);
		display(X);
		display(getRows(X, new int[]{1,2}));
		display(getRows(X, 1, 2));
		display(norm(X, 1));
		display(norm(X, 2));
		display(norm(X, inf));
		display(norm(X, "fro"));
		X = eye(10);
		X.setEntry(5, 6, 1.2);
		display("X:");
		display(X);
		display(sumAll(X));
		RealMatrix X1 = new OpenMapRealMatrix(3, 4);
		X1.setSubMatrix(X.getSubMatrix(colon(1, 3), colon(0, 3)).getData(), 0, 0);
		display(X1);
		setSubMatrix(X1, colon(0, 2), colon(0, 3), X.getSubMatrix(colon(1, 3), colon(0, 3)));
		display(X1);
		RealMatrix Y1 = lt(X, 5);
		display(Y1);
		display(vertcat(X, Y1));
		display(cat(1, X, Y1));
		display(cat(2, X, Y1));
		display(horzcat(X, Y1));
		display(logicalIndexing(X, Y1));
		logicalIndexingAssignment(X, Y1, logicalIndexing(X, Y1));
		display(X);
		X = X.add(X.transpose());
		out.println("X =\n");
		display(X);
		RealMatrix eigRes[] = eigs(X, 3, "lm");
		out.println("V =\n");
		display(eigRes[0]);
		out.println("D =\n");
		display(eigRes[1]);
		out.println("V * D * V' =\n");
		display(eigRes[0].multiply(eigRes[1].multiply(eigRes[0].transpose())));
		
		double[][] aData = { {1d,2d}, {3d,4d}};
		double[][] bData = { {3d,6d}, {4d,7d}};
		RealMatrix A = new BlockRealMatrix(aData);
		RealMatrix B = new BlockRealMatrix(bData);
		
		display(A);
		display(B);
		display(ldivide(A, B));
		display(rdivide(A, B));
		display(mldivide(A, B));
		display(mrdivide(A, B));
		
		display(ldivide(A, 2));
		display(rdivide(A, 2));
		display(ldivide(2, B));
		display(rdivide(2, B));

		RealVector V = new ArrayRealVector(new double[]{0, 1, -2, 2});
		int[] indices = find(V);
		for (int i = 0; i < indices.length; i++) {
			System.out.println(String.format("Index: %d", indices[i]));
		}
		System.out.println("Original vector:");
		display(V);
		int[] IX = sort(V, "descend");
		for (int i = 0; i < IX.length; i++) {
			System.out.println(String.format("Value: %f, index: %d", V.getEntry(i), IX[i]));
		}
		System.out.println("Sorted vector in a descending order:");
		display(V);

		RealMatrix K = V.outerProduct(V);
		display(K);

		RealMatrix K_i = K.getSubMatrix(indices, indices);
		printMatrix(K_i);

		K.getSubMatrix(indices, indices).setEntry(0, 0, 100.0);
		printMatrix(K);

		RealMatrix I = Matlab.eye(2);
		double[][] data = I.getData();
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[0].length; j++) {
				System.out.print(String.format("%.4f", data[i][j]) + "\t");
			}
			System.out.println();
		}
		System.out.println();

		int[] rowIdxVec = new int[]{0, 1, 1, 0};
		int[] colIdxVec = new int[]{0, 1};

		//colIdxVec = new ArrayRealVector(colIdxVec).mapAdd(-1).getData();
		double[][] res = new double[rowIdxVec.length][];
		for (int i = 0; i < rowIdxVec.length; i++) {
			res[i] = new double[colIdxVec.length];
		}
		I.copySubMatrix(rowIdxVec, colIdxVec, res);
		RealMatrix T = new Array2DRowRealMatrix(res);
		printMatrix(T);

		printMatrix(find2(T).get("row"));

		printMatrix(sum(T));
		printMatrix(sum(sum(T)));

		printMatrix(max(T));
		printMatrix(max(max(T)));

		printMatrix(min(T));
		printMatrix(min(min(T)));

		printMatrix(T);
		T = getTFIDF(T);
		printMatrix(T);
		T = normalizeByColumns(T);
		printMatrix(T);

		System.out.println(1.0 / Double.POSITIVE_INFINITY );
		System.out.println(Math.exp(1000));
		System.out.println(1.0 / Math.exp(1000) );
		System.out.println(1.0 / 0.0);

		printMatrix(linspace(1, 4, 8));
		printMatrix(colon(1, 1.2, 4));

		double[][] dataA = { {10d,-5d,0d,3d}, {2d,0d,1d,2d}, {1d,6d,0d,5d}};

		A = new Array2DRowRealMatrix(dataA);

		printMatrix(A);
		printMatrix(subplus(A));
		printMatrix(normalizeByColumns(A));
		printMatrix(sigmoid(A));

		printMatrix(A);
		printMatrix(A.scalarMultiply(2.0));
		printMatrix(A);
		printMatrix(A.scalarAdd(2.0));
		printMatrix(A);
		printMatrix(reshape(A, new int[]{2, 6}));
		printMatrix(reshape(vec(A).getColumnVector(0), new int[]{2, 6}));

		printMatrix(reshape(A, new int[]{1, 12}));
		printMatrix(reshape(A, new int[]{12, 1}));
		printMatrix(vec(A));

		printMatrix(times(ones(size(A)), zeros(size(A))));

		double[][] dataB = { {1d,3d}, {2d,4d}};
		B = new Array2DRowRealMatrix(dataB);

		printMatrix(A);
		printMatrix(B);
		printMatrix(kron(B, A));

		double[][] matrixData = { {1d,5d,0d}, {2d,0d,1d}};

		double[][] XData = { {1d,5d,0d}, {2d,0d,1d}};
		double[][] YData = { {1d,5d}, {2d,0d}};

		X = new Array2DRowRealMatrix(XData);
		RealMatrix Y = new Array2DRowRealMatrix(YData);

		System.out.println("X:");
		printMatrix(X);

		System.out.println("Y:");
		printMatrix(Y);

		System.out.println("l2 distance matrix between X and Y:");
		printMatrix(l2Distance(X, Y));

		RealMatrix M = new Array2DRowRealMatrix(matrixData);

		RealMatrix sortedMatrix = M.copy();

		System.out.println("Original matrix:");
		printMatrix(M);

		System.out.println("Index matrix along columns in ascending order:");
		printMatrix(sort2(sortedMatrix, 1, "ascend"));

		System.out.println("Sorted matrix along columns in ascending order:");
		printMatrix(sortedMatrix);

		sortedMatrix = M.copy();

		System.out.println("Index matrix along rows in descending order:");
		printMatrix(sort2(sortedMatrix, 2, "descend"));

		System.out.println("Sorted matrix along rows in descending order:");
		printMatrix(sortedMatrix);

		System.out.println("TFIDF matrix:");
		printMatrix(getTFIDF(M));

		System.out.println("Normalized TFIDF matrix:");
		printMatrix(normalizeByColumns(getTFIDF(M)));

		System.out.println("Original matrix:");
		printMatrix(M);

		System.out.println("Summation along columns:");
		printMatrix(sum(M, 1));

		System.out.println("Summation along rows:");
		printMatrix(sum(M, 2));

		printMatrix(min(M, 1).get("val"));
		printMatrix(min(M, 1).get("idx"));

		printMatrix(max(M, 2).get("val"));
		printMatrix(max(M, 2).get("idx"));

		printMatrix(max(max(M, 1).get("val"), 2).get("val"));
		printMatrix(max(max(M, 1).get("val"), 2).get("idx"));

		System.out.println("1000 x 1000 matrix multiplication test.");
		long start = System.currentTimeMillis();

		RealMatrix L = ones(1000, 1000);
		RealMatrix R = ones(1000, 1000);
		L.multiply(R);

		System.out.format(System.getProperty("line.separator") + "Elapsed time: %.3f seconds.\n", 
				(System.currentTimeMillis() - start) / 1000F);

	}

	/**
	 * A constant holding the smallest positive
	 * nonzero value of type double, 2-1074.
	 */
	public static double eps = Double.MIN_VALUE;
	
	/**
	 * A constant holding the positive infinity of type double.
	 */
	public static double inf = Double.POSITIVE_INFINITY;

	/**
	 * Compute eigenvalues and eigenvectors of a symmetric real matrix.
	 * 
	 * @param A a symmetric real matrix
	 * 
	 * @param K number of eigenvalues selected
	 * 
	 * @param sigma either "lm" (largest magnitude) or "sm" (smallest magnitude)
	 * 
	 * @return a matrix array [V, D], V is the selected K eigenvectors (normalized 
	 *         to 1), and D is a diagonal matrix holding selected K eigenvalues.
	 *         
	 */
	public static RealMatrix[] eigs(RealMatrix A, int K, String sigma) {
		
		EigenDecompositionImpl eigImpl = new EigenDecompositionImpl(A, 1e-6);
		RealMatrix eigV = eigImpl.getV();
		RealMatrix eigD = eigImpl.getD();
		
		int N = A.getRowDimension();
		RealMatrix[] res = new RealMatrix[2];
		
		RealVector eigenValueVector = diag(eigD).getColumnVector(0);
		RealMatrix eigenVectors = null;
		if (sigma.equals("lm")) {
			eigenValueVector = eigenValueVector.getSubVector(0, K);
			eigenVectors = eigV.getSubMatrix(0, N - 1, 0, K - 1);
		} else if (sigma.equals("sm")) {
			eigenValueVector = eigenValueVector.getSubVector(N - K, K);
			sort(eigenValueVector, "ascend");
			eigenVectors = new BlockRealMatrix(N, K);
			for(int i = N - 1, k = 0; k < K ; i--, k++) {
				eigenVectors.setColumnVector(k, eigV.getColumnVector(i));
			}
		} else {
			System.err.println("sigma should be either \"lm\" or \"sm\"");
			System.exit(-1);
		}
		
		res[0] = eigenVectors;
		res[1] = diag(eigenValueVector);
		
		return res;
		
	}
	
	/**
	 * Compute the "economy size" matrix singular value decomposition.
	 * 
	 * @param A a real matrix
	 * 
	 * @return a matrix array [U, S, V] where U is left orthonormal matrix, S is a 
	 * 		   a diagonal matrix, and V is the right orthonormal matrix such that 
	 *         A = U * S * V'
	 * 
	 */
	public static RealMatrix[] svd(RealMatrix A) {
		
		SingularValueDecompositionImpl svdImpl = new SingularValueDecompositionImpl(A);
		RealMatrix U = svdImpl.getU();
		RealMatrix S = svdImpl.getS();
		RealMatrix V = svdImpl.getV();
		
		RealMatrix[] res = new RealMatrix[3];
		res[0] = U;
		res[1] = S;
		res[2] = V;
		
		return res;
		
	}
	
	/**
	 * The rank function provides an estimate of the number of 
	 * linearly independent rows or columns of a full matrix.
	 * </br>
	 * k = rank(A) returns the number of singular values of A 
	 * that are larger than the default tolerance, max(size(A))*1e-16.
	 * 
	 * @param A a real matrix
	 * 
	 * @return the estimated rank of A
	 * 
	 */
	public static int rank(RealMatrix A) {
		RealMatrix S = diag(svd(A)[1]);
		double tol = Math.max(size(A, 1), size(A, 2)) * 1e-10;
		int r = (int) sumAll(gt(S, tol));
		return r;
	}
	
	/**
	 * The rank function provides an estimate of the number of 
	 * linearly independent rows or columns of a full matrix.
	 * </br>
	 * k = rank(A,tol) returns the number of singular values 
	 * of A that are larger than tol.
	 * 
	 * @param A a real matrix
	 * 
	 * @param tol tolerance to estimate the rank of A
	 * 
	 * @return the estimated rank of A
	 * 
	 */
	public static int rank(RealMatrix A, double tol) {
		RealMatrix S = diag(svd(A)[1]);
		int r = (int) sumAll(gt(S, tol));
		return r;
	}
	
	/**
	 * Random permutation. 
	 * </br>
	 * randperm(n) returns a random permutation of the integers 1:n.
	 * 
	 * @param n an integer
	 * 
	 * @return randperm(n)
	 */
	public static int[] randperm(int n) {
		
		int[] res = new int[n];
		
		Set<Integer> leftSet = new TreeSet<Integer>();
		for (int i = 0; i < n; i++) {
			leftSet.add(i);
		}
		
		Random generator = new Random();
		for (int i = 0; i < n; i++) {
			double[] uniformDist = ArrayOperation.allocateVector(n - i, 1.0 / (n - i));
			
			double rndRealScalor = generator.nextDouble();
			double sum = 0;
			for (int j = 0, k = 0; j < n; j++) {
				if (!leftSet.contains(j))
					continue;
				sum += uniformDist[k];
				if (rndRealScalor <= sum) {
					res[i] = j + 1;
					leftSet.remove(j);
					break;
				} else {
					k++;
				}
			}
		}
		
		return res;

	}


	/**
	 * Find nonzero elements and return their indices.
	 * 
	 * @param V a real vector
	 * 
	 * @return an integer array of indices of nonzero elements of V
	 * 
	 */
	public static int[] find(RealVector V) {

		RealMatrix X = new Array2DRowRealMatrix(V.getData());
		double[] indices = find2(X).get("row").getColumn(0);
		int[] res = new int[indices.length];
		for (int i = 0; i < indices.length; i++)
			res[i] = (int) indices[i];

		return res;

	}

	/**
	 * Find nonzero elements and return their value, row and column indices.
	 * 
	 * @param A a real matrix
	 * 
	 * @return a {@code FindResult} data structure which has three instance
	 * data members:<br/>
	 * rows: row indices array for non-zero elements of a matrix<br/>
	 * cols: column indices array for non-zero elements of a matrix<br/>
	 * vals: values array for non-zero elements of a matrix<br/>
	 *         
	 */
	public static FindResult find(RealMatrix A) {

		HashMap<String, RealMatrix> map = find2(A);
		int[] rows = double1DArray2Int1DArray(map.get("row").getColumn(0));
		int[] cols = double1DArray2Int1DArray(map.get("col").getColumn(0));
		double[] vals = map.get("val").getColumn(0);

		return new FindResult(rows, cols, vals);

	}

	/**
	 * Find nonzero elements and return their value, row and column indices.
	 * 
	 * @param A a matrix
	 * 
	 * @return a {@code HashMap<String, RealMatrix>}.
	 * <p>
	 * "row": a column matrix recording row indices <br/>
	 * "col": a column matrix recording dolumn indices <br/>
	 * "val": a column matrix recording non-zero values <br/>
	 * </p>
	 *         
	 */
	public static HashMap<String, RealMatrix> find2(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		ArrayList<Integer> rowIdxList = new ArrayList<Integer>();
		ArrayList<Integer> colIdxList = new ArrayList<Integer>();
		ArrayList<Double> valList = new ArrayList<Double>();
		for (int j = 0; j < nCol; j++) {
			for (int i = 0; i < nRow; i++) {
				if (A.getEntry(i, j) != 0) {
					rowIdxList.add(i);
					colIdxList.add(j);
					valList.add(A.getEntry(i, j));
				}
			}
		}

		RealMatrix rowIdxMatrix = new Array2DRowRealMatrix(rowIdxList.size(), 1);
		RealMatrix colIdxMatrix = new Array2DRowRealMatrix(colIdxList.size(), 1);
		RealMatrix valMatrix = new Array2DRowRealMatrix(valList.size(), 1);
		for (int i = 0; i < rowIdxList.size(); i++) {
			rowIdxMatrix.setEntry(i, 0, rowIdxList.get(i));
			colIdxMatrix.setEntry(i, 0, colIdxList.get(i));
			valMatrix.setEntry(i, 0, valList.get(i));
		}

		HashMap<String, RealMatrix> res = new HashMap<String, RealMatrix>();
		res.put("row", rowIdxMatrix);
		res.put("col", colIdxMatrix);
		res.put("val", valMatrix);

		return res;

	}

	/**
	 * Compute the l2 distance matrix between column vectors in matrix X
	 * and column vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j)th entry being the l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, , i.e., || X(:, i) - Y(:, j) ||_2
	 * 
	 */
	public static RealMatrix l2Distance(RealMatrix X, RealMatrix Y) {

		return sqrt(l2DistanceSquare(X, Y));

	}

	/**
	 * Compute the squared l2 distance matrix between column vectors in matrix X
	 * and column vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j) entry being the squared l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X(:, i) - Y(:, j) ||_2^2
	 * 
	 */
	public static RealMatrix l2DistanceSquare(RealMatrix X, RealMatrix Y) {

		int nX = X.getColumnDimension();
		int nY = Y.getColumnDimension();

		RealMatrix dist = new BlockRealMatrix(nX, nY);

		dist = sum(times(X, X), 1).transpose().multiply(ones(1, nY)).add( 
				ones(nX, 1).multiply(sum(times(Y, Y), 1))).subtract(
						X.transpose().multiply(Y).scalarMultiply(2d));
		
		RealMatrix I = lt(dist, 0);
		logicalIndexingAssignment(dist, I, 0);

		return dist;

	}

	/**
	 * Calculate the element-wise multiplication of two matrices X and Y.
	 * 
	 * @param X
	 *        Matrix X.
	 *        
	 * @param Y
	 *        Matrix Y.
	 *        
	 * @return X .* Y
	 */
	public static RealMatrix times(RealMatrix X, RealMatrix Y) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		
		if (dX == 1 && nX == 1) {
			return times(X.getEntry(0, 0), Y);
		} else if (dY == 1 && nY == 1) {
			return times(X, Y.getEntry(0, 0));
		}
		
		if (nX != nY || dX != dY) {
			System.err.println("The operands for Hadmada product should be of same shapes!");
		}

		RealMatrix res = null;
		if (X instanceof OpenMapRealMatrix || Y instanceof OpenMapRealMatrix) {
			res = new OpenMapRealMatrix(dX, nX);
		} else {
			res = X.copy();
		}

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) * Y.getEntry(i, j));
			}
		}

		return res;
		
	}
	
	/**
	 * Matrix multiplication.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X * Y
	 * 
	 */
	public static RealMatrix mtimes(RealMatrix X, RealMatrix Y) {
		return X.multiply(Y);
	}
	
	/**
	 * Matrix multiplication.
	 * 
	 * @param X a matrix
	 * 
	 * @param v a real scalar
	 * 
	 * @return X * v
	 * 
	 */
	public static RealMatrix mtimes(RealMatrix X, double v) {
		return times(X, v);
	}
	
	/**
	 * Matrix multiplication.
	 * 
	 * @param v a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return X * v
	 * 
	 */
	public static RealMatrix mtimes(double v, RealMatrix X) {
		return times(v, X);
	}
	
	/**
	 * Array multiplication between a scalar and a matrix.
	 * 
	 * @param v a real number
	 * 
	 * @param A a matrix
	 * 
	 * @return v * A
	 * 
	 */
	public static RealMatrix times(double v, RealMatrix A) {
		return A.scalarMultiply(v);
	}
	
	/**
	 * Array multiplication between a matrix and a scalar.
	 * 
	 * @param A a matrix
	 * 
	 * @param v a real number
	 * 
	 * @return A * v
	 * 
	 */
	public static RealMatrix times(RealMatrix A, double v) {
		return A.scalarMultiply(v);
	}
	
	/**
	 * Scalar multiplication between two scalars a and b.
	 * 
	 * @param a a real scalar
	 * 
	 * @param b a real scalar
	 * 
	 * @return a * b
	 * 
	 */
	public static double times(double a, double b) {
		return a * b;
	}
	
	/**
	 * Array addition between two matrices.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X + Y
	 * 
	 */
	public static RealMatrix plus(RealMatrix X, RealMatrix Y) {
		return X.add(Y);
	}
	
	/**
	 * Array addition between a matrix and a scalar.
	 * 
	 * @param A a matrix
	 * 
	 * @param v a scalar
	 * 
	 * @return A + v
	 * 
	 */
	public static RealMatrix plus(RealMatrix A, double v) {
		return A.scalarAdd(v);
	}
	
	/**
	 * Array addition between a scalar and a matrix.
	 * 
	 * @param v a scalar
	 * 
	 * @param A a matrix
	 * 
	 * @return v + A
	 * 
	 */
	public static RealMatrix plus(double v, RealMatrix A) {
		return A.scalarAdd(v);
	}
	
	/**
	 * Addition between two scalars a and b.
	 * 
	 * @param a a scalar
	 * 
	 * @param b a scalar
	 * 
	 * @return a + b
	 * 
	 */
	public static double plus(double a, double b) {
		return a + b;
	}
	
	/**
	 * Array subtraction between two matrices.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X - Y
	 * 
	 */
	public static RealMatrix minus(RealMatrix X, RealMatrix Y) {
		return X.subtract(Y);
	}
	
	/**
	 * Array subtraction between a matrix and a scalar.
	 * 
	 * @param A a matrix
	 * 
	 * @param v a scalar
	 * 
	 * @return A - v
	 * 
	 */
	public static RealMatrix minus(RealMatrix A, double v) {
		return A.scalarAdd(-v);
	}
	
	/**
	 * Array subtraction between a scalar and a matrix.
	 * 
	 * @param v a scalar
	 * 
	 * @param A a matrix
	 * 
	 * @return v - A
	 * 
	 */
	public static RealMatrix minus(double v, RealMatrix A) {
		return A.scalarAdd(-v).scalarMultiply(-1);
	}
	
	/**
	 * Subtraction between two scalars a and b.
	 * 
	 * @param a a scalar
	 * 
	 * @param b a scalar
	 * 
	 * @return a - b
	 * 
	 */
	public static double minus(double a, double b) {
		return a - b;
	}
	
	/**
	 * Unary plus.
	 * 
	 * @param A a matrix
	 * 
	 * @return +A
	 * 
	 */
	public static RealMatrix uplus(RealMatrix A) {
		return A;
	}
	
	/**
	 * Unary minus.
	 * 
	 * @param A a matrix
	 * 
	 * @return -A
	 * 
	 */
	public static RealMatrix uminus(RealMatrix A) {
		if (A == null) {
			return null;
		} else {
			return A.scalarMultiply(-1);
		}
	}
	
	/**
	 * Compute the inner product between two matrices (vectors could be 
	 * viewed as column matrices).
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return inner product of X and Y <X, Y> or trace(X'Y)
	 */
	public static double innerProduct(RealMatrix X, RealMatrix Y) {
		
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();

		if (nX != nY || dX != dY) {
			System.err.println("The operands for inner product should be of same shapes!");
		}
		
		double res = 0;
		for (int j = 0; j < nX; j++) {
			res += X.getColumnVector(j).dotProduct(Y.getColumnVector(j));
		}
		
		return res;
		
	}

	/**
	 * Get the nonnegative part of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return a matrix which is the nonnegative part of a matrix A
	 * 
	 */
	public static RealMatrix subplus(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, A.getEntry(i, j) > 0d ? A.getEntry(i, j) : 0d);
			}
		}

		return res;

	}

	/**
	 * Calculate the trace of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return trace(A)
	 * 
	 */
	public static double trace(RealMatrix A) {

		double res = 0;

		if (A.getRowDimension() != A.getColumnDimension()) {
			System.err.println("The input matrix should be square matrix!");
			System.exit(1);
		}
		int n = A.getColumnDimension();
		for (int i = 0; i < n; i++) {
			res += A.getEntry(i, i);
		}
		return res;

	}

	/**
	 * Calculate element by element multiplication of two matrices.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X .* Y
	 * 
	 */
	public static RealMatrix dotMultiply(RealMatrix X, RealMatrix Y) {
		return times(X, Y);
	}

	/**
	 * Calculate element by element division of two matrices.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X ./ Y
	 * 
	 */
	public static RealMatrix dotDivide(RealMatrix X, RealMatrix Y) {
		return ebeDivide(X, Y);
	}

	/**
	 * Calculate element by element division between a scalar and a matrix.
	 * 
	 * @param v a real number to be divided by all entries of a matrix
	 * 
	 * @param X a matrix
	 * 
	 * @return v ./ X
	 * 
	 */
	public static RealMatrix dotDivide(double v, RealMatrix X) {
		return scalarDivide(v, X);
	}

	/**
	 * Compute element-wise exponentiation of a matrix.
	 * 
	 * @param X a matrix
	 * 
	 * @param v exponent
	 * 
	 * @return X .^ v
	 * 
	 */
	public static RealMatrix power(RealMatrix X, double v) {
		
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();

		RealMatrix res = new BlockRealMatrix(dX, nX);
		RealVector ones = new ArrayRealVector(dX);
		ones.set(1);
		for (int j = 0; j < nX; j++) {
			try {
				res.setColumnVector(j, X.getColumnVector(j).map(new Power(v)));
			} catch (MatrixIndexException e) {
				e.printStackTrace();
			} catch (InvalidMatrixException e) {
				e.printStackTrace();
			} catch (FunctionEvaluationException e) {
				e.printStackTrace();
			}
		}
		return res;
		
	}
	
	/**
	 * Compute element-wise exponentiation of a matrix.
	 * 
	 * @param X a matrix
	 * 
	 * @param v exponent
	 * 
	 * @return v .^ X
	 * 
	 */
	public static RealMatrix power(double v, RealMatrix X) {
		
		int nRow = X.getRowDimension();
		int nCol = X.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.pow(v, X.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Array power. Z = X .^ Y denotes element-by-element powers. X and Y
     * must have the same dimensions.
     * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X .^ Y
	 * 
	 */
	public static RealMatrix power(RealMatrix X, RealMatrix Y) {
		
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();

		if (nX != nY || dX != dY) {
			System.err.println("The operands for Hadmada product should be of same shapes!");
		}

		RealMatrix res = new BlockRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, Math.pow(X.getEntry(i, j), Y.getEntry(i, j)));
			}
		}

		return res;
		
	}	
	
	/**
	 * Left array division.
	 * 
	 * @param v a scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return v .\ X
	 * 
	 */
	public static RealMatrix ldivide(double v, RealMatrix X) {
		return X.scalarMultiply(1 / v);
	}
	
	/**
	 * Left array division.
	 * 
	 * @param X a matrix
	 * 
	 * @param v a scalar
	 * 
	 * @return X .\ v == v ./ X
	 * 
	 */
	public static RealMatrix ldivide(RealMatrix X, double v) {
		return rdivide(v, X);
	}
	
	/**
	 * Left array division.
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return A .\ B
	 * 
	 */
	public static RealMatrix ldivide(RealMatrix A, RealMatrix B) {
		return rdivide(B, A);
	}
	
	/**
	 * Right array division.
	 * 
	 * @param v a scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return v ./ X
	 * 
	 */
	public static RealMatrix rdivide(double v, RealMatrix X) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();

		RealMatrix res = new BlockRealMatrix(dX, nX);
		RealVector V = new ArrayRealVector(dX);
		V.set(v);
		for (int j = 0; j < nX; j++) {
			res.setColumnVector(j, V.ebeDivide(X.getColumnVector(j)));
		}
		return res;
		
	}
	
	/**
	 * Right array division.
	 * 
	 * @param X a matrix
	 * 
	 * @param v a scalar
	 * 
	 * @return X ./ v
	 * 
	 */
	public static RealMatrix rdivide(RealMatrix X, double v) {
		
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();

		RealMatrix res = new BlockRealMatrix(dX, nX);
		RealVector V = new ArrayRealVector(dX);
		V.set(v);
		for (int j = 0; j < nX; j++) {
			res.setColumnVector(j, X.getColumnVector(j).ebeDivide(V));
		}
		return res;
		
	}
	
	/**
	 * Right array division.
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return A ./ B
	 * 
	 */
	public static RealMatrix rdivide(RealMatrix A, RealMatrix B) {

		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		
		if (dA == 1 && nA == 1) {
			return rdivide(A.getEntry(0, 0), B);
		} else if (dB == 1 && nB == 1) {
			return rdivide(A, B.getEntry(0, 0));
		}
		
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new BlockRealMatrix(dA, nA);
		for (int j = 0; j < nA; j++) {
			res.setColumnVector(j, A.getColumnVector(j).ebeDivide(B.getColumnVector(j)));
		}
		return res;
		
	}
	
	/**
	 * Calculate element by element division between a scalar and a matrix.
	 * 
	 * @param v a real number to be divided by all entries of a matrix
	 * 
	 * @param X a matrix
	 * 
	 * @return v ./ X
	 * 
	 */
	public static RealMatrix scalarDivide(double v, RealMatrix X) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();

		RealMatrix res = new BlockRealMatrix(dX, nX);
		RealVector V = new ArrayRealVector(dX);
		V.set(v);
		for (int j = 0; j < nX; j++) {
			res.setColumnVector(j, V.ebeDivide(X.getColumnVector(j)));
		}
		return res;

	}

	/**
	 * Get the element-wise division of two matrix.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X ./ Y
	 * 
	 */
	public static RealMatrix ebeDivide(RealMatrix X, RealMatrix Y) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new BlockRealMatrix(dX, nX);
		for (int j = 0; j < nX; j++) {
			res.setColumnVector(j, X.getColumnVector(j).ebeDivide(Y.getColumnVector(j)));
		}
		return res;

	}

	/**
	 * Get the element-wise multiplication of two matrix.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X .* Y
	 * 
	 */
	public static RealMatrix ebeMultiply(RealMatrix X, RealMatrix Y) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new BlockRealMatrix(dX, nX);
		for (int j = 0; j < nX; j++) {
			res.setColumnVector(j, X.getColumnVector(j).ebeMultiply(Y.getColumnVector(j)));
		}
		return res;

	}

	/**
	 * Calculate the Frobenius norm of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @param type can only be "fro"
	 * 
	 * @return ||A||_F
	 * 
	 */
	public static double norm(RealMatrix A, String type) {

		double res = 0;
		if (type.compareToIgnoreCase("fro") == 0) {
			res = A.getFrobeniusNorm();
		} else if (type.equals("inf")){
			res = norm(A, inf);
		} else {
			System.err.println(String.format("Norm %s unimplemented!" , type));
		}
		return res;
	}
	
	/**
	 * Compute the norm of a matrix or a vector.
	 * 
	 * @param A a matrix
	 * 
	 * @param type 1, 2
	 * 
	 * @return ||A||_{type}
	 * 
	 */
	public static double norm(RealMatrix A, int type) {
		
		/*double res = 0;
		
		if (type == 2) {
			RealMatrix M = A.transpose().multiply(A);
			res = Math.sqrt(eigs(M, 1, "lm")[1].getEntry(0, 0));
		} else if (type == 1) {
			res = max(sum(abs(A), 1)).getEntry(0, 0);
		} else {
			System.err.printf("Sorry, %d-norm of a matrix is not supported currently.", type);
		}
		
		return res;*/
		
		return norm(A, (double)type);
		
	}
	
	/**
	 * Compute the induced vector norm of a matrix or a vector.
	 * 
	 * @param A a matrix or a vector
	 * 
	 * @param type 1, 2, or, inf for a matrix or a positive real 
	 *             number for a vector
	 * 
	 * @return ||A||_{type}
	 * 
	 */
	public static double norm(RealMatrix A, double type) {
		
		double res = 0;
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		
		if (nRow == 1 || nCol == 1) {
			
			if (Double.isInfinite(type)) {
				return max(A).getEntry(0, 0);
			} else if (type > 0) {
				Math.pow(sumAll(power(abs(A), type)), 1.0 / type);
			} else {
				System.err.printf("Error norm type!\n", type);
				System.exit(1);
			}
			
		}
		
		if (type == 2) {
			RealMatrix M = A.transpose().multiply(A);
			res = Math.sqrt(eigs(M, 1, "lm")[1].getEntry(0, 0));
		} else if (Double.isInfinite(type)) {
			res = max(sum(abs(A), 2)).getEntry(0, 0); 
		} else if (type == 1) {
			res = max(sum(abs(A), 1)).getEntry(0, 0);
		} else {
			System.err.printf("Sorry, %f-norm of a matrix is not supported currently.", type);
		}
		
		return res;
		
	}
	
	/**
	 * Calculate the induced 2-norm of a matrix A or 2-norm
	 * of a vector.
	 * 
	 * @param A a matrix or a vector
	 * 
	 * @return ||A||_2
	 * 
	 */
	public static double norm(RealMatrix A) {
		return norm(A, 2);
	}

	/**
	 * Calculate the sigmoid of a matrix A by rows. Specifically, supposing 
	 * that the input activation matrix is [a11, a12; a21, a22], the output 
	 * value is 
	 * <p>
	 * [exp(a11) / exp(a11) + exp(a12), exp(a12) / exp(a11) + exp(a12); 
	 * </br>
	 * exp(a21) / exp(a21) + exp(a22), exp(a22) / exp(a21) + exp(a22)].
	 * 
	 * @param A a real matrix
	 * 
	 * @return sigmoid(A)
	 * 
	 */
	public static RealMatrix sigmoid(RealMatrix A) {

		int[] size = size(A);
		int nr = size[0];
		int nc = size[1];

		RealMatrix C = kron(A, ones(nc, 1)).subtract(repmat(reshape(A.transpose(), new int[]{nr * nc, 1}), new int[]{1, nc}));
		RealMatrix S = sum(exp(C), 2);
		RealMatrix One = ones(size(A));
		RealVector V_vec = vec(One).getColumnVector(0).ebeDivide(S.getColumnVector(0));
		return reshape(V_vec, new int[]{nc, nr}).transpose();

	}
	
	/**
	 * Generate an nRow-by-nCol matrix containing pseudo-random values drawn 
	 * from the standard uniform distribution on the open interval (0,1).
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return rand(nRow, nCol)
	 * 
	 */
	public static RealMatrix rand(int nRow, int nCol) {
		Random generator = new Random();
		RealMatrix res = new BlockRealMatrix(nRow, nCol);
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, generator.nextDouble());
			}
		}
		return res;
	}
	
	/**
	 * Generate an n-by-n matrix containing pseudo-random values drawn 
	 * from the standard uniform distribution on the open interval (0,1).
	 * 
	 * @param n number of rows or columns
	 * 
	 * @return rand(n, n)
	 * 
	 */
	public static RealMatrix rand(int n) {
		return rand(n, n);
	}
	
	/**
	 * Generate an nRow-by-nCol matrix containing pseudo-random values drawn 
	 * from the standard normal distribution.
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return randn(nRow, nCol)
	 * 
	 */
	public static RealMatrix randn(int nRow, int nCol) {
		Random generator = new Random();
		RealMatrix res = new BlockRealMatrix(nRow, nCol);
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, generator.nextGaussian());
			}
		}
		return res;
	}
	
	/**
	 * Average or mean value of array. If A is a vector, 
	 * mean(A) returns the mean value of A. If A is a matrix, 
	 * mean(A) treats the columns of A as vectors, returning 
	 * a row vector of mean values.
	 * 
	 * @param X a real matrix
	 * 
	 * @return mean(A)
	 * 
	 */
	public static RealMatrix mean(RealMatrix X) {
		return mean(X, 1);
	}
	
	/**
	 * M = mean(A, dim) returns the mean values for elements 
	 * along the dimension of A specified by scalar dim. 
	 * For matrices, mean(A, 2) is a column matrix containing 
	 * the mean value of each row.
	 *
	 * @param X a real matrix
	 * 
	 * @param dim dimension order
	 * 
	 * @return mean(A, dim)
	 * 
	 */
	public static RealMatrix mean(RealMatrix X, int dim) {
		
		int nRow = size(X, 1);
		int nCol = size(X, 2);
		
		if (nRow == 1) {
			return new Array2DRowRealMatrix(new double[] {sumAll(X) / nRow});
		} else if (nCol == 1) {
			return new Array2DRowRealMatrix(new double[] {sumAll(X) / nCol});
		} else if (dim == 1) {
			return rdivide(sum(X, 1), nRow);
		} else if (dim == 2) {
			return rdivide(sum(X, 2), nCol);
		} else {
			System.err.println("dim should be either 1 or 2.");
			System.exit(1);
			return null;
		}
		
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and covariance SIGMA.
	 * 
	 * @param MU 1 x d mean vector
	 * 
	 * @param SIGMA covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static RealMatrix mvnrnd(RealMatrix MU, RealMatrix SIGMA, int cases) {
		return MultivariateGaussianDistribution.mvnrnd(MU, SIGMA, cases);
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and covariance SIGMA.
	 * 
	 * @param MU a 1D {@code double} array holding the mean vector
	 * 
	 * @param SIGMA a 2D {@code double} array holding the covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static RealMatrix mvnrnd(double[] MU, double[][] SIGMA, int cases) {
		return mvnrnd(new Array2DRowRealMatrix(MU).transpose(), new BlockRealMatrix(SIGMA), cases);
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and a diagonal covariance SIGMA.
	 * 
	 * @param MU a 1D {@code double} array holding the mean vector
	 * 
	 * @param SIGMA a 1D {@code double} array holding the diagonal elements
	 *        of the covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static RealMatrix mvnrnd(double[] MU, double[] SIGMA, int cases) {
		return mvnrnd(new Array2DRowRealMatrix(MU).transpose(), diag(SIGMA), cases);
	}
	
	/**
	 * Generate an n-by-n matrix containing pseudo-random values drawn 
	 * from the standard normal distribution.
	 * 
	 * @param n number of rows or columns
	 * 
	 * @return randn(n, n)
	 * 
	 */
	public static RealMatrix randn(int n) {
		return randn(n, n);
	}

	/**
	 * Replicate and tile an array. 
	 * 
	 * @param A a matrix
	 * 
	 * @param M number of rows to replicate
	 * 
	 * @param N number of columns to replicate
	 * 
	 * @return repmat(A, M, N)
	 * 
	 */
	public static RealMatrix repmat(RealMatrix A, int M, int N) {

		return kron(ones(M, N), A);

	}
	
	/**
	 * Replicate and tile an array. 
	 * 
	 * @param A a matrix
	 * 
	 * @param size a int[2] vector [M N]
	 * 
	 * @return repmat(A, size)
	 * 
	 */
	public static RealMatrix repmat(RealMatrix A, int[] size) {

		return kron(ones(size), A);

	}

	/**
	 * Calculate the element-wise logarithm of a matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return log(A)
	 * 
	 */
	public static RealMatrix log(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = A.copy();

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.log(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Calculate the element-wise exponential of a matrix
	 * 
	 * @param A a matrix
	 * 
	 * @return exp(A)
	 * 
	 */
	public static RealMatrix exp(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = A.copy();

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.exp(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Generates a linearly spaced vector.
	 * 
	 * @param begin starting point
	 * 
	 * @param end ending point
	 * 
	 * @param n number of points generated
	 * 
	 * @return a row matrix containing n points between begin and end
	 * 
	 */
	public static RealMatrix linspace(double begin, double end, int n) {

		if (n < 1) {
			System.err.println("The number of points should not be less than one!");
			System.exit(1);
		}

		double d = (end - begin) / (n - 1);
		RealMatrix res = new BlockRealMatrix(1, n);

		for (int i = 0; i < n; i++) {
			res.setEntry(0, i, begin + i * d);
		}

		return res;

	}

	/**
	 * Generates a linearly spaced vector with distance of
	 * D between two consecutive numbers. colon(J, D, K) is
	 * the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
	 * 
	 * @param begin starting point
	 * 
	 * @param d distance between two consecutive numbers
	 * 
	 * @param end ending point
	 * 
	 * @return a linearly spaced row matrix equally spaced by d
	 * 
	 */
	public static RealMatrix colon(double begin, double d, double end) {

		int m = fix((end - begin) / d);
		if (m < 0) {
			System.err.println("Difference error!");
			System.exit(1);
		}
		RealMatrix res = new BlockRealMatrix(1, m + 1);

		for (int i = 0; i <= m; i++) {
			res.setEntry(0, i, begin + i * d);
		}

		return res;

	}

	/**
	 * Same as colon(begin, 1, end).
	 * 
	 * @param begin starting point
	 * 
	 * @param end ending point
	 * 
	 * @return a linearly spaced row matrix equally spaced by 1
	 * 
	 */
	public static RealMatrix colon(double begin, double end) {
		return colon(begin, 1, end);
	}
	
	/**
	 * Generates a linearly spaced integer array with distance of
	 * D between two consecutive numbers. colon(J, D, K) is
	 * the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
	 * 
	 * @param begin starting point (inclusive)
	 * 
	 * @param d distance between two consecutive numbers
	 * 
	 * @param end ending point (inclusive if possible)
	 * 
	 * @return indices array for the syntax begin:d:end
	 * 
	 */
	public static int[] colon(int begin, int d, int end) {

		int m = fix((end - begin) / d);
		if (m < 0) {
			System.err.println("Difference error!");
			System.exit(1);
		}
		
		int[] res = new int[m + 1];
		
		for (int i = 0; i <= m; i++) {
			res[i] = begin + i * d;
		}

		return res;

	}

	/**
	 * Same as colon(begin, 1, end).
	 * 
	 * @param begin starting point (inclusive)
	 * 
	 * @param end ending point (inclusive)
	 * 
	 * @return indices array for the syntax begin:end
	 * 
	 */
	public static int[] colon(int begin, int end) {
		return colon(begin, 1, end);
	}

	/**
	 * Returns an array that contains 1's where
	 * the elements of X are NaN's and 0's where they are not.
	 * 
	 * @param A a matrix
	 * 
	 * @return a 0-1 matrix: isnan(A)
	 * 
	 */
	public static RealMatrix isnan(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Double.isNaN(A.getEntry(i, j)) ? 1 : 0);
			}
		}

		return res;

	}

	/**
	 * returns an array that contains 1's where the
	 * elements of X are +Inf or -Inf and 0's where they are not.
	 * 
	 * @param A a matrix
	 * 
	 * @return a 0-1 matrix: isinf(A)
	 * 
	 */
	public static RealMatrix isinf(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Double.isInfinite(A.getEntry(i, j)) ? 1 : 0);
			}
		}

		return res;

	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * A and B must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X < Y or lt(X, Y)
	 * 
	 */
	public static RealMatrix lt(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) < Y.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X < x or lt(X, x)
	 * 
	 */
	public static RealMatrix lt(RealMatrix X, double x) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) < x ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x < X or lt(x, X)
	 * 
	 */
	public static RealMatrix lt(double x, RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, x < X.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * A and B must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X > Y or gt(X, Y)
	 * 
	 */
	public static RealMatrix gt(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) > Y.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X > x or gt(X, x)
	 * 
	 */
	public static RealMatrix gt(RealMatrix X, double x) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) > x ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x > X or gt(x, X)
	 * 
	 */
	public static RealMatrix gt(double x, RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, x > X.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X <= Y or le(X, Y)
	 * 
	 */
	public static RealMatrix le(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) <= Y.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X <= x or le(X, x)
	 * 
	 */
	public static RealMatrix le(RealMatrix X, double x) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) <= x ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x <= X or le(x, X)
	 * 
	 */
	public static RealMatrix le(double x, RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, x <= X.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X >= Y or ge(X, Y)
	 * 
	 */
	public static RealMatrix ge(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) >= Y.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X >= x or ge(X, x)
	 * 
	 */
	public static RealMatrix ge(RealMatrix X, double x) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) >= x ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x >= X or ge(x, X)
	 * 
	 */
	public static RealMatrix ge(double x, RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, x >= X.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X == Y or eq(X, Y)
	 * 
	 */
	public static RealMatrix eq(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) == Y.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X == x or eq(X, x)
	 * 
	 */
	public static RealMatrix eq(RealMatrix X, double x) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) == x ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x == X or eq(x, X)
	 * 
	 */
	public static RealMatrix eq(double x, RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, x == X.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X ~= Y or ne(X, Y)
	 * 
	 */
	public static RealMatrix ne(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) != Y.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X ~= x or ne(X, x)
	 * 
	 */
	public static RealMatrix ne(RealMatrix X, double x) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, X.getEntry(i, j) != x ? 1 : 0);
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x ~= X or ne(x, X)
	 * 
	 */
	public static RealMatrix ne(double x, RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				res.setEntry(i, j, x != X.getEntry(i, j) ? 1 : 0);
			}
		}

		return res;
	}

	/**
	 * Performs a logical NOT of input array X, and returns an array
     * containing elements set to either 1 (TRUE) or 0 (FALSE). An 
     * element of the output array is set to 1 if X contains a zero
     * value element at that same array location. Otherwise, that
     * element is set to 0.
	 * 
	 * @param X a matrix
	 * 
	 * @return ~X or not(X)
	 * 
	 */
	public static RealMatrix not(RealMatrix X) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		double x = 0;
		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				x = X.getEntry(i, j);
				if (x == 1 || x == 0)
					res.setEntry(i, j, 1 - x);
				else
					System.err.println("Elements should be either 1 or 0!");
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X & Y or and(X, Y)
	 * 
	 */
	public static RealMatrix and(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		double x, y;
		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				x = X.getEntry(i, j);
				y = Y.getEntry(i, j);
				if (x == 0) {
					if (y == 1 || y == 0)
						res.setEntry(i, j, 0);
					else
						System.err.println("Elements should be either 1 or 0!");
				} else if (x == 1) {
					if (y == 1 || y == 0)
						res.setEntry(i, j, y);
					else
						System.err.println("Elements should be either 1 or 0!");
				} else
					System.err.println("Elements should be either 1 or 0!");
			}
		}

		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X | Y or or(X, Y)
	 * 
	 */
	public static RealMatrix or(RealMatrix X, RealMatrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (nX != nY || dX != dY) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		RealMatrix res = new OpenMapRealMatrix(dX, nX);

		double x, y;
		for (int i = 0; i < dX; i++) {
			for (int j = 0; j < nX; j++) {
				x = X.getEntry(i, j);
				y = Y.getEntry(i, j);
				if (x == 1) {
					if (y == 1 || y == 0)
						res.setEntry(i, j, 1);
					else
						System.err.println("Elements should be either 1 or 0!");
				} else if (x == 0) {
					if (y == 1 || y == 0)
						res.setEntry(i, j, y);
					else
						System.err.println("Elements should be either 1 or 0!");
				} else
					System.err.println("Elements should be either 1 or 0!");
			}
		}

		return res;
	}
	
	/**
	 * Logical indexing A by B for the syntax A(B). A logical matrix 
	 * provides a different type of array indexing in MATLAB. While
	 * most indices are numeric, indicating a certain row or column
	 * number, logical indices are positional. That is, it is the
	 * position of each 1 in the logical matrix that determines which
	 * array element is being referred to.
	 * 
	 * @param A a real matrix
	 * 
	 * @param B a logical matrix with elements being either 1 or 0
	 * 
	 * @return A(B)
	 * 
	 */
	public static RealMatrix logicalIndexing(RealMatrix A, RealMatrix B) {
		
		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		
		ArrayList<Double> vals = new ArrayList<Double>();
		
		double b;
		for (int j = 0; j < nA; j++) {
			for (int i = 0; i < dA; i++) {
				b = B.getEntry(i, j);
				if (b == 1)
					vals.add(A.getEntry(i, j));
				else if (b != 0)
					System.err.println("Elements of the logical matrix should be either 1 or 0!");
			}
		}
		
		Double[] Data = new Double[vals.size()];
		vals.toArray(Data);
		
		double[] data = new double[vals.size()];
		for (int i = 0; i < vals.size(); i++) {
			data[i] = Data[i];
		}		
		
		if (data.length != 0)
			return new Array2DRowRealMatrix(data);
		else
			return null;
		
	}
	
	/**
	 * Linear indexing V by an index array.
	 * 
	 * @param V an {@code int} array
	 * 
	 * @param indices an {@code int} array of selected indices
	 * 
	 * @return V(indices)
	 * 
	 */
	public static int[] linearIndexing(int[] V, int[] indices) {
		
		if (indices == null || indices.length == 0) {
			return null;
		}
		
		int[] res = new int[indices.length];
		for (int i = 0; i < indices.length; i++) {
			res[i] = V[indices[i]];
		}
		
		return res;
		
	}
	
	/**
	 * Linear indexing V by an index array.
	 * 
	 * @param V an {@code double} array
	 * 
	 * @param indices an {@code int} array of selected indices
	 * 
	 * @return V(indices)
	 * 
	 */
	public static double[] linearIndexing(double[] V, int[] indices) {
		
		if (indices == null || indices.length == 0) {
			return null;
		}
		
		double[] res = new double[indices.length];
		for (int i = 0; i < indices.length; i++) {
			res[i] = V[indices[i]];
		}
		
		return res;
		
	}
	
	/**
	 * Linear indexing A by an index array.
	 * 
	 * @param A a real matrix
	 * 
	 * @param indices an {@code int} array of selected indices
	 * 
	 * @return V(indices)
	 * 
	 */
	public static RealMatrix linearIndexing(RealMatrix A, int[] indices) {
		
		if (indices == null || indices.length == 0) {
			return null;
		}
		
		RealMatrix res = null;
		if (A instanceof Array2DRowRealMatrix || A instanceof BlockRealMatrix) {
			res = new BlockRealMatrix(indices.length, 1);
		} else {
			res = new OpenMapRealMatrix(indices.length, 1);
		}
		
		int nRow = A.getRowDimension();
		// int nCol = A.getColumnDimension();
		int r = -1;
		int c = -1;
		int index = -1;
		for (int i = 0; i < indices.length; i++) {
			index = indices[i];
			r = index % nRow;
			c = index / nRow;
			res.setEntry(i, 0, A.getEntry(r, c));
		}
		
		return res;
		
	}
	
	
	/**
	 * Matrix assignment by linear indexing for the syntax A(B) = V.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param idx a linear index vector
	 *          
	 * @param V a column matrix to assign A(idx)
	 * 
	 */
	public static void linearIndexingAssignment(RealMatrix A, int[] idx, RealMatrix V) {
		
		if (V == null)
			return;
		
		int nV = V.getColumnDimension();
		int dV = V.getRowDimension();
		
		if (nV != 1)
			System.err.println("Assignment matrix should be a column matrix!");
		
		if (idx.length != dV)
			System.err.println("Assignment with different number of elements!");
		
		int nRow = A.getRowDimension();
		// int nCol = A.getColumnDimension();
		int r = -1;
		int c = -1;
		int index = -1;
		for (int i = 0; i < idx.length; i++) {
			index = idx[i];
			r = index % nRow;
			c = index / nRow;
			A.setEntry(r, c, V.getEntry(i, 0));
		}
		
	}
	
	/**
	 * Matrix assignment by linear indexing for the syntax A(B) = v.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param idx a linear index vector
	 *          
	 * @param v a real scalar to assign A(idx)
	 * 
	 */
	public static void linearIndexingAssignment(RealMatrix A, int[] idx, double v) {
		
		int nRow = A.getRowDimension();
		// int nCol = A.getColumnDimension();
		int r = -1;
		int c = -1;
		int index = -1;
		for (int i = 0; i < idx.length; i++) {
			index = idx[i];
			r = index % nRow;
			c = index / nRow;
			A.setEntry(r, c, v);
		}
		
	}
	
	/**
	 * Matrix assignment by logical indexing for the syntax A(B) = V.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param B a logical matrix where position of each 1 determines
	 *          which array element is being assigned
	 *          
	 * @param V a column matrix to assign A(B)
	 * 
	 */
	public static void logicalIndexingAssignment(RealMatrix A, RealMatrix B, RealMatrix V) {
		
		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices for logical indexing should have same size!");
			System.exit(1);
		}
		
		/*if (V.getData().length == 0)
			return;*/
		
		if (V == null)
			return;
		
		int nV = V.getColumnDimension();
		int dV = V.getRowDimension();
		
		if (nV != 1)
			System.err.println("Assignment matrix should be a column matrix!");
		
		double b;
		int cnt = 0;
		for (int j = 0; j < nA; j++) {
			for (int i = 0; i < dA; i++) {
				b = B.getEntry(i, j);
				if (b == 1) {
					A.setEntry(i, j, V.getEntry(cnt++, 0));

				}
				else if (b != 0)
					System.err.println("Elements of the logical matrix should be either 1 or 0!");
			}
		}
		if (cnt != dV)
			System.err.println("Assignment with different number of elements!");
		
	}
	
	/**
	 * Matrix assignment by logical indexing for the syntax A(B) = v.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param B a logical matrix where position of each 1 determines
	 *          which array element is being assigned
	 *          
	 * @param v a real scalar to assign A(B)
	 * 
	 */
	public static void logicalIndexingAssignment(RealMatrix A, RealMatrix B, double v) {
		
		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices for logical indexing should have same size!");
			System.exit(1);
		}
		
		double b;
		for (int j = 0; j < nA; j++) {
			for (int i = 0; i < dA; i++) {
				b = B.getEntry(i, j);
				if (b == 1) {
					A.setEntry(i, j, v);

				}
				else if (b != 0)
					System.err.println("Elements of the logical matrix should be either 1 or 0!");
			}
		}
		
	}
	
	/**
	 * Calculate element-wise absolute value of the elements of matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return abs(A)
	 * 
	 */
	public static RealMatrix abs(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.abs(A.getEntry(i, j)));
			}
		}

		return res;

	}
	
	/**
	 * Signum function.
	 * <p>
     * For each element of X, SIGN(X) returns 1 if the element
     * is greater than zero, 0 if it equals zero and -1 if it is
     * less than zero.
     * </p>
     * 
	 * @param A a real matrix
	 * 
	 * @return sign(A)
	 * 
	 */
	public static RealMatrix sign(RealMatrix A) {
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);
		double value = 0;
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				value = A.getEntry(i, j);
				if (value > 0) {
					res.setEntry(i, j, 1);
				} else if (value < 0) {
					res.setEntry(i, j, -1);
				} else {
					res.setEntry(i, j, 0);
				}
			}
		}

		return res;
		
	}

	/**
	 * Round towards zero.
	 * 
	 * @param x a real number
	 * 
	 * @return fix(x)
	 * 
	 */
	public static int fix(double x) {

		if (x > 0) {
			return (int)Math.floor(x);
		} else {
			return (int)Math.ceil(x);
		}

	}

	/**
	 * Round elements of a matrix A towards zero to.
	 * 
	 * @param A a matrix
	 * 
	 * @return fix(A)
	 * 
	 */
	public static RealMatrix fix(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, fix(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Round towards minus infinity of elements of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return floor(A)
	 * 
	 */
	public static RealMatrix floor(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.floor(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Round elements of a matrix A towards plus infinity.
	 * 
	 * @param A a matrix
	 * 
	 * @return ceil(A)
	 * 
	 */
	public static RealMatrix ceil(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.ceil(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Round elements of a matrix A towards nearest integer.
	 * 
	 * @param A a matrix
	 * 
	 * @return round(A)
	 * 
	 */
	public static RealMatrix round(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.round(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Calculate square root for all elements of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return sqrt(A)
	 * 
	 */
	public static RealMatrix sqrt(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.sqrt(A.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Generate a diagonal matrix with its elements of a vector V 
	 * on its main diagonal.
	 *   
	 * @param V a vector
	 * 
	 * @return diag(V)
	 * 
	 */
	public static RealMatrix diag(RealVector V) {

		int d = V.getDimension();
		RealMatrix res = new OpenMapRealMatrix(d, d);

		for (int i = 0; i < d; i++) {
			res.setEntry(i, i, V.getEntry(i));
		}

		return res;

	}
	
	/**
	 * Generate a diagonal matrix with its elements of a 1D {@code double} 
	 * array on its main diagonal.
	 *   
	 * @param V a 1D {@code double} array holding the diagonal elements
	 * 
	 * @return diag(V)
	 * 
	 */
	public static RealMatrix diag(double[] V) {

		int d = V.length;
		RealMatrix res = new OpenMapRealMatrix(d, d);

		for (int i = 0; i < d; i++) {
			res.setEntry(i, i, V[i]);
		}

		return res;

	}

	/**
	 * If A is a 1-row or 1-column matrix, then diag(A) is a
	 * sparse diagonal matrix with elements of A as its main diagonal,
	 * else diag(A) is a column matrix holding A's diagonal elements.
	 * 
	 * @param A a matrix
	 * 
	 * @return diag(A)
	 * 
	 */
	public static RealMatrix diag(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		RealMatrix res = null;

		if (nRow == 1) {
			res = new OpenMapRealMatrix(nCol, nCol);
			for (int i = 0; i < nCol; i++) {
				res.setEntry(i, i, A.getEntry(0, i));
			} 
		} else if (nCol == 1) {
			res = new OpenMapRealMatrix(nRow, nRow);
			for (int i = 0; i < nRow; i++) {
				res.setEntry(i, i, A.getEntry(i, 0));
			}
		} else if (nRow == nCol) {
			res = new BlockRealMatrix(nRow, 1);
			for (int i = 0; i < nRow; i++) {
				res.setEntry(i, 0, A.getEntry(i, i));
			}
		}

		return res;

	}

	/**
	 * Generate an all one matrix with nRow rows and nCol columns.
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return ones(nRow, nCol)
	 * 
	 */
	public static RealMatrix ones(int nRow, int nCol) {

		if (nRow == 0 || nCol == 0) {
			return null;
		}
		
		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, 1);
			}
		}

		return res;

	}

	/**
	 * Generate an all one matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return an all one matrix with its shape specified by size 
	 * 
	 */
	public static RealMatrix ones(int[] size) {
		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
		}
		return ones(size[0], size[1]);
	}

	/**
	 * Generate an n by n all one matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return ones(n)
	 * 
	 */
	public static RealMatrix ones(int n) {
		return ones(n, n);
	}

	/**
	 * Generate nRow by nCol all zero matrix.
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return zeros(nRow, nCol)
	 * 
	 */
	public static RealMatrix zeros(int nRow, int nCol) {
		
		if (nRow == 0 || nCol == 0) {
			return null;
		}

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, 0);
			}
		}

		return res;

	}

	/**
	 * Generate an all zero matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return an all zero matrix with its shape specified by size 
	 * 
	 */
	public static RealMatrix zeros(int[] size) {
		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
		}
		return zeros(size[0], size[1]);
	}

	/**
	 * Generate an n by n all zero matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return ones(n)
	 * 
	 */
	public static RealMatrix zeros(int n) {
		return zeros(n, n);
	}

	/**
	 * Get a two dimensional integer array for size of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return size(A)
	 * 
	 */
	public static int[] size(RealMatrix A) {

		int[] res = new int[2];
		res[0] = A.getRowDimension();
		res[1] = A.getColumnDimension();
		return res;

	}

	/**
	 * Get the dimensionality on dim-th dimension for a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @param dim dimension order
	 * 
	 * @return size(A, dim)
	 * 
	 */
	public static int size(RealMatrix A, int dim) {
		if (dim == 1) {
			return A.getRowDimension();
		} else if (dim == 2) {
			return A.getColumnDimension();
		} else {
			System.err.println("Dim error!");
			return 0;
		}
	}

	/**
	 * Vectorize a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return Vectorization of a matrix A
	 */
	public static RealMatrix vec(RealMatrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow * nCol, 1);

		for (int j = 0; j < nCol; j++) {
			res.setSubMatrix(A.getColumnMatrix(j).getData(), j * nRow, 0);
		}

		return res;

	}
	
	/**
	 * Concatenate matrices vertically. All matrices in the argument
	 * list must have the same number of columns.
	 * 
	 * @param As matrices to be concatenated vertically
	 * 
	 * @return [A1; A2; ...]
	 * 
	 */
	public static RealMatrix vertcat(final RealMatrix ... As) {
		int nM = As.length;
		int nRow = 0;
		int nCol = 0;
		for (int i = 0; i < nM; i++) {
			if (As[i] == null)
				continue;
			nRow += As[i].getRowDimension();
			nCol = As[i].getColumnDimension();
		}
		
		for (int i = 1; i < nM; i++) {
			if (As[i] != null && nCol != As[i].getColumnDimension())
				System.err.println("Any matrix in the argument list should either " +
						"be empty matrix or have the same number of columns to the others!");
		}
		
		if (nRow == 0 || nCol == 0) {
			return null;
		}
		
		// RealMatrix res = new OpenMapRealMatrix(nRow, nCol);
		RealMatrix res = new BlockRealMatrix(nRow, nCol);
		int idx = 0;
		for (int i = 0; i < nM; i++) {
			if (i > 0 && As[i - 1] != null)
				idx += As[i - 1].getRowDimension();
			if (As[i] == null)
				continue;
			for (int r = 0; r < As[i].getRowDimension(); r++) {
				res.setRow(idx + r, As[i].getRow(r));
			}
		}
		return res;
	}
	
	/**
	 * Concatenate matrices horizontally. All matrices in the argument
	 * list must have the same number of rows.
	 * 
	 * @param As matrices to be concatenated horizontally
	 * 
	 * @return [A1 A2 ...]
	 * 
	 */
	public static RealMatrix horzcat(final RealMatrix ... As) {
		int nM = As.length;
		int nCol = 0;
		int nRow = 0;
		for (int i = 0; i < nM; i++) {
			if (As[i] != null) {
				nCol += As[i].getColumnDimension();
				nRow = As[i].getRowDimension();
			}
		}
		
		for (int i = 1; i < nM; i++) {
			if (As[i] != null && nRow != As[i].getRowDimension())
				System.err.println("Any matrix in the argument list should either " +
						"be empty matrix or have the same number of rows to the others!");
		}
		
		// RealMatrix res = new OpenMapRealMatrix(nRow, nCol);
		
		if (nRow == 0 || nCol == 0) {
			return null;
		}
		
		RealMatrix res = new BlockRealMatrix(nRow, nCol);
		int idx = 0;
		for (int i = 0; i < nM; i++) {
			if (i > 0 && As[i - 1] != null)
				idx += As[i - 1].getColumnDimension();
			if (As[i] == null)
				continue;
			for (int c = 0; c < As[i].getColumnDimension(); c++) {
				res.setColumn(idx + c, As[i].getColumn(c));
			}
		}
		return res;
		
	}
	
	/**
	 * Concatenate matrices along specified dimension.
	 * 
	 * @param dim specified dimension, can only be either 1 or 2 currently
	 * 
	 * @param As matrices to be concatenated
	 * 
	 * @return a concatenation of all the matrices in the argument list
	 * 
	 */
	public static RealMatrix cat(int dim, final RealMatrix... As) {
		RealMatrix res = null;
		if (dim == 1)
			res = vertcat(As);
		else if (dim == 2)
			res = horzcat(As);
		else
			System.err.println("Specified dimension can only be either 1 or 2 currently!");
		
		return res;
	}

	/**
	 * Calculate the Kronecker tensor product of A and B
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return Kronecker product of A and B
	 * 
	 */
	public static RealMatrix kron(RealMatrix A, RealMatrix B) {

		int nRowLeft = A.getRowDimension();
		int nColLeft = A.getColumnDimension();
		int nRowRight = B.getRowDimension();
		int nColRight = B.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRowLeft * nRowRight, nColLeft * nColRight);

		for (int i = 0; i < nRowLeft; i++) {
			for (int j = 0; j < nColLeft; j++) {
				res.setSubMatrix(B.scalarMultiply(A.getEntry(i, j)).getData(), i * nRowRight, j * nColRight);
			}
		}

		return res;

	}
	
	/**
	 * Reshape a matrix to a new shape specified number of rows and
	 * columns.
	 * 
	 * @param A a matrix
	 * 
	 * @param M number of rows of the new shape
	 * 
	 * @param N number of columns of the new shape
	 * 
	 * @return a new M-by-N matrix whose elements are taken columnwise
	 *         from A
	 * 
	 */
	public static RealMatrix reshape(RealMatrix A, int M, int N) {
		return reshape(A, new int[]{M, N});
	}

	/**
	 * Reshape a matrix to a new shape specified by a two dimensional
	 * integer array.
	 * 
	 * @param A a matrix
	 * 
	 * @param size a two dimensional integer array describing a new shape
	 * 
	 * @return a new matrix with a shape specified by size 
	 * 
	 */
	public static RealMatrix reshape(RealMatrix A, int[] size) {

		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
			System.exit(1);
		}

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (size[0] * size[1] != nRow * nCol) {
			System.err.println("Wrong shape!");
			System.exit(1);
		}

		RealMatrix res = new BlockRealMatrix(size[0], size[1]);
		double[] colSrcData = null;
		double[] colDestData = new double[size[0]];

		int cntRow = 0;
		int cntCol = 0;
		for (int j = 0; j < nCol; j++) {
			colSrcData = A.getColumnVector(j).getData();
			for (int i = 0; i < nRow; i++) {
				colDestData[cntRow++] = colSrcData[i];
				if (cntRow == size[0]) {
					res.setColumn(cntCol++, colDestData);
					cntRow = 0;
				}
			}
		}

		return res;

	}
	
	/**
	 * Reshape a vector to a new shape specified number of rows and
	 * columns.
	 * 
	 * @param V a vector
	 * 
	 * @param M number of rows of the new shape
	 * 
	 * @param N number of columns of the new shape
	 * 
	 * @return a new M-by-N matrix whose elements are taken from V
	 * 
	 */
	public static RealMatrix reshape(RealVector V, int M, int N) {
		return reshape(V, new int[]{M, N});
	}

	/**
	 * Reshape a vector to a matrix with a shape specified by a two dimensional
	 * integer array.
	 * 
	 * @param V a vector
	 * 
	 * @param size a two dimensional integer array describing a new shape
	 * 
	 * @return a new matrix with a shape specified by size 
	 * 
	 */
	public static RealMatrix reshape(RealVector V, int[] size) {

		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
		}

		int dim = V.getDimension();

		if (size[0] * size[1] != dim) {
			System.err.println("Wrong shape!");
		}

		RealMatrix res = new BlockRealMatrix(size[0], size[1]);
		double[] colSrcData = null;
		double[] colDestData = new double[size[0]];

		int cntRow = 0;
		int cntCol = 0;

		colSrcData = V.getData();
		for (int i = 0; i < dim; i++) {
			colDestData[cntRow++] = colSrcData[i];
			if (cntRow == size[0]) {
				res.setColumn(cntCol++, colDestData);
				cntRow = 0;
			}
		}

		return res;

	}
	
	/**
	 * Sort elements of a matrix A on a direction in a specified order.
	 * A will not be modified.
	 * 
	 * @param A a matrix to be sorted
	 * 
	 * @param dim sorting direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @param order sorting order, either "ascend" or "descend"
	 * 
	 * @return a {@code SortResult} data structure which has two instance
	 * data members:<br/>
	 * B: sorted values as a {@code RealMatrix} data structure<br/>
	 * IX: indices of sorted values in the original matrix<br/>
	 * 
	 */
	public static SortResult sort(final RealMatrix A, int dim, String order) {
		RealMatrix B = A.copy();
		int[][] IX = double2DArray2Int2DArray(sort2(B, dim, order).getData());
		return new SortResult(B, IX);
	}
	
	/**
	 * Sort elements of a matrix A on a direction in a increasing order.
	 * A will not be modified.
	 * 
	 * @param A a matrix to be sorted
	 * 
	 * @param dim sorting direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @return a {@code SortResult} data structure which has two instance
	 * data members:<br/>
	 * B: sorted values as a {@code RealMatrix} data structure<br/>
	 * IX: indices of sorted values in the original matrix<br/>
	 * 
	 */
	public static SortResult sort(final RealMatrix A, int dim) {
		return sort(A, dim, "ascend");
	}
	
	/**
	 * Sort elements of a real vector V in place in a specified order.
	 * 
	 * @param V a real vector to be sorted
	 * 
	 * @param order sorting order, either "ascend" or "descend"
	 * 
	 * @return 1D array of indices for sorted values in the original vector
	 * 
	 */
	public static int[] sort(RealVector V, String order) {
		RealMatrix A = new Array2DRowRealMatrix(V.getData());
		int[] IX = double1DArray2Int1DArray(sort2(A, 1, order).getColumn(0));
		V.setSubVector(0, A.getColumnVector(0));
		return IX;
	}
	
	/**
	 * Sort elements of a matrix A on a direction in an increasing order.
	 * 
	 * @param A a matrix, A will be sorted in place
	 * 
	 * @param dim sorting direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @return index matrix, indices start from 0
	 * 
	 */
	public static RealMatrix sort2(RealMatrix A, int dim) {
		return sort2(A, dim, "ascend");
	}
	
	/**
	 * Sort elements of a matrix A column-wisely in an increasing order.
	 * 
	 * @param A a matrix, A will be sorted in place
	 * 
	 * @return index matrix, indices start from 0
	 * 
	 */
	public static RealMatrix sort2(RealMatrix A) {
		int dim = 1;
		return sort2(A, dim, "ascend");
	}

	/**
	 * Sort elements of a matrix A on a direction in a specified order.
	 * 
	 * @param A a matrix, A will be sorted in place
	 * 
	 * @param dim sorting direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @param order sorting order, either "ascend" or "descend"
	 * 
	 * @return index matrix, indices start from 0
	 * 
	 */
	public static RealMatrix sort2(RealMatrix A, int dim, String order) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix indexMatrix = new BlockRealMatrix(nRow, nCol);

		// Sort columns
		if (dim == 1) {

			for (int j = 0; j < nCol; j++) {

				Vector<Pair<Integer, Double>> srcVec = new Vector<Pair<Integer, Double>>();

				for(int i = 0; i < nRow; i++) {
					srcVec.add(new Pair<Integer, Double>(i, A.getEntry(i, j)));
				}	

				// Sort idxVector based on valueVector's value
				if ( order.compareTo("ascend") == 0 ) {
					Collections.sort(srcVec, new Comparator<Pair<Integer, Double>>() { 
						@Override 
						public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
							return Double.compare(o1.getSecond(), o2.getSecond());
						} 
					});
				} else if ( order.compareTo("descend") == 0 ) {
					Collections.sort(srcVec, new Comparator<Pair<Integer, Double>>() { 
						@Override 
						public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
							return Double.compare(o2.getSecond(), o1.getSecond());
						} 
					});
				} else {
					System.err.println("dim should be either 1 or 2!");
				}

				for(int i = 0; i < nRow; i++) {
					A.setEntry(i, j, srcVec.get(i).getSecond());
					indexMatrix.setEntry(i, j, srcVec.get(i).getFirst());
				}

			}
			// Sort rows	
		} else if (dim == 2) {

			for (int i = 0; i < nRow; i++) {

				Vector<Pair<Integer, Double>> srcVec = new Vector<Pair<Integer, Double>>();

				for(int j = 0; j < nCol; j++) {
					srcVec.add(new Pair<Integer, Double>(j, A.getEntry(i, j)));
				}

				// Sort idxVector based on valueVector's value
				if ( order.compareTo("ascend") == 0 ) {
					Collections.sort(srcVec, new Comparator<Pair<Integer, Double>>() { 
						@Override 
						public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
							return Double.compare(o1.getSecond(), o2.getSecond());
						} 
					});
				} else if ( order.compareTo("descend") == 0 ) {
					Collections.sort(srcVec, new Comparator<Pair<Integer, Double>>() { 
						@Override 
						public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
							return Double.compare(o2.getSecond(), o1.getSecond());
						} 
					});
				} else {
					System.err.println("dim should be either 1 or 2!");
				}

				for(int j = 0; j < nCol; j++) {
					A.setEntry(i, j, srcVec.get(j).getSecond());
					indexMatrix.setEntry(i, j, srcVec.get(j).getFirst());
				}
			}
		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return indexMatrix;

	}

	@Deprecated
	public static RealMatrix sort3(RealMatrix A, int dim, String order) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix indexMatrix = new BlockRealMatrix(nRow, nCol);

		// Sort columns
		if (dim == 1) {

			for (int j = 0; j < nCol; j++) {
				final Double[] valueVector = new Double[nRow];
				for(int i = 0; i < nRow; i++) {
					valueVector[i] = A.getEntry(i, j);
				}	

				Integer[] idxVector = new Integer[nRow];

				for(int i = 0; i < nRow; i++) {
					idxVector[i] = i;
				}

				// Sort idxVector based on valueVector's value
				if ( order.compareTo("ascend") == 0 ) {
					Arrays.sort(idxVector, new Comparator<Integer>() { 
						@Override 
						public int compare(final Integer o1, final Integer o2) {
							return Double.compare(valueVector[o1], valueVector[o2]);
						} 
					});
				} else if ( order.compareTo("descend") == 0 ) {
					Arrays.sort(idxVector, new Comparator<Integer>() { 
						@Override 
						public int compare(final Integer o1, final Integer o2) {
							return Double.compare(valueVector[o2], valueVector[o1]);
						} 
					});
				} else {
					System.err.println("dim should be either 1 or 2!");
				}

				for(int i = 0; i < nRow; i++) {
					indexMatrix.setEntry(i, j, idxVector[i]);
				}

				// Sort valueVector itself
				if ( order.compareTo("ascend") == 0 ) {
					Arrays.sort(valueVector);
				} else if ( order.compareTo("descend") == 0 ) {
					Arrays.sort(valueVector, Collections.reverseOrder());		
				} else {
					System.err.println("dim should be either 1 or 2!");
				}
				for(int i = 0; i < nRow; i++) {
					A.setEntry(i, j, valueVector[i]);
				}
			}
			// Sort rows	
		} else if (dim == 2) {
			for (int i = 0; i < nRow; i++) {
				final Double[] valueVector = new Double[nCol];
				for(int j = 0; j < nCol; j++) {
					valueVector[j] = A.getEntry(i, j);
				}	

				Integer[] idxVector = new Integer[nCol];

				for(int j = 0; j < nCol; j++) {
					idxVector[j] = j;
				}

				// Sort idxVector based on valueVector's value
				if ( order.compareTo("ascend") == 0 ) {
					Arrays.sort(idxVector, new Comparator<Integer>() { 
						@Override 
						public int compare(final Integer o1, final Integer o2) {
							return Double.compare(valueVector[o1], valueVector[o2]);
						} 
					});
				} else if ( order.compareTo("descend") == 0 ) {
					Arrays.sort(idxVector, new Comparator<Integer>() { 
						@Override 
						public int compare(final Integer o1, final Integer o2) {
							return Double.compare(valueVector[o2], valueVector[o1]);
						} 
					});
				} else {
					System.err.println("dim should be either 1 or 2!");
				}

				for(int j = 0; j < nCol; j++) {
					indexMatrix.setEntry(i, j, idxVector[j]);
				}

				// Sort valueVector itself
				if ( order.compareTo("ascend") == 0 ) {
					Arrays.sort(valueVector);
				} else if ( order.compareTo("descend") == 0 ) {
					Arrays.sort(valueVector, Collections.reverseOrder());		
				} else {
					System.err.println("dim should be either 1 or 2!");
				}
				for(int j = 0; j < nCol; j++) {
					A.setEntry(i, j, valueVector[j]);
				}
			}
		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return indexMatrix;

	}

	/**
	 * Calculate the sum of a row matrix A or its column-wise sum.
	 * 
	 * @param A a matrix
	 * 
	 * @return sum(A)
	 * 
	 */
	public static RealMatrix sum(RealMatrix A) {

		if (A.getRowDimension() == 1) {
			return sum(A, 2);
		} else {
			return sum(A, 1);
		}

	}
	
	/**
	 * Compute the sum of all elements of a matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return sum(sum(A))
	 * 
	 */
	public static double sumAll(RealMatrix A) {
		if (A == null) {
			return 0;
		}
		if (A.getRowDimension() == 1) {
			return sum(A, 2).getEntry(0, 0);
		} else {
			return sumAll(sum(A, 1));
		}
	}

	/**
	 * Calculate the sum of elements of A on a dimension.
	 * 
	 * @param A a real matrix
	 * 
	 * @param dim 1: column-wise; 2: row-wise
	 * 
	 * @return sum(A, dim)
	 * 
	 */
	public static RealMatrix sum(RealMatrix A, int dim) {

		RealMatrix res = null;
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {

			res = new BlockRealMatrix(1, nCol);

			for (int j = 0; j < nCol; j++) {
				double sumVal = A.getEntry(0, j);
				for (int i = 1; i < nRow; i++ ) {
					sumVal += A.getEntry(i, j);
				}
				res.setEntry(0, j, sumVal);
			}

		} else if (dim == 2) {

			res = new BlockRealMatrix(nRow, 1);

			for (int i = 0; i < nRow; i++) {
				double sumVal = A.getEntry(i, 0);
				for (int j = 1; j < nCol; j++ ) {
					sumVal += A.getEntry(i, j);
				}
				res.setEntry(i, 0, sumVal);
			}

		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}
	
	/**
	 * Compute the sum of all elements of a vector.
	 * 
	 * @param V a vector
	 * 
	 * @return sum_i V_i
	 * 
	 */
	public static double sum(RealVector V) {
		
		double sum = 0;
		for (int i = 0; i < V.getDimension(); i++)
			sum += V.getEntry(i);
		return sum;
		
	}

	/**
	 * Calculate the maximum of a row matrix A or its column-wise maximum.
	 * 
	 * @param A a matrix
	 * 
	 * @return max(A)
	 * 
	 */
	public static RealMatrix max(RealMatrix A) {

		if (A.getRowDimension() == 1) {
			return max(A, 2).get("val");
		} else {
			return max(A, 1).get("val");
		}

	}

	/**
	 * Calculate maximum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param A a matrix
	 * 
	 * @param c a real number
	 * 
	 * @return max(A, c)
	 * 
	 */
	public static RealMatrix max(RealMatrix A, double c) {

		if (A == null)
			return null;
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, A.getEntry(i, j) > c ? A.getEntry(i, j) : c);
			}
		}

		return res;

	}
	
	/**
	 * Calculate maximum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param c a real number
	 * 
	 * @param A a matrix
	 * 
	 * @return max(c, A)
	 * 
	 */
	public static RealMatrix max(double c, RealMatrix A) {
		return max(A, c);
	}

	/**
	 * Calculate element-wise maximum between two matrices X and Y.
	 * 
	 * @param X a real matrix
	 * 
	 * @param Y a real matrix
	 * 
	 * @return max(X, Y)
	 * 
	 */
	public static RealMatrix max(RealMatrix X, RealMatrix Y) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();

		if (nX != nY || dX != dY) {
			System.err.println("The operands for Hadmada product should be of same shapes!");
		}

		int nRow = dX;
		int nCol = nY;

		RealMatrix res = X.copy();
		double value = 0;
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				value = Math.max(X.getEntry(i, j), Y.getEntry(i, j));
				if (value != 0)
					res.setEntry(i, j, value);
			}
		}

		return res;

	}

	/**
	 * Calculate the maximal value and its row or column index
	 * of a matrix A. The row and column indices start from 0.
	 * 
	 * @param A a matrix
	 * 
	 * @param dim dimension order
	 * 
	 * @return 
	 *         A TreeMap<String, RealMatrix>
	 *         <pre>
	 * "val": maximal values
	 * "idx": indices of the maximal values in A
	 *         </pre>
	 *         
	 */
	public static TreeMap<String, RealMatrix> max(RealMatrix A, int dim) {

		TreeMap<String, RealMatrix> res = new TreeMap<String, RealMatrix>();

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {

			RealMatrix valueMatrix = new BlockRealMatrix(1, nCol);
			RealMatrix indexMatrix = new BlockRealMatrix(1, nCol);

			for (int j = 0; j < nCol; j++) {
				double maxVal = A.getEntry(0, j);
				int maxIdx = 0;
				for (int i = 1; i < nRow; i++ ) {
					if (maxVal < A.getEntry(i, j)) {
						maxVal = A.getEntry(i, j);
						maxIdx = i;
					}
				}
				valueMatrix.setEntry(0, j, maxVal);
				indexMatrix.setEntry(0, j, maxIdx);
			}

			res.put("val", valueMatrix);
			res.put("idx", indexMatrix);

		} else if (dim == 2) {

			RealMatrix valueMatrix = new BlockRealMatrix(nRow, 1);
			RealMatrix indexMatrix = new BlockRealMatrix(nRow, 1);

			for (int i = 0; i < nRow; i++) {
				double maxVal = A.getEntry(i, 0);
				int maxIdx = 0;
				for (int j = 1; j < nCol; j++ ) {
					if (maxVal < A.getEntry(i, j)) {
						maxVal = A.getEntry(i, j);
						maxIdx = j;
					}
				}
				valueMatrix.setEntry(i, 0, maxVal);
				indexMatrix.setEntry(i, 0, maxIdx);
			}

			res.put("val", valueMatrix);
			res.put("idx", indexMatrix);

		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}
	
	@Deprecated
	public static ArrayList<RealMatrix> max1(RealMatrix A, int dim) {

		ArrayList<RealMatrix> res = new ArrayList<RealMatrix>(2);

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {

			RealMatrix valueMatrix = new BlockRealMatrix(1, nCol);
			RealMatrix indexMatrix = new BlockRealMatrix(1, nCol);

			for (int j = 0; j < nCol; j++) {
				double maxVal = A.getEntry(0, j);
				int maxIdx = 0;
				for (int i = 1; i < nRow; i++ ) {
					if (maxVal < A.getEntry(i, j)) {
						maxVal = A.getEntry(i, j);
						maxIdx = i;
					}
				}
				valueMatrix.setEntry(0, j, maxVal);
				indexMatrix.setEntry(0, j, maxIdx);
			}

			res.add(valueMatrix);
			res.add(indexMatrix);

		} else if (dim == 2) {

			RealMatrix valueMatrix = new BlockRealMatrix(nRow, 1);
			RealMatrix indexMatrix = new BlockRealMatrix(nRow, 1);

			for (int i = 0; i < nRow; i++) {
				double maxVal = A.getEntry(i, 0);
				int maxIdx = 0;
				for (int j = 1; j < nCol; j++ ) {
					if (maxVal < A.getEntry(i, j)) {
						maxVal = A.getEntry(i, j);
						maxIdx = j;
					}
				}
				valueMatrix.setEntry(i, 0, maxVal);
				indexMatrix.setEntry(i, 0, maxIdx);
			}
			res.add(valueMatrix);
			res.add(indexMatrix);
		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}

	@Deprecated
	public static ArrayList<RealVector> max2(RealMatrix A, int dim) {

		ArrayList<RealVector> res = new ArrayList<RealVector>();

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {
			RealVector valueVector = new ArrayRealVector(nCol, 0);
			RealVector indexVector = new ArrayRealVector(nCol, 0);
			for (int j = 0; j < nCol; j++) {
				double maxValue = A.getEntry(0, j);
				int maxIdx = 0;
				for (int i = 1; i < nRow; i++ ) {
					if (maxValue < A.getEntry(i, j)) {
						maxValue = A.getEntry(i, j);
						maxIdx = i;
					}
				}
				valueVector.setEntry(j, maxValue);
				indexVector.setEntry(j, maxIdx);
			}
			res.add(valueVector);
			res.add(indexVector);
		} else if (dim == 2) {
			RealVector valueVector = new ArrayRealVector(nRow, 0);
			RealVector indexVector = new ArrayRealVector(nRow, 0);
			for (int i = 0; i < nRow; i++) {
				double maxValue = A.getEntry(i, 0);
				int maxIdx = 0;
				for (int j = 1; j < nCol; j++ ) {
					if (maxValue < A.getEntry(i, j)) {
						maxValue = A.getEntry(i, j);
						maxIdx = j;
					}
				}
				valueVector.setEntry(i, maxValue);
				indexVector.setEntry(i, maxIdx);
			}
			res.add(valueVector);
			res.add(indexVector);
		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}

	/**
	 * Get the maximal value and its index in V.
	 * 
	 * @param V a vector
	 * 
	 * @return a two dimensional double array int[2] res, res[0] is the maximum,
	 *         and res[1] is the index
	 *         
	 */
	public static double[] max(RealVector V) {

		double[] res = new double[2];

		int d = V.getDimension();

		double maxValue = V.getEntry(0);
		int maxIdx = 0;
		for (int i = 1; i < d; i++) {
			if (maxValue < V.getEntry(i)) {
				maxValue = V.getEntry(i);
				maxIdx = i;
			}
		}
		res[0] = maxValue;
		res[1] = maxIdx;

		return res;
	}

	/**
	 * Calculate the minimum of a row matrix A or its column-wise minimum.
	 * 
	 * @param A a matrix
	 * 
	 * @return min(A)
	 * 
	 */
	public static RealMatrix min(RealMatrix A) {

		if (A.getRowDimension() == 1) {
			return min(A, 2).get("val");
		} else {
			return min(A, 1).get("val");
		}

	}

	/**
	 * Calculate minimum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param A a matrix
	 * 
	 * @param c a real number
	 * 
	 * @return min(A, c)
	 * 
	 */
	public static RealMatrix min(RealMatrix A, double c) {

		if (A == null)
			return null;
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, A.getEntry(i, j) < c ? A.getEntry(i, j) : c);
			}
		}

		return res;

	}
	
	/**
	 * Calculate minimum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param c a real number
	 * 
	 * @param A a matrix
	 * 
	 * @return min(c, A)
	 * 
	 */
	public static RealMatrix min(double c, RealMatrix A) {
		return min(A, c);
	}

	/**
	 * Calculate element-wise minimum between two matrices X and Y.
	 * 
	 * @param X a real matrix
	 * 
	 * @param Y a real matrix
	 * 
	 * @return min(X, Y)
	 * 
	 */
	public static RealMatrix min(RealMatrix X, RealMatrix Y) {

		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();

		if (nX != nY || dX != dY) {
			System.err.println("The operands for Hadmada product should be of same shapes!");
		}

		int nRow = dX;
		int nCol = nY;

		RealMatrix res = new BlockRealMatrix(nRow, nCol);

		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.min(X.getEntry(i, j), Y.getEntry(i, j)));
			}
		}

		return res;

	}

	/**
	 * Calculate the minimal value and its row and column index
	 * of a matrix A. The row and column indices start from 0.
	 * 
	 * @param A a matrix
	 * 
	 * @param dim dimension order
	 * 
	 * @return 
	 *         A TreeMap<String, RealMatrix>
	 *         <pre>
	 * "val": minimal values
	 * "idx": indices of the minimal values in A
	 *         </pre>   
	 */
	public static TreeMap<String, RealMatrix> min(RealMatrix A, int dim) {

		TreeMap<String, RealMatrix> res = new TreeMap<String, RealMatrix>();

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {

			RealMatrix valueMatrix = new BlockRealMatrix(1, nCol);
			RealMatrix indexMatrix = new BlockRealMatrix(1, nCol);

			for (int j = 0; j < nCol; j++) {
				double minVal = A.getEntry(0, j);
				int minIdx = 0;
				for (int i = 1; i < nRow; i++ ) {
					if (minVal > A.getEntry(i, j)) {
						minVal = A.getEntry(i, j);
						minIdx = i;
					}
				}
				valueMatrix.setEntry(0, j, minVal);
				indexMatrix.setEntry(0, j, minIdx);
			}

			res.put("val", valueMatrix);
			res.put("idx", indexMatrix);

		} else if (dim == 2) {

			RealMatrix valueMatrix = new BlockRealMatrix(nRow, 1);
			RealMatrix indexMatrix = new BlockRealMatrix(nRow, 1);

			for (int i = 0; i < nRow; i++) {
				double minVal = A.getEntry(i, 0);
				int minIdx = 0;
				for (int j = 1; j < nCol; j++ ) {
					if (minVal > A.getEntry(i, j)) {
						minVal = A.getEntry(i, j);
						minIdx = j;
					}
				}
				valueMatrix.setEntry(i, 0, minVal);
				indexMatrix.setEntry(i, 0, minIdx);
			}

			res.put("val", valueMatrix);
			res.put("idx", indexMatrix);

		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}

	@Deprecated
	public static ArrayList<RealMatrix> min1(RealMatrix A, int dim) {

		ArrayList<RealMatrix> res = new ArrayList<RealMatrix>(2);

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {

			RealMatrix valueMatrix = new BlockRealMatrix(1, nCol);
			RealMatrix indexMatrix = new BlockRealMatrix(1, nCol);

			for (int j = 0; j < nCol; j++) {
				double minVal = A.getEntry(0, j);
				int minIdx = 0;
				for (int i = 1; i < nRow; i++ ) {
					if (minVal > A.getEntry(i, j)) {
						minVal = A.getEntry(i, j);
						minIdx = i;
					}
				}
				valueMatrix.setEntry(0, j, minVal);
				indexMatrix.setEntry(0, j, minIdx);
			}

			res.add(valueMatrix);
			res.add(indexMatrix);

		} else if (dim == 2) {

			RealMatrix valueMatrix = new BlockRealMatrix(nRow, 1);
			RealMatrix indexMatrix = new BlockRealMatrix(nRow, 1);

			for (int i = 0; i < nRow; i++) {
				double minVal = A.getEntry(i, 0);
				int minIdx = 0;
				for (int j = 1; j < nCol; j++ ) {
					if (minVal > A.getEntry(i, j)) {
						minVal = A.getEntry(i, j);
						minIdx = j;
					}
				}
				valueMatrix.setEntry(i, 0, minVal);
				indexMatrix.setEntry(i, 0, minIdx);
			}
			res.add(valueMatrix);
			res.add(indexMatrix);
		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}

	@Deprecated
	public static ArrayList<RealVector> min2(RealMatrix A, int dim) {

		ArrayList<RealVector> res = new ArrayList<RealVector>();

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		if (dim == 1) {
			RealVector valueVector = new ArrayRealVector(nCol, 0);
			RealVector indexVector = new ArrayRealVector(nCol, 0);
			for (int j = 0; j < nCol; j++) {
				double minValue = A.getEntry(0, j);
				int minIdx = 0;
				for (int i = 1; i < nRow; i++ ) {
					if (minValue > A.getEntry(i, j)) {
						minValue = A.getEntry(i, j);
						minIdx = i;
					}
				}
				valueVector.setEntry(j, minValue);
				indexVector.setEntry(j, minIdx);
			}
			res.add(valueVector);
			res.add(indexVector);
		} else if (dim == 2) {
			RealVector valueVector = new ArrayRealVector(nRow, 0);
			RealVector indexVector = new ArrayRealVector(nRow, 0);
			for (int i = 0; i < nRow; i++) {
				double minValue = A.getEntry(i, 0);
				int minIdx = 0;
				for (int j = 1; j < nCol; j++ ) {
					if (minValue > A.getEntry(i, j)) {
						minValue = A.getEntry(i, j);
						minIdx = j;
					}
				}
				valueVector.setEntry(i, minValue);
				indexVector.setEntry(i, minIdx);
			}
			res.add(valueVector);
			res.add(indexVector);
		} else {
			System.err.println("dim should be either 1 or 2!");
		}

		return res;

	}

	/**
	 * Get the minimal value and its index in V.
	 * 
	 * @param V a vector
	 * 
	 * @return a two dimensional double array int[2] res, res[0] is the minimum,
	 *         and res[1] is the index
	 *         
	 */
	public static double[] min(RealVector V) {

		double[] res = new double[2];

		int d = V.getDimension();

		double minValue = V.getEntry(0);
		int minIdx = 0;
		for (int i = 1; i < d; i++) {
			if (minValue < V.getEntry(i)) {
				minValue = V.getEntry(i);
				minIdx = i;
			}
		}
		res[0] = minValue;
		res[1] = minIdx;

		return res;
	}

	/**
	 * Construct an n-by-n sparse identity matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return an n-by-n sparse identity matrix
	 * 
	 */
	public static RealMatrix eye(int n) {

		if (n == 0)
			return null;
		
		RealMatrix res = new OpenMapRealMatrix(n, n);
		for (int i = 0; i < n; i++) {
			res.setEntry(i, i, 1.0);
		}
		return res;
		
	}
	
	/**
	 * Construct an m-by-n sparse identity matrix.
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @return an m-by-n sparse identity matrix
	 * 
	 */
	public static RealMatrix eye(int m, int n) {

		if (m == 0 || n == 0)
			return null;
		
		RealMatrix res = new OpenMapRealMatrix(m, n);
		for (int i = 0; i < Math.min(m, n); i++) {
			res.setEntry(i, i, 1.0);
		}
		return res;
		
	}
	
	/**
	 * Generate a sparse identity matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return a sparse identity matrix with its shape specified by size 
	 * 
	 */
	public static RealMatrix eye(int... size) {
		if (size.length != 2) {
			System.err.println("Input size vector should have two elements!");
		}
		return eye(size[0], size[1]);
	}

	/**
	 * Calculate TFIDF of a doc-term-count matrix, each column
	 * is a data sample.
	 * 
	 * @param docTermCountMatrix a matrix, each column is a data sample
	 * 
	 * @return TFIDF of docTermCountMatrix
	 * 
	 */
	public static RealMatrix getTFIDF(RealMatrix docTermCountMatrix) {

		final int NTerm = docTermCountMatrix.getRowDimension();
		final int NDoc = docTermCountMatrix.getColumnDimension();

		// Get TF vector
		double[] tfVector = new double[NTerm];
		for (int i = 0; i < docTermCountMatrix.getRowDimension(); i++) {
			tfVector[i] = 0;
			for (int j = 0; j < docTermCountMatrix.getColumnDimension(); j++) {
				tfVector[i] += docTermCountMatrix.getEntry(i, j) > 0 ? 1 : 0;
			}
		}

		RealMatrix res = docTermCountMatrix.copy();
		for (int i = 0; i < docTermCountMatrix.getRowDimension(); i++) {
			for (int j = 0; j < docTermCountMatrix.getColumnDimension(); j++) {
				if (res.getEntry(i, j) > 0) {
					res.setEntry(i, j, res.getEntry(i, j) * (tfVector[i] > 0 ? Math.log(NDoc / tfVector[i]) : 0));
				}
			}
		}

		return res;

	}

	/**
	 * Normalize A by columns.
	 * @param A a matrix
	 * @return a column-wise normalized matrix
	 */
	public static RealMatrix normalizeByColumns(RealMatrix A) {

		RealMatrix res = A.copy();

		for (int j = 0; j < res.getColumnDimension(); j++) {
			RealVector feaVec = res.getColumnVector(j);
			double squaredSum = Math.sqrt(feaVec.dotProduct(feaVec));
			feaVec.mapDivideToSelf(squaredSum > 0 ? squaredSum : 1d);
			res.setColumnVector(j, feaVec);
		}

		return res;
	}

	/**
	 * Normalize A by rows.
	 * @param A a matrix
	 * @return a row-wise normalized matrix
	 */
	public static RealMatrix normalizeByRows(RealMatrix A) {

		RealMatrix res = A.copy();

		for (int i = 0; i < res.getRowDimension(); i++) {
			RealVector feaVec = res.getRowVector(i);
			double squaredSum = Math.sqrt(feaVec.dotProduct(feaVec));
			feaVec.mapDivideToSelf(squaredSum > 0 ? squaredSum : 1d);
			res.setRowVector(i, feaVec);
		}

		return res;
	}

	/**
	 * Calculate the left division of the form A \ B. A \ B is the
	 * matrix division of A into B, which is roughly the same as
	 * INV(A)*B , except it is computed in a different way. For 
	 * implementation, we actually solve the system of linear 
	 * equations A * X = B.
	 * 
	 * @param A divisor
	 * 
	 * @param B dividend
	 * 
	 * @return A \ B
	 * 
	 */
	public static RealMatrix mldivide(RealMatrix A, RealMatrix B) {
		return new QRDecompositionImpl(A).getSolver().solve(B);
	}

	/**
	 * Calculate the right division of B into A, i.e., A / B. For
	 * implementation, we actually solve the system of linear 
	 * equations X * B = A.
	 * <p>
	 * Note: X = A / B <=> X * B = A <=> B' * X' = A' <=> X' = B' \ A'
	 * <=> X = (B' \ A')'
	 * </p>
	 * 
	 * @param A dividend
	 * 
	 * @param B divisor
	 * 
	 * @return A / B
	 * 
	 */
	public static RealMatrix mrdivide(RealMatrix A, RealMatrix B) {
		return mldivide(B.transpose(), A.transpose()).transpose();
	}

	/**
	 * Compute the inverse (or pseudo-inverse) of the decomposed matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return A^{-1}
	 * 
	 */
	public static RealMatrix inverse(RealMatrix A) {
		return new LUDecompositionImpl(A).getSolver().getInverse();
	}

	/**
	 * Set submatrix of A with selected rows and selected columns by elements of B.
	 * B should have the same shape to the submatrix of A to be set. It is equivalent
	 * to the syntax A(selectedRows, selectedColumns) = B.
	 * 
	 * @param A a matrix whose submatrix is to be set
	 * 
	 * @param selectedRows {@code int[]} holding indices of selected rows
	 * 
	 * @param selectedColumns {@code int[]} holding indices of selected columns
	 * 
	 * @param B a matrix to set the submatrix of A
	 * 
	 */
	public static void setSubMatrix(RealMatrix A, int[] selectedRows, 
			int[] selectedColumns, RealMatrix B) {

		int r, c;
		for (int i = 0; i < selectedRows.length; i++) {
			for (int j = 0; j < selectedColumns.length; j++) {
				r = selectedRows[i];
				c = selectedColumns[j];
				A.setEntry(r, c, B.getEntry(i, j));
			}
		}

	}
	
	/**
	 * Get the subMatrix containing the elements of the specified rows.
	 * Rows are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param startRow initial row index (inclusive)
	 * 
	 * @param endRow final row index (inclusive)
	 * 
	 * @return the subMatrix of A containing the data of the specified rows
	 * 
	 */
	public static RealMatrix getRows(RealMatrix A, int startRow, int endRow) {
		return A.getSubMatrix(startRow, endRow, 0, A.getColumnDimension() - 1);
	}
	
	/**
	 * Get the subMatrix containing the elements of the specified rows.
	 * Rows are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param selectedRows indices of selected rows
	 * 
	 * @return the subMatrix of A containing the data of the specified rows
	 * 
	 */
	public static RealMatrix getRows(RealMatrix A, int... selectedRows) {
		return A.getSubMatrix(selectedRows, colon(0, A.getColumnDimension() - 1));
	}
	
	/**
	 * Get the subMatrix containing the elements of the specified columns.
	 * Columns are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param startColumn initial column index (inclusive)
	 * 
	 * @param endColumn final column index (inclusive)
	 * 
	 * @return the subMatrix of A containing the data of the specified rows
	 * 
	 */
	public static RealMatrix getColumns(RealMatrix A, int startColumn, int endColumn) {
		return A.getSubMatrix(0, A.getRowDimension() - 1, startColumn, endColumn);
	}
	
	/**
	 * Get the subMatrix containing the elements of the specified columns.
	 * Columns are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param selectedColumns indices of selected columns
	 * 
	 * @return the subMatrix of A containing the data of the specified columns
	 * 
	 */
	public static RealMatrix getColumns(RealMatrix A, int... selectedColumns) {
		return A.getSubMatrix(colon(0, A.getRowDimension() - 1), selectedColumns);
	}
	
	/**
	 * Set all the elements of X to be those of Y, this is particularly
	 * useful when we want to change elements of the object referred by X 
	 * rather than the reference X itself.
	 *  
	 * @param X a matrix to be set
	 * 
	 * @param Y a matrix to set X
	 * 
	 */
	public static void setMatrix(RealMatrix X, RealMatrix Y) {
		X.setSubMatrix(Y.getData(), 0, 0);
	}

	/**
	 * Convert a dense matrix into a sparse matrix.
	 *
	 * @param D a dense matrix
	 * 
	 * @return a sparse matrix
	 * 
	 */
	public static RealMatrix sparse(RealMatrix D) {

		int nRow = D.getRowDimension();
		int nCol = D.getColumnDimension();

		RealMatrix res = new OpenMapRealMatrix(nRow, nCol);

		double value = 0;
		for (int j = 0; j < nCol; j++) {
			for (int i = 0; i < nRow; i++) {
				value = D.getEntry(i, j);
				if (value != 0) {
					res.setEntry(i, j, value);
				}
			}
		}

		return res;

	}

	/**
	 * Convert a sparse matrix into a dense matrix.
	 * 
	 * @param S a sparse matrix
	 * 
	 * @return a dense matrix
	 * 
	 */
	public static RealMatrix full(RealMatrix S) {

		int d = S.getRowDimension();
		int n = S.getColumnDimension();

		RealMatrix res = new BlockRealMatrix(d, n);
		for (int j = 0; j < n; j++) {
			res.setColumnVector(j, S.getColumnVector(j));
		}
		return res;

	}

	/**
	 * convert a {@code double[]} to an {@code int[]}.
	 * 
	 * @param x a {@code double[]}
	 * 
	 * @return an {@code int[]}
	 * 
	 */
	public static int[] double1DArray2Int1DArray(double[] x) {
		int[] res = new int[x.length];
		for (int i = 0; i < x.length; i++) {
			res[i] = (int)x[i];
		}
		return res;
	}
	
	/**
	 * convert a {@code double[][]} to an {@code int[][]}.
	 * 
	 * @param X a {@code double[][]}
	 * 
	 * @return an {@code int[][]}
	 * 
	 */
	public static int[][] double2DArray2Int2DArray(double[][] X) {
		int[][] res = new int[X.length][];
		for (int i = 0; i < X.length; i++) {
			res[i] = new int[X[i].length];
			for (int j = 0; j < X[i].length; j++) {
				res[i][j] = (int)X[i][j];
			}
		}
		return res;
	}
	
	/**
	 * Display a matrix.
	 * 
	 * @param M a matrix
	 */
	public static void disp(RealMatrix M) {
		display(M);
	}
	
	/**
	 * Display a matrix with a specified precision.
	 * 
	 * @param M a matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void disp(RealMatrix M, int p) {
		display(M, p);
	}
	
	/**
	 * Display a string.
	 * 
	 * @param str a string to display
	 */
	public static void disp(String str) {
		fprintf("%s%n", str);
	}
	
	/**
	 * Display a matrix.
	 * 
	 * @param M a matrix
	 */
	public static void display(RealMatrix M) {
		printMatrix(M);
	}
	
	/**
	 * Display a matrix with a specified precision.
	 * 
	 * @param M a matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void display(RealMatrix M, int p) {
		printMatrix(M, p);
	}
	
	/**
	 * Display an integer 2D array.
	 * 
	 * @param M an integer 2D array
	 */
	public static void display(int[][] M) {
		if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		for (int i = 0; i < M.length; i++) {
			System.out.print("  ");
			for (int j = 0; j < M[0].length; j++) {
				String valueString = "";
				double v = M[i][j];
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = String.format("%.4f", v);
				else
					valueString = String.format("%d", rv);
				System.out.print(String.format("%7s", valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 * Display a real valued 2D array.
	 * 
	 * @param M a real valued 2D array
	 */
	public static void display(double[][] M) {
		if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		for (int i = 0; i < M.length; i++) {
			System.out.print("  ");
			for (int j = 0; j < M[0].length; j++) {
				String valueString = "";
				double v = M[i][j];
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = String.format("%.4f", v);
				else
					valueString = String.format("%d", rv);
				System.out.print(String.format("%7s", valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 * Display a real valued 2D array with a specified precision.
	 * 
	 * @param M a real valued 2D array
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void display(double[][] M, int p) {
		if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		for (int i = 0; i < M.length; i++) {
			System.out.print("  ");
			for (int j = 0; j < M[0].length; j++) {
				String valueString = "";
				double v = M[i][j];
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = String.format(sprintf("%%.%df", p), v);
				else
					valueString = String.format("%d", rv);
				System.out.print(String.format(sprintf("%%%ds", 7 + p - 4), valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 * Display a double scalar.
	 * 
	 * @param v a double scalar
	 * 
	 */
	public static void display(double v) {
		RealMatrix M = new BlockRealMatrix(1, 1);
		M.setEntry(0, 0, v);
		display(M);
	}
	
	/**
	 * Display a string.
	 * 
	 * @param s a string
	 * 
	 */
	public static void display(String s) {
		fprintf(s + System.getProperty("line.separator"));
	}
	
	/**
	 * Display a vector.
	 * 
	 * @param V a vector
	 * 
	 */
	public static void display(RealVector V) {
		printVector(V);
	}
	
	/**
	 * Display a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 */
	public static void display(double[] V) {
		printVector(new ArrayRealVector(V));
	}
	
	/**
	 * Display a 1D {@code double} array with a specified precision.
	 * 
	 * @param V a 1D {@code double} array
	 *
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void display(double[] V, int p) {
		printVector(new ArrayRealVector(V), p);
	}
	
	/**
	 * Display a 1D integer array.
	 * 
	 * @param V a 1D integer array
	 * 
	 */
	public static void display(int[] V) {
		
		if (V == null) {
			System.out.println("Empty vector!");
			return;
		}
		
		for (int i = 0; i < V.length; i++) {
			System.out.print("  ");
			String valueString = "";
			double v = V[i];
			int rv = (int) Math.round(v);
			if (v != rv)
				valueString = String.format("%.4f", v);
			else
				valueString = String.format("%d", rv);
			System.out.print(String.format("%7s", valueString));
			System.out.print("  ");
			// System.out.println();
		}
		System.out.println();
		
	}

	/**
	 * Print a matrix.
	 * 
	 * @param M a matrix
	 * 
	 */
	public static void printMatrix(RealMatrix M) {
		/*if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		if (M instanceof OpenMapRealMatrix) {
			printSparseMatrix((OpenMapRealMatrix)M);
			return;
		}
		for (int i = 0; i < M.getRowDimension(); i++) {
			System.out.print("  ");
			for (int j = 0; j < M.getColumnDimension(); j++) {
				String valueString = "";
				double v = M.getEntry(i, j);
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = String.format("%.4f", v);
				else
					valueString = String.format("%d", rv);
				System.out.print(String.format("%7s", valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();*/
		printMatrix(M, 4);
	}
	
	/**
	 * Print a matrix with a specified precision.
	 * 
	 * @param M a matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printMatrix(RealMatrix M, int p) {
		if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		if (M instanceof OpenMapRealMatrix) {
			printSparseMatrix((OpenMapRealMatrix)M);
			return;
		}
		for (int i = 0; i < M.getRowDimension(); i++) {
			System.out.print("  ");
			for (int j = 0; j < M.getColumnDimension(); j++) {
				String valueString = "";
				double v = M.getEntry(i, j);
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				System.out.print(sprintf(sprintf("%%%ds", 7 + p - 4), valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 * Print a sparse matrix.
	 * 
	 * @param M a sparse matrix
	 * 
	 */
	private static void printSparseMatrix(RealMatrix M) {
		
		if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		
		int nRow = M.getRowDimension();
		int nCol = M.getColumnDimension();
		
		String leftFormat = String.format("  %%%ds, ", String.valueOf(nRow).length() + 1);
		String rightFormat = String.format("%%-%ds", String.valueOf(nCol).length() + 2);
		String format = leftFormat + rightFormat + "%7s";
		
		for (int j = 0; j < M.getColumnDimension(); j++) {
			for (int i = 0; i < M.getRowDimension(); i++) {
				String valueString = "";
				double v = M.getEntry(i, j);
				if (v == 0)
					continue;
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = String.format("%.4f", v);
				else
					valueString = String.format("%d", rv);
				String leftString = String.format("(%d", i + 1);
				String rightString = String.format("%d)", j + 1);
				System.out.println(String.format(format, leftString, rightString, valueString));
			}
		}
		System.out.println();
		
	}
	
	/**
	 * Print a vector.
	 * 
	 * @param V a vector
	 * 
	 */
	public static void printVector(RealVector V) {
		
		if (V == null) {
			System.out.println("Empty vector!");
			return;
		}
		
		/*for (int i = 0; i < V.getDimension(); i++) {
			System.out.println(String.format("%.4f", V.getEntry(i)));
		}*/
		
		for (int i = 0; i < V.getDimension(); i++) {
			System.out.print("  ");
			String valueString = "";
			double v = V.getEntry(i);
			int rv = (int) Math.round(v);
			if (v != rv)
				valueString = String.format("%.4f", v);
			else
				valueString = String.format("%d", rv);
			System.out.print(String.format("%7s", valueString));
			System.out.print("  ");
			System.out.println();
		}
		System.out.println();
		
	}
	
	/**
	 * Print a vector with a specified precision.
	 * 
	 * @param V a vector
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printVector(RealVector V, int p) {
		
		if (V == null) {
			System.out.println("Empty vector!");
			return;
		}
		
		for (int i = 0; i < V.getDimension(); i++) {
			System.out.print("  ");
			String valueString = "";
			double v = V.getEntry(i);
			int rv = (int) Math.round(v);
			if (v != rv)
				valueString = String.format(sprintf("%%.%df", p), v);
			else
				valueString = String.format("%d", rv);
			System.out.print(String.format(sprintf("%%%ds", 7 + p - 4), valueString));
			System.out.print("  ");
			System.out.println();
		}
		System.out.println();
		
	}
	
	/**
	 * Write a formatted string to the standard output (the screen).
	 * 
	 * @param format a string in single quotation marks that
	 *        describes the format of the output fields
	 *        
	 * @param os argument list applying the format to
	 * 
	 */
	public static void fprintf(String format, Object... os) {
		System.out.format(format, os);
	}
	
	/**
	 * Format variables into a string.
	 * 
	 * @param format a string in single quotation marks that
	 *        describes the format of the output fields
	 *        
	 * @param os argument list applying the format to
	 * 
	 * @return a formatted string
	 * 
	 */
	public static String sprintf(String format, Object... os) {
		return String.format(format, os);
	}
	
	/**
	 * Calculate the l_2 norm of a vector or row vectors of a matrix.
	 * 
	 * @param A a vector or a matrix
	 * 
	 * @return the l_2 norm of a vecor or row vectors of a matrix
	 * 
	 */
	public static RealMatrix l2NormByRows(RealMatrix A) {
		
		RealMatrix res = null;
		if (A.getRowDimension() == 1) {
			res = new Array2DRowRealMatrix(1, 1);
			res.setEntry(0, 0, norm(A));
			
		} else {
			res = power(sum(power(A, 2), 2), 0.5);
		}
		return res;
		
	}

}
