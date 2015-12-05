package jml.clustering;

import java.util.ArrayList;

import jml.data.Data;
import jml.matlab.Matlab;
import jml.options.KMeansOptions;
import jml.options.L1NMFOptions;
import jml.options.Options;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation for L1NMF which solves the following
 * optimization problem:
 * <p>
 * min || X - G * F ||_F^2 + gamma * ||F||_{sav} + nu * ||G||_{sav}</br>
 * s.t. G >= 0, F >= 0
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 3rd, 2013
 */
public class L1NMF extends Clustering {

	public double epsilon;
	public int maxIter;

	public double gamma;
	public double mu;

	public boolean calc_OV;
	public boolean verbose;
	
	public ArrayList<Double> valueList;
	
	RealMatrix initializer = null;

	public L1NMF(Options options) {
		maxIter = options.maxIter;
		epsilon = options.epsilon;
		gamma = options.gamma;
		mu = options.mu;
		verbose = options.verbose;
		calc_OV = options.calc_OV;
		nClus = options.nClus;
	}
	
	public L1NMF(L1NMFOptions L1NMFOptions) {
		maxIter = L1NMFOptions.maxIter;
		epsilon = L1NMFOptions.epsilon;
		gamma = L1NMFOptions.gamma;
		mu = L1NMFOptions.mu;
		verbose = L1NMFOptions.verbose;
		calc_OV = L1NMFOptions.calc_OV;
		nClus = L1NMFOptions.nClus;
	}

	public L1NMF() {
		L1NMFOptions options = new L1NMFOptions();
		maxIter = options.maxIter;
		epsilon = options.epsilon;
		gamma = options.gamma;
		mu = options.mu;
		verbose = options.verbose;
		calc_OV = options.calc_OV;
		nClus = options.nClus;
	}
	
	public void initialize(RealMatrix G0) {
		
		if (G0 != null) {
			initializer = G0;
			return;
		}
		
		KMeansOptions kMeansOptions = new KMeansOptions();
		kMeansOptions.nClus = nClus;
		kMeansOptions.maxIter = 50;
		kMeansOptions.verbose = true;
		
		System.out.println("Using KMeans to initialize...");
		Clustering KMeans = new KMeans(kMeansOptions);
		KMeans.feedData(dataMatrix);
		// KMeans.initialize(null);
		KMeans.clustering();
		
		initializer = KMeans.getIndicatorMatrix();
		
	}
	
	@Override
	public void clustering() {
		if (initializer == null) {
			initialize(null);
			// initializer = indicatorMatrix;
		}
		clustering(initializer);
	}
	
	public void clustering(RealMatrix G0) {
		
		if (G0 == null) {
			initialize(null);
			G0 = initializer;
		}
		
		RealMatrix X = dataMatrix;
		RealMatrix G = G0;
		// RealMatrix F = X.multiply(G).multiply(new LUDecompositionImpl(G.transpose().multiply(G)).getSolver().getInverse());
		RealMatrix F = Matlab.mrdivide(X.multiply(G), G.transpose().multiply(G));
			
		ArrayList<Double> J = new ArrayList<Double>();
		RealMatrix F_pos = Matlab.subplus(F);
		F = F_pos.scalarAdd(0.2 * Matlab.sum(Matlab.sum(F_pos)).getEntry(0, 0)
				/ Matlab.find2(F_pos).get("row").getRowDimension());
		
		RealMatrix E_F = Matlab.ones(Matlab.size(F)).scalarMultiply(gamma / 2);
		RealMatrix E_G = Matlab.ones(Matlab.size(G)).scalarMultiply( mu / 2 );

		if (calc_OV) {
			J.add(f(X, F, G, E_F, E_G));
		}

		int ind = 0;
		RealMatrix G_old = new BlockRealMatrix(G.getRowDimension(), G.getColumnDimension());
		double d = 0;

		while (true) {
			
			G_old.setSubMatrix(G.getData(), 0, 0);
			
			// Fixing F, updating G
			G = UpdateG(X, F, mu, G);

			// Fixing G, updating F
			F = UpdateF(X, G, gamma, F);


			ind = ind + 1;
			if (ind > maxIter) {
				System.out.println("Maximal iterations");
				break;
			}

			d = Matlab.norm( G.subtract(G_old), "fro");

			if (calc_OV) {
				J.add(f(X, F, G, E_F, E_G));
			}

			if (ind % 10 == 0 && verbose) {
				if (calc_OV) {
					System.out.println(String.format("Iteration %d, delta G: %f, J: %f", ind, d, J.get(J.size() - 1)));
					// System.out.flush();
				} else {
					System.out.println(String.format("Iteration %d, delta G: %f", ind, d));
					// System.out.flush();
				}
			}

			if (calc_OV) {
				if (Math.abs(J.get(J.size() - 2) - J.get(J.size() - 1)) < epsilon && d < epsilon) {
					System.out.println("Converge successfully!");
					break;	
				}
			} else if (d < epsilon) {
				System.out.println("Converge successfully!");
				break;
			}

			if (Matlab.sum(Matlab.sum(Matlab.isnan(G))).getEntry(0, 0) > 0) {
				break;
			}

		}
		
		centers = F;
		indicatorMatrix = G;
		valueList = J;

	}

	private RealMatrix UpdateG(RealMatrix Y, RealMatrix X, double mu, RealMatrix A0) {

		// min|| Y - X * A^T ||_F^2 + mu * || A ||_1
		// s.t. A >= 0

		int MaxIter = 10000;
		double epsilon = 1e-1;

		int K = Matlab.size(X, 2);
		int NDoc = Matlab.size(Y, 2);

		RealMatrix YTX = Y.transpose().multiply(X);
		RealMatrix XTX = X.transpose().multiply(X);
		RealMatrix C = YTX.scalarMultiply(-1).scalarAdd(mu / 2);

		RealMatrix D = Matlab.repmat(Matlab.diag(XTX).transpose(), new int[]{NDoc, 1});
		RealMatrix A = A0;
		int ind = 0;
		double d = 0;

		RealMatrix A_old = new BlockRealMatrix(A.getRowDimension(), A.getColumnDimension());

		while (true) {

			A_old.setSubMatrix(A.getData(), 0, 0);
			for (int j = 0; j < K; j++) {

				A.setColumnMatrix(j, Matlab.max(
						A.getColumnMatrix(j).subtract(
								Matlab.ebeDivide(C.getColumnMatrix(j).add(A.multiply(XTX.getColumnMatrix(j))), D.getColumnMatrix(j))
						), 0d));

				// A(:, j) = max(A(:, j) - (C(:, j) + A * XTX(:, j)) ./ (D(:, j)), 0);
			}

			ind = ind + 1;
			if (ind > MaxIter) {
				break;
			}

			d = Matlab.sum(Matlab.sum(Matlab.abs(A.subtract(A_old)))).getEntry(0, 0);
			if (d < epsilon) {
				break;
			}

		}

		return A;

	}

	private RealMatrix UpdateF(RealMatrix Y, RealMatrix X, double gamma, RealMatrix A0) {

		// min|| Y - A * X^T ||_F^2 + gamma * || A ||_1
		// s.t. A >= 0

		int MaxIter = 10000;
		double epsilon = 1e-1;

		int K = Matlab.size(X, 2);
		int NTerm = Matlab.size(Y, 1);

		RealMatrix YX = Y.multiply(X);
		RealMatrix XTX = X.transpose().multiply(X);
		RealMatrix C = YX.scalarMultiply(-1).scalarAdd(gamma / 2);
		RealMatrix D = Matlab.repmat(Matlab.diag(XTX).transpose(), new int[]{NTerm, 1});
		RealMatrix A = A0;
		int ind = 0;
		double d = 0;
		RealMatrix A_old = new BlockRealMatrix(A.getRowDimension(), A.getColumnDimension());

		while (true) {

			A_old.setSubMatrix(A.getData(), 0, 0);

			for (int j = 0; j < K; j++) {

				A.setColumnMatrix(j, Matlab.max(
						A.getColumnMatrix(j).subtract(
								Matlab.ebeDivide(C.getColumnMatrix(j).add(A.multiply(XTX.getColumnMatrix(j))), D.getColumnMatrix(j))
						), 0d));

				// A(:, j) = max(A(:, j) - (C(:, j) + A * XTX(:, j)) ./ (D(:, j)), 0);
			}

			ind = ind + 1;
			if (ind > MaxIter) {
				break;
			}

			d = Matlab.sum(Matlab.sum(Matlab.abs(A.subtract(A_old)))).getEntry(0, 0);
			if (d < epsilon) {
				break;
			}

		}

		return A;

	}

	private double f(RealMatrix X, RealMatrix F, RealMatrix G, RealMatrix E_F, RealMatrix E_G) {
		return Math.pow(Matlab.norm( X.subtract(F.multiply(G.transpose())), "fro"), 2)
		+ 2 * Matlab.trace( E_F.transpose().multiply(F) )
		+ 2 * Matlab.trace( E_G.transpose().multiply(G) );
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		String dataMatrixFilePath = "CNN - DocTermCount.txt";
		
		long start = System.currentTimeMillis();
		RealMatrix X = Data.loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		X = Matlab.getTFIDF(X);
		X = Matlab.normalizeByColumns(X);
		
		KMeansOptions kMeansOptions = new KMeansOptions();
		kMeansOptions.nClus = 10;
		kMeansOptions.maxIter = 50;
		kMeansOptions.verbose = true;
		
		KMeans KMeans = new KMeans(kMeansOptions);
		KMeans.feedData(X);
		// KMeans.initialize(null);
		KMeans.clustering();
		
		RealMatrix G0 = KMeans.getIndicatorMatrix();
		
		// RealMatrix X = Data.loadSparseMatrix("X.txt");
		G0 = Data.loadDenseMatrix("G0.txt");
		L1NMFOptions L1NMFOptions = new L1NMFOptions();
		L1NMFOptions.nClus = 10;
		L1NMFOptions.gamma = 1 * 0.0001;
		L1NMFOptions.mu = 1 * 0.1;
		L1NMFOptions.maxIter = 50;
		L1NMFOptions.verbose = true;
		L1NMFOptions.calc_OV = !true;
		L1NMFOptions.epsilon = 1e-5;
		Clustering L1NMF = new L1NMF(L1NMFOptions);
		L1NMF.feedData(X);
		// L1NMF.initialize(G0);
		
		// Matlab takes 12.5413 seconds
		// jblas takes 29.368 seconds
		// Commons-math takes 129 seconds (Using Array2DRowRealMatrix)
		// Commons-math takes 115 seconds (Using BlockRealMatrix)
		// start = System.currentTimeMillis();
		
		L1NMF.clustering(G0); // Use null for random initialization
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
		
		Data.saveDenseMatrix("F.txt", L1NMF.centers);
		Data.saveDenseMatrix("G.txt", L1NMF.indicatorMatrix);
		
	}

}
