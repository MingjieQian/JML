package jml.clustering;

import jml.data.Data;
import jml.matlab.Matlab;
import jml.options.KMeansOptions;
import jml.options.NMFOptions;
import jml.options.Options;

import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation for NMF which solves the following
 * optimization problem:
 * <p>
 * min || X - G * F ||_F^2</br>
 * s.t. G >= 0, F >= 0
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 3rd, 2013
 */
public class NMF extends L1NMF {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		runNMF();

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
		KMeans.initialize(null);
		KMeans.clustering();
		
		RealMatrix G0 = KMeans.getIndicatorMatrix();
		
		// RealMatrix X = Data.loadSparseMatrix("X.txt");
		// RealMatrix G0 = Data.loadDenseMatrix("G0.txt");
		NMFOptions NMFOptions = new NMFOptions();
		NMFOptions.maxIter = 300;
		NMFOptions.verbose = true;
		NMFOptions.calc_OV = false;
		NMFOptions.epsilon = 1e-5;
		Clustering NMF = new NMF(NMFOptions);
		NMF.feedData(X);
		NMF.initialize(G0);
		
		// Matlab takes 12.5413 seconds
		// jblas takes 29.368 seconds
		// Commons-math takes 129 seconds (Using Array2DRowRealMatrix)
		// Commons-math takes 115 seconds (Using BlockRealMatrix)
		// start = System.currentTimeMillis();
		
		NMF.clustering();
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
		
		Data.saveDenseMatrix("F.txt", NMF.centers);
		Data.saveDenseMatrix("G.txt", NMF.indicatorMatrix);
		
	}
	
	public NMF(Options options) {
		super(options);
		gamma = 0;
		mu = 0;
	}
	
	public NMF(NMFOptions NMFOptions) {
		nClus = NMFOptions.nClus;
		maxIter = NMFOptions.maxIter;
		epsilon = NMFOptions.epsilon;
		verbose = NMFOptions.verbose;
		calc_OV = NMFOptions.calc_OV;
		gamma = 0;
		mu = 0;
	}
	
	public NMF() {
		Options options = new Options();
		nClus = options.nClus;
		maxIter = options.maxIter;
		epsilon = options.epsilon;
		verbose = options.verbose;
		calc_OV = options.calc_OV;
		gamma = 0;
		mu = 0;
	}
	
	public static void runNMF() {
		
		double[][] data = { {3.5, 4.4, 1.3},
                {5.3, 2.2, 0.5},
                {0.2, 0.3, 4.1},
                {1.2, 0.4, 3.2} };
        
		KMeansOptions options = new KMeansOptions();
        options.nClus = 2;
        options.verbose = true;
        options.maxIter = 100;
    
        KMeans KMeans= new KMeans(options);

        KMeans.feedData(data);
        KMeans.initialize(null);
        KMeans.clustering();
        RealMatrix G0 = KMeans.getIndicatorMatrix();
        
        NMFOptions NMFOptions = new NMFOptions();
        NMFOptions.nClus = 2;
		NMFOptions.maxIter = 50;
		NMFOptions.verbose = true;
		NMFOptions.calc_OV = false;
		NMFOptions.epsilon = 1e-5;
		Clustering NMF = new NMF(NMFOptions);
		
		NMF.feedData(data);
		// NMF.initialize(null);
		NMF.clustering(G0); // If null, KMeans will be used for initialization
		
		System.out.println("Basis Matrix:");
		Matlab.printMatrix(Matlab.full(NMF.getCenters()));
		
		System.out.println("Indicator Matrix:");
		Matlab.printMatrix(Matlab.full(NMF.getIndicatorMatrix()));
		
	}

}
