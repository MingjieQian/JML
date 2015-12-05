package jml.clustering;

import static jml.matlab.Matlab.getTFIDF;
import static jml.matlab.Matlab.normalizeByColumns;

import java.util.TreeMap;

import jml.data.Data;
import jml.matlab.Matlab;
import jml.options.KMeansOptions;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation for KMeans.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 3rd, 2013
 */
public class KMeans extends Clustering {

	KMeansOptions options;
	
	public KMeans(int nClus) {
		super(nClus);
		options.maxIter = 100;
		options.verbose = false;
	}
	
	public KMeans(int nClus, int maxIter) {
		super(nClus);
		options.maxIter = maxIter;
		options.verbose = false;
	}
	
	public KMeans(int nClus, int maxIter, boolean verbose) {
		super(nClus);
		options.maxIter = maxIter;
		options.verbose = verbose;
	}
	
	public KMeans(KMeansOptions options) {
		super(options.nClus);
		/*if (options.nClus == -1) {
			System.err.println("Number of clusters undefined!");
			System.exit(1);
		} else if (options.nClus == 0) {
			System.err.println("Number of clusters is zero!");
			System.exit(1);
		}*/
		this.options = options;
	}
	
	/*@Override
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
		
	}*/
	
	/**
	 * Initializer needs not be explicitly specified. If the initial
	 * indicator matrix is not given, random initialization will be
	 * used.
	 */
	@Override
	public void clustering() {
		
		if (indicatorMatrix == null) {
			initialize(null);
		}
		
		int cnt = 0;
		// indicatorMatrix = new OpenMapRealMatrix(nSample, nClus);
		
		RealMatrix DistMatrix = null;
		double mse = 0;
		
		TreeMap<String, RealMatrix> minResult = null;
		RealMatrix minMatrix = null;
		RealMatrix idxMatrix = null;
		
		/*List<Integer> indList = new ArrayList<Integer>();
		for (int i = 0; i < nSample; i++) {
			indList.add(i);
		}*/
		
		/*Random rdn = new Random(System.currentTimeMillis());
		Collections.shuffle(indList, rdn);*/
		
		/*for (int i = 0; i < nClus; i++) {
			indicatorMatrix.setEntry(indList.get(i), i, 1);
		}*/
		
		while (cnt < options.maxIter) {
			
			RealMatrix indOld = indicatorMatrix;
			
			long start = System.currentTimeMillis();
			
			centers = dataMatrix.multiply(indicatorMatrix).multiply(
					Matlab.diag(Matlab.ones(options.nClus, 1).getColumnVector(0).ebeDivide(
							Matlab.diag(indicatorMatrix.transpose().multiply(indicatorMatrix)).getColumnVector(0))));
			
			DistMatrix = Matlab.l2DistanceSquare(dataMatrix, centers);
			// Matlab.disp(DistMatrix.transpose());
			
			minResult = Matlab.min(DistMatrix, 2);
			minMatrix = minResult.get("val");
			idxMatrix = minResult.get("idx");
			// Data.saveMatrix("indicators", indicatorMatrix);
			indicatorMatrix = new OpenMapRealMatrix(nSample, nClus);
			for (int i = 0; i < nSample; i++) {
				indicatorMatrix.setEntry(i, (int)idxMatrix.getEntry(i, 0), 1);
			}			
			
			mse = Matlab.sum(minMatrix, 1).getTrace() / nSample;
			
			// Debug
			/*if (Double.isNaN(sse)) {
				int a = 1;
				a = a + 1;
				Matlab.display(DistMatrix);
				Matlab.display(dataMatrix.getColumnMatrix(6));
				Matlab.display(centers.getColumnMatrix(0));
				Matlab.display(Matlab.norm(dataMatrix.getColumnMatrix(6).subtract(centers.getColumnMatrix(0))));
				Matlab.display(Matlab.l2Distance(dataMatrix.getColumnMatrix(6), centers));
				Matlab.display(Matlab.l2DistanceSquare(dataMatrix.getColumnMatrix(6), centers));
			}*/
			
			if (indOld.subtract(indicatorMatrix).getFrobeniusNorm() == 0) {
				System.out.println("KMeans complete.");
				break;
			}
			
			double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
			
			cnt += 1;
			
			if (options.verbose) {
				System.out.format("Iter %d: mse = %.3f (%.3f secs)\n", cnt, mse, elapsedTime);
			}
			
		}
		
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		runKMeans();

		int K = 3;
		int maxIter = 100;
		boolean verbose = true;
		KMeansOptions options = new KMeansOptions(K, maxIter, verbose);
		Clustering KMeans = new KMeans(options);
		
		//double[][] matrixData = { {1d, 0d, 3d}, {2d, 5d, 3d}, {4d, 1d, 0d}, {3d, 0d, 1d}, {2d, 5d, 3d}};
		double[][] matrixData2 = { {1d, 0d, 3d, 2d, 0d}, 
				                   {2d, 5d, 3d, 1d, 0d}, 
				                   {4d, 1d, 0d, 0d, 1d}, 
				                   {3d, 0d, 1d, 0d, 2d}, 
				                   {2d, 5d, 3d, 1d, 6d} };
		
		RealMatrix dataMatrix = new BlockRealMatrix(matrixData2);
		Matlab.printMatrix(dataMatrix);
		
		RealMatrix X = Data.loadMatrix("CNNTest-TrainingData.txt");
		RealMatrix X2 = normalizeByColumns(getTFIDF(X));
		KMeans.feedData(X2);
		RealMatrix initializer = Data.loadMatrix("indicators");
		initializer = null;
		KMeans.initialize(initializer);
		KMeans.clustering();
		
		/*System.out.println("Input data matrix:");
		Matlab.printMatrix(dataMatrix);*/
		
		System.out.println("Indicator Matrix:");
		Matlab.printMatrix(Matlab.full(KMeans.getIndicatorMatrix()));
		
	}
	
	public static void runKMeans() {
        
		double[][] data = { {3.5, 4.4, 1.3},
                {5.3, 2.2, 0.5},
                {0.2, 0.3, 4.1},
                {-1.2, 0.4, 3.2} };
		
		KMeansOptions options = new KMeansOptions();
        options.nClus = 2;
        options.verbose = true;
        options.maxIter = 100;
    
        KMeans KMeans= new KMeans(options);
        
        KMeans.feedData(data);
        // KMeans.initialize(null);
        RealMatrix initializer = null;
        initializer = new OpenMapRealMatrix(3, 2);
        initializer.setEntry(0, 0, 1);
        initializer.setEntry(1, 1, 1);
        initializer.setEntry(2, 0, 1);
        KMeans.clustering(initializer); // Use null for random initialization
        
        System.out.println("Indicator Matrix:");
		Matlab.printMatrix(Matlab.full(KMeans.getIndicatorMatrix()));
		
    }

}
