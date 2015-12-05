package jml.classification;

import static jml.matlab.Matlab.log;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.sigmoid;
import static jml.matlab.Matlab.sum;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.zeros;
import static jml.matlab.Matlab.logicalIndexing;
import static jml.matlab.Matlab.display;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.full;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import jml.optimization.LBFGS;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation for the maximum entropy modeling.
 * We aim to maximize the log-likelihood of p(y_n|x_n) for 
 * {(x_n, y_n)|n = 1, 2, ..., N}, which can be written as 
 * L(X) = sum_n [log P(y_n|x_n)] / N.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 18th, 2013
 */
public class MaxEnt extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -316339495680314422L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		long start = System.currentTimeMillis();
		
		/*
		 * a 3D {@code double} array, where data[n][i][k]
	     * is the i-th feature value on the k-th class 
	     * for the n-th sample
		 */
		double[][][] data = new double[][][] {
				{{1, 0, 0}, {2, 1, -1}, {0, 1, 2}, {-1, 2, 1}},
				{{0, 2, 0}, {1, 0, -1}, {0, 1, 1}, {-1, 3, 0.5}},
				{{0, 0, 0.8}, {2, 1, -1}, {1, 3, 0}, {-0.5, -1, 2}},
				{{0.5, 0, 0}, {1, 1, -1}, {0, 0.5, 1.5}, {-2, 1.5, 1}},
		};
		
		/*double [][] labels = new double[][] { 
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
				{1, 0, 0}
		};*/
		int[] labels = new int[] {1, 2, 3, 1};
		
		MaxEnt maxEnt = new MaxEnt();
		maxEnt.feedData(data);
		maxEnt.feedLabels(labels);
		maxEnt.train();
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
		
		fprintf("MaxEnt parameters:\n");
		display(maxEnt.W);
		String modelFilePath = "MaxEnt-Model.dat";
		maxEnt.saveModel(modelFilePath);
		
		maxEnt = new MaxEnt();
		maxEnt.loadModel(modelFilePath);
		fprintf("Predicted probability matrix:\n");
		display(maxEnt.predictLabelScoreMatrix(data));
		fprintf("Predicted label matrix:\n");
		display(full(maxEnt.predictLabelMatrix(data)));
		fprintf("Predicted labels:\n");
		display(maxEnt.predict(data));

	}
	
	/**
	 * Feature matrix array. F[n] is the feature matrix for
	 * the n-th sample, where F[n][i][k] is the i-th feature
	 * value on the k-th class for the n-th sample.
	 */
	private RealMatrix[] F;
	
	/**
	 * Feed features for this maximum entropy model.
	 * 
	 * @param data a 3D {@code double} array, where data[n][i][k]
	 * 			   is the i-th feature value on the k-th class 
	 *             for the n-th sample
	 *             
	 */
	public void feedData(double[][][] data) {
		this.F = new RealMatrix[data.length];
		for (int n = 0; n < data.length; n++) {
			this.F[n] = new BlockRealMatrix(data[n]);
		}
		nExample = data.length;
		nFeature = data[0].length;
		nClass = data[0][0].length;
	}
	
	/**
	 * Feed features for this maximum entropy model.
	 * 
	 * @param F a feature matrix array. F[n] is the feature 
	 *          matrix for the n-th sample, where F[n][i][k]
	 *          is the i-th feature value on the k-th class 
	 *          for the n-th sample
	 *          
	 */
	public void feedData(RealMatrix[] F) {
		this.F = F;
		nExample = F.length;
		nFeature = F[0].getRowDimension();
		nClass = F[0].getColumnDimension();
	}

	@Override
	public void train() {
		
		RealMatrix Grad = null;
		RealMatrix A = null;
		RealMatrix V = null;
		RealMatrix G = null;
		
		double fval = 0;
		
		/*
		 * Minimize the negative log-likelihood
		 * L(X) = -sum_n [log P(y_n|x_n)] / N.
		 * Grad = sum_n {E_{P(y|x_n)}[F_{:,y}(x_n)] - F_{:, y_n}(x_n)} / N
		 *      = sum_n {{F(x_n) * s(x_n)} - F_{:, y_n}(x_n)} / N
		 */
		A = new BlockRealMatrix(nExample, nClass);
		W = zeros(nFeature, 1);
		for (int n = 0; n < nExample; n++){
			A.setRowMatrix(n, W.transpose().multiply(F[n]));
		}
		V = sigmoid(A);
		
		for (int n = 0; n < nExample; n++){
			G = rdivide(minus(mtimes(F[n], V.getRowMatrix(n).transpose()), F[n].getColumnMatrix(labelIDs[n])), nExample);
			if (n == 0)
				Grad = G;
			else
				Grad = plus(Grad, G);
		}
		
		fval = -sum(log(logicalIndexing(V, Y)), 1).getEntry(0, 0) / nExample;
		/*for (int i = 0; i < nSample; i++){
			v = Math.log(V.getEntry(i, labelIDs[i])) / nSample;
			if (i == 0)
				fval = v;
			else
				fval += v;
		}*/
		
		boolean flags[] = null;
		while (true) {
			flags = LBFGS.run(Grad, fval, epsilon, W);
			if (flags[0])
				break;
			for (int n = 0; n < nExample; n++){
				A.setRowMatrix(n, W.transpose().multiply(F[n]));
			}
			V = sigmoid(A);
			fval = -sum(log(logicalIndexing(V, Y)), 1).getEntry(0, 0) / nExample;
			if (flags[1]) {
				for (int n = 0; n < nExample; n++){
					G = rdivide(minus(mtimes(F[n], V.getRowMatrix(n).transpose()), F[n].getColumnMatrix(labelIDs[n])), nExample);
					if (n == 0)
						Grad = G;
					else
						Grad = plus(Grad, G);
				}
			}
		}
		
	}
	
	/**
	 * Predict labels for the test data formated as a 3D
	 * {@code double} array.
	 * 
	 * @param data a 3D {@code double} array, where data[n][i][k]
	 * 			   is the i-th feature value on the k-th class 
	 *             for the n-th sample
	 *             
	 * @return predicted label array with original integer label code
	 * 
	 */
	public int[] predict(double[][][] data) {
		int Nt = data.length;
		RealMatrix[] Ft = new RealMatrix[Nt];
		for (int n = 0; n < Nt; n++) {
			Ft[n] = new BlockRealMatrix(data[n]);
		}
		return predict(Ft);
	}
	
	/**
	 * Predict labels for the test data formated as a 1D
	 * {@code RealMatrix} array.
	 * 
	 * @param Ft a feature matrix array. F[n] is the feature 
	 *           matrix for the n-th sample, where F[n][i][k]
	 *           is the i-th feature value on the k-th class 
	 *           for the n-th sample
	 *             
	 * @return predicted label array with original integer label code
	 * 
	 */
	public int[] predict(RealMatrix[] Ft) {
		RealMatrix Yt = predictLabelScoreMatrix(Ft);
		/*
		 * Because column vectors of W are arranged according to the 
		 * order of observation of class labels, in this case, label
		 * indices predicted from the label score matrix are identical
		 * to the latent label IDs, and labels can be inferred by the
		 * IDLabelMap structure.
		 */
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
		int[] labels = new int[labelIndices.length];
		for (int i = 0; i < labelIndices.length; i++) {
			labels[i] = IDLabelMap[labelIndices[i]];
		}
		return labels;
	}

	@Override
	public void loadModel(String filePath) {
		
		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			MaxEntModel MaxEntModel = (MaxEntModel)ois.readObject();
			nClass = MaxEntModel.nClass;
			W = MaxEntModel.W;
			IDLabelMap = MaxEntModel.IDLabelMap;
			nFeature = MaxEntModel.nFeature;
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
	}

	@Override
	public void saveModel(String filePath){

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(new MaxEntModel(nClass, W, IDLabelMap));
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	@Override
	public RealMatrix predictLabelScoreMatrix(RealMatrix Ft) {
		return null;
	}
	
	/**
	 * Predict the label score matrix given test data formated as a
	 * {@code RealMatrix} array.
	 * 
	 * @param Ft a feature matrix array. F[n] is the feature 
	 *           matrix for the n-th sample, where F[n][i][k]
	 *           is the i-th feature value on the k-th class 
	 *           for the n-th sample
	 *           
	 * @return predicted N x K label score matrix, where N is 
	 *         the number of test examples, and K is the number 
	 *         of classes
	 *         
	 */
	public RealMatrix predictLabelScoreMatrix(RealMatrix[] Ft) {
		int Nt = Ft.length;
		int K = Ft[0].getColumnDimension();
		RealMatrix A = null;
		RealMatrix V = null;
		A = new BlockRealMatrix(Nt, K);
		for (int i = 0; i < Nt; i++){
			A.setRowMatrix(i, W.transpose().multiply(Ft[i]));
		}
		V = sigmoid(A);
		return V;
	}
	
	/**
	 * Predict the label score matrix given test data formated as a
	 * 3D {@code double} array.
	 * 
	 * @param data a 3D {@code double} array, where data[n][i][k]
	 * 			   is the i-th feature value on the k-th class 
	 *             for the n-th sample
	 *           
	 * @return predicted N x K label score matrix, where N is 
	 *         the number of test examples, and K is the number 
	 *         of classes
	 *         
	 */
	public RealMatrix predictLabelScoreMatrix(double[][][] data) {
		int Nt = data.length;
		RealMatrix[] Ft = new RealMatrix[Nt];
		for (int n = 0; n < Nt; n++) {
			Ft[n] = new BlockRealMatrix(data[n]);
		}
		return predictLabelScoreMatrix(Ft);
	}
	
	/**
	 * Predict the label matrix given test data formated as a 1D
	 * {@code RealMatrix} array.
	 * 
	 * @param Ft a feature matrix array. F[n] is the feature 
	 *           matrix for the n-th sample, where F[n][i][k]
	 *           is the i-th feature value on the k-th class 
	 *           for the n-th sample
	 * 
	 * @return predicted N x K label matrix, where N is the number of
	 *         test examples, and K is the number of classes
	 * 
	 */
	public RealMatrix predictLabelMatrix(RealMatrix[] Ft) {
		RealMatrix Yt = predictLabelScoreMatrix(Ft);
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
		// int nClass = Ft[0].getColumnDimension();
		return labelIndexArray2LabelMatrix(labelIndices, nClass);
	}
	
	/**
	 * Predict the label matrix given test data formated as a
	 * 3D {@code double} array.
	 * 
	 * @param data a 3D {@code double} array, where data[n][i][k]
	 * 			   is the i-th feature value on the k-th class 
	 *             for the n-th sample
	 *           
	 * @return predicted N x K label matrix, where N is the number of
	 *         test examples, and K is the number of classes
	 *         
	 */
	public RealMatrix predictLabelMatrix(double[][][] data) {
		int Nt = data.length;
		RealMatrix[] Ft = new RealMatrix[Nt];
		for (int n = 0; n < Nt; n++) {
			Ft[n] = new BlockRealMatrix(data[n]);
		}
		return predictLabelMatrix(Ft);
	}

}

/***
 * Maximum entropy model parameters.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 18th, 2013
 */
class MaxEntModel implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8767272469004168519L;

	/**
	 * Number of classes.
	 */
	public int nClass;
	
	/**
	 * Number of features, without bias dummy features,
	 * i.e., for SVM.
	 */
	public int nFeature;
	
	/**
	 * An nFeature x 1 {@code RealMatrix}.
	 */
	RealMatrix W;
	
	/**
	 * An ID to integer label mapping array. IDs start from 0.
	 */
	int[] IDLabelMap;
	
	/**
	 * Constructor for a MaxEnt model.
	 * 
	 * @param nClass number of classes
	 * 
	 * @param W an nFeature x 1 {@code RealMatrix}
	 * 
	 * @param IDLabelMap an ID to integer label mapping array 
	 *        where IDs start from 0
	 *        
	 */
	public MaxEntModel(int nClass, RealMatrix W, int[] IDLabelMap) {
		this.nClass = nClass;
		this.W = W;
		this.IDLabelMap = IDLabelMap;
		this.nFeature = W.getRowDimension();
	}
	
}
