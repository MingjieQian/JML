package jml.classification;

import java.io.Serializable;
import java.util.TreeMap;

import jml.matlab.Matlab;
import jml.options.Options;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


/***
 * Abstract super class for all classifier subclasses.
 *  
 * @author Mingjie Qian
 * @version 1.0 Dec. 30th, 2012
 */
public abstract class Classifier implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7315629872244004590L;

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
	 * Number of samples.
	 */
	public int nExample;
	
	/**
	 * Training data matrix (nFeature x nExample),
	 * each column is a feature vector. The data 
	 * matrix should not include bias dummy features.
	 */
	public RealMatrix X;
	
	/**
	 * Label matrix for training (nExample x nClass).
	 * Y_{i,k} = 1 if x_i belongs to class k, and 0 otherwise.
	 */
	public RealMatrix Y;
	
	/**
	 * LabelID array for training data, starting from 0.
	 * The label ID array for the training data is latent,
	 * and we don't need to know them. They are only meaningful
	 * for reconstructing the integer labels by using IDLabelMap
	 * structure. 
	 */
	int[] labelIDs;
	
	/**
	 * Label array for training data with original integer code.
	 */
	int[] labels;
	
	/**
	 * Projection matrix (nFeature x nClass), column i is the projector for class i.
	 */
	public RealMatrix W;
	
	/**
	 * Convergence tolerance.
	 */
	public double epsilon;
	
	/**
	 * An ID to integer label mapping array. IDs start from 0.
	 */
	int[] IDLabelMap;
	
	/**
	 * Default constructor for a classifier.
	 */
	public Classifier() {
		nClass = 0;
		nFeature = 0;
		nExample = 0;
		X = null;
		Y = null;
		W = null;
		epsilon = 1e-4;
	}
	
	/**
	 * Constructor for a classifier initialized with options
	 * wrapped in a {@code Options} object.
	 * 
	 * @param options classification options
	 * 
	 */
	public Classifier(Options options) {
		nClass = 0;
		nFeature = 0;
		nExample = 0;
		X = null;
		Y = null;
		W = null;
		epsilon = options.epsilon;
	}
	
	/**
	 * Load the model for a classifier.
	 * 
	 * @param filePath file path to load the model
	 * 
	 */
	public abstract void loadModel(String filePath);
	
	/**
	 * Save the model for a classifier.
	 * 
	 * @param filePath file path to save the model
	 * 
	 */
	public abstract void saveModel(String filePath);
	
	/**
	 * Feed training data with original data matrix for this classifier.
	 * 
	 * @param X original data matrix without bias dummy features
	 * 
	 */
	public void feedData(RealMatrix X) {
		this.X = X;
		nFeature = X.getRowDimension();
		nExample = X.getColumnDimension();
	}
	
	/**
	 * Feed training data for this classification method.
	 * 
	 * @param data a d x n 2D {@code double} array with each
	 *             column being a data sample
	 */
	public void feedData(double[][] data) {
		feedData(new BlockRealMatrix(data));
	}
	
	/**
	 * Infer the number of classes from a given label sequence.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 *               
	 * @return number of classes
	 * 
	 */
	public static int calcNumClass(int[] labels) {
		
		TreeMap<Integer, Integer> IDLabelMap = new TreeMap<Integer, Integer>();
		int ID = 0;
		int label = -1;
		for (int i = 0; i < labels.length; i++) {
			label = labels[i];
			if (!IDLabelMap.containsValue(label)) {
				IDLabelMap.put(ID++, label);
			}
		}
		int nClass = IDLabelMap.size();
		return nClass;
	}
	
	/**
	 * Get an ID to integer label mapping array. IDs start from 0.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 *               
	 * @return ID to integer label mapping array
	 * 
	 */
	public static int[] getIDLabelMap(int[] labels) {
		TreeMap<Integer, Integer> IDLabelMap = new TreeMap<Integer, Integer>();
		int ID = 0;
		int label = -1;
		for (int i = 0; i < labels.length; i++) {
			label = labels[i];
			if (!IDLabelMap.containsValue(label)) {
				IDLabelMap.put(ID++, label);
			}
		}
		int nClass = IDLabelMap.size();
		int[] IDLabelArray = new int[nClass];
		for (int idx : IDLabelMap.keySet()) {
			IDLabelArray[idx] = IDLabelMap.get(idx);
		}
		return IDLabelArray;
	}
	
	/**
	 * Get a mapping from labels to IDs. IDs start from 0.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 *               
	 * @return a mapping from labels to IDs
	 * 
	 */
	public static TreeMap<Integer, Integer> getLabelIDMap(int[] labels) {
		TreeMap<Integer, Integer> labelIDMap = new TreeMap<Integer, Integer>();
		int ID = 0;
		int label = -1;
		for (int i = 0; i < labels.length; i++) {
			label = labels[i];
			if (!labelIDMap.containsKey(label)) {
				labelIDMap.put(label, ID++);
			}
		}
		return labelIDMap;
	}
	
	/**
	 * Feed labels of training data to the classifier.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 * 
	 */
	public void feedLabels(int[] labels) {
		nClass = calcNumClass(labels);
		IDLabelMap = getIDLabelMap(labels);
		TreeMap<Integer, Integer> labelIDMap = getLabelIDMap(labels);
		int[] labelIDs = new int[labels.length];
		for (int i = 0; i < labels.length; i++) {
			labelIDs[i] = labelIDMap.get(labels[i]);
		}
		int[] labelIndices = labelIDs;
		Y = labelIndexArray2LabelMatrix(labelIndices, nClass);
		this.labels = labels;
		this.labelIDs = labelIndices;
	}
	
	/**
	 * Feed labels for training data from a matrix.
	 * Note that if we feed the classifier with only
	 * label matrix, then we don't have original integer
	 * labels actually. In this case, label IDs will be
	 * inferred according to the label matrix. The first
	 * observed label index will be assigned ID 0, the second
	 * observed label index will be assigned ID 1, and so on.
	 * And labels will be the label indices in the given
	 * label matrix
	 * 
	 * @param Y an N x K label matrix, where N is the number of
	 *          training samples, and K is the number of classes
	 * 
	 */
	public void feedLabels(RealMatrix Y) {
		this.Y = Y;
		nClass = Y.getColumnDimension();
		if (nExample != Y.getRowDimension()) {
			System.err.println("Number of labels error!");
			System.exit(1);
		}
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Y);
		labels = labelIndices;
		IDLabelMap = getIDLabelMap(labels);
		labelIDs = labelIndices;
	}
	
	/**
	 * Feed labels for this classification method.
	 * 
	 * @param labels an n x c 2D {@code double} array
	 * 
	 */
	public void feedLabels(double[][] labels) {
		feedLabels(new BlockRealMatrix(labels));
	}
	
	/**
	 * Train the classifier.
	 */
	public abstract void train();
	
	/**
	 * Predict the labels for the test data formated as an original data matrix.
	 * The original data matrix should not include bias dummy features.
	 * 
	 * @param Xt test data matrix with each column being a feature vector
	 * 
	 * @return predicted label array with original integer label code
	 * 
	 */
	public int[] predict(RealMatrix Xt) {
		RealMatrix Yt = predictLabelScoreMatrix(Xt);
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
	
	/**
	 * Predict the labels for the test data formated as an original 2D
	 * {@code double} array. The original data matrix should not
	 * include bias dummy features.
	 * 
	 * @param Xt a d x n 2D {@code double} array with each
	 *           column being a data sample
	 *           
	 * @return predicted label array with original integer label code
	 * 
	 */
	public int[] predict(double[][] Xt) {
		return predict(new BlockRealMatrix(Xt));
	}
	
	/**
	 * Predict the label matrix given test data formated as an
	 * original data matrix.
	 * 
	 * Note that if a method of an abstract class is declared as
	 * abstract, it is implemented as an interface function in Java.
	 * Thus subclasses need to implement this abstract method rather
	 * than to override it.
	 * 
	 * @param Xt test data matrix with each column being a feature vector
	 * 
	 * @return predicted N x K label matrix, where N is the number of
	 *         test samples, and K is the number of classes
	 * 
	 */
	public RealMatrix predictLabelMatrix(RealMatrix Xt) {
		RealMatrix Yt = predictLabelScoreMatrix(Xt);
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
		return labelIndexArray2LabelMatrix(labelIndices, nClass);
	}
	
	/**
	 * Predict the label matrix given test data formated as an
	 * original 2D {@code double} array.
	 * 
	 * @param Xt a d x n 2D {@code double} array with each
	 *           column being a data sample
	 *           
	 * @return predicted N x K label matrix, where N is the number of
	 *         test samples, and K is the number of classes
	 *         
	 */
	public RealMatrix predictLabelMatrix(double[][] Xt) {
		return predictLabelMatrix(new BlockRealMatrix(Xt));
	}
	
	/**
	 * Predict the label score matrix given test data formated as an
	 * original data matrix.
	 * 
	 * Note that if a method of an abstract class is declared as
	 * abstract, it is implemented as an interface function in Java.
	 * Thus subclass needs to implement this abstract method rather
	 * than to override it.
	 * 
	 * @param Xt test data matrix with each column being a feature vector
	 * 
	 * @return predicted N x K label score matrix, where N is the number of
	 *         test samples, and K is the number of classes
	 * 
	 */
	public abstract RealMatrix predictLabelScoreMatrix(RealMatrix Xt);
	
	/**
	 * Predict the label score matrix given test data formated as an
	 * original data matrix.
	 * 
	 * @param Xt a d x n 2D {@code double} array with each
	 *           column being a data sample
	 *           
	 * @return predicted N x K label score matrix, where N is the number of
	 *         test samples, and K is the number of classes
	 *         
	 */
	public RealMatrix predictLabelScoreMatrix(double[][] Xt) {
		return predictLabelScoreMatrix(new BlockRealMatrix(Xt));
	}
	
	/**
	 * Get accuracy for a classification task.
	 * 
	 * @param pre_labels predicted labels
	 * 
	 * @param labels true labels
	 * 
	 * @return accuracy
	 * 
	 */
	public static double getAccuracy(int[] pre_labels, int[] labels) {
		if (pre_labels.length != labels.length) {
			System.err.println("Number of predicted labels " +
					"and number of true labels mismatch.");
			System.exit(1);
		}
		int N = labels.length;
		int cnt_correct = 0;
		for ( int i = 0; i < N; i ++ ) {
			if ( pre_labels[i] == labels[i] )
				cnt_correct ++;
		}
		double accuracy = (double)cnt_correct / (double)N;
		System.out.println(String.format("Accuracy: %.2f%%\n", accuracy * 100));
		return accuracy;
	}
	
	/**
	 * Get projection matrix for this classifier.
	 * 
	 * @return a d x c projection matrix
	 * 
	 */
	public RealMatrix getProjectionMatrix() {
		return W;
	}
	
	/**
	 * Get ground truth label matrix for training data.
	 * 
	 * @return an n x c label matrix
	 * 
	 */
	public RealMatrix getTrainingLabelMatrix() {
		return Y;
	}
	
	/**
	 * Convert label array to label matrix. Label indices start from 0.
	 * 
	 * @param labels label array with original integer code
	 * 
	 * @param nClass number of classes
	 * 
	 * @return label matrix
	 * 
	 *//*
	protected static RealMatrix labelArray2LabelMatrix(int[] labels, int nClass) {
		RealMatrix Y = new OpenMapRealMatrix(labels.length, nClass);
		for (int i = 0; i < labels.length; i++) {
			Y.setEntry(i, labels[i], 1);
		}
		return Y;
	}*/
	
	/**
	 * Convert a label matrix to a label index array. Label indices start from 0.
	 * 
	 * @param Y label matrix
	 * 
	 * @return a label index array
	 * 
	 */
	public static int[] labelScoreMatrix2LabelIndexArray(RealMatrix Y) {
		double[] IDs = Matlab.max(Y, 2).get("idx").getColumn(0);
		int[] labelIndices = new int[IDs.length];
		for (int i = 0; i < IDs.length; i++) {
			labelIndices[i] = (int)IDs[i];
		}
		return labelIndices;
	}
	
	/**
	 * Convert a label index array to a label matrix. Label indices start from 0.
	 * 
	 * @param labelIndices a label index array
	 * 
	 * @param nClass number of classes
	 * 
	 * @return label matrix
	 * 
	 */
	public static RealMatrix labelIndexArray2LabelMatrix(int[] labelIndices, int nClass) {
		RealMatrix Y = new OpenMapRealMatrix(labelIndices.length, nClass);
		for (int i = 0; i < labelIndices.length; i++) {
			Y.setEntry(i, labelIndices[i], 1);
		}
		return Y;
	}

}
