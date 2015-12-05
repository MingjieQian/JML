package jml.feature.selection;

import java.util.TreeMap;

import jml.classification.Classifier;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * Abstract class for supervised feature selection algorithms.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 4th, 2012
 */
public class SupervisedFeatureSelection extends FeatureSelection {

	/**
	 * An n x c label matrix.
	 */
	protected RealMatrix Y;
	
	/**
	 * Number of classes.
	 */
	private int nClass;
	
	public SupervisedFeatureSelection() {
		super();
	}
	
	/**
	 * Feed labels for this supervised feature selection algorithm.
	 * 
	 * @param Y an n x c label matrix
	 * 
	 */
	public void feedLabels(RealMatrix Y) {
		this.Y = Y;
		nClass = Y.getColumnDimension();
	}
	
	/**
	 * Feed labels for this supervised feature selection algorithm.
	 * 
	 * @param labels an n x c 2D {@code double} array
	 * 
	 */
	public void feedLabels(double[][] labels) {
		this.Y = new BlockRealMatrix(labels);
		nClass = Y.getColumnDimension();
	}
	
	/**
	 * Feed labels for this supervised feature selection algorithm.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 * 
	 */
	public void feedLabels(int[] labels) {
		nClass = Classifier.calcNumClass(labels);
		// int[] IDLabelMap = Classifier.getIDLabelMap(labels);
		TreeMap<Integer, Integer> labelIDMap = Classifier.getLabelIDMap(labels);
		int[] labelIDs = new int[labels.length];
		for (int i = 0; i < labels.length; i++) {
			labelIDs[i] = labelIDMap.get(labels[i]);
		}
		int[] labelIndices = labelIDs;
		Y = Classifier.labelIndexArray2LabelMatrix(labelIndices, nClass);
		/*this.labels = labels;
		this.labelIDs = labelIndices;*/
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public void run() {
	}
	
	/**
	 * Get label matrix.
	 * 
	 * @return an n x c label matrix
	 * 
	 */
	public RealMatrix getY() {
		return Y;
	}

}
